"""
Fine-tune (or train from scratch) the looped CrackViT on multi-class benchmarks:
CUB-200, FGVC-Aircraft, Flowers-102, Food-101, CIFAR-100, or any <class>/*.jpg folder.

Recipe (see README.md): initialise the looped blocks from a pretrained ViT (timm DeiT-S),
fine-tune with layer-wise LR decay + mixup/cutmix + RandAugment, and compare against
controlled baselines run with the SAME script / resolution / epochs / augmentation:

    # ours: looped 6x2 GIPA, DeiT-S init (layers i and i+6 averaged)
    python train_finetune.py --dataset-dir datasets/cub200
    # plain ViT, no looping  (12 untied blocks == DeiT-S)
    python train_finetune.py --dataset-dir datasets/cub200 --attention MHSA --n-core 12 --n-passes 1 --pass-embed false
    # parameter-matched shallow ViT (6 untied blocks)
    python train_finetune.py --dataset-dir datasets/cub200 --attention MHSA --n-core 6 --n-passes 1 --pass-embed false
    # looped, plain attention (isolates GIPA from looping)
    python train_finetune.py --dataset-dir datasets/cub200 --attention MHSA
    # from-scratch experiment (CIFAR-100, equal compute)
    python train_finetune.py --config config_scratch_cifar.yaml --dataset-dir datasets/cifar100

Outputs -> runs_bench/<dataset>/<run>/ : best.pt last.pt log.csv history.json results.json results.csv
per_exit.csv (accuracy vs loop count) early_exit.csv per_class.txt confusion_matrix.csv gates.csv curves.png
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

import metrics as M
from data import build_folder_loaders
from model import CrackViTConfig, LoopedCrackViT
from train import (EarlyStopping, complexity, cosine_lr, get_args, param_groups, pick_device, seed_all, tqdm)


# ------------------------------------------------------------------ mixup / cutmix
class MixupCutmix:
    """Batch-level mixup OR cutmix; returns mixed images and soft targets (with label smoothing)."""

    def __init__(self, mixup_alpha, cutmix_alpha, prob, switch_prob, smoothing, num_classes):
        self.ma, self.ca, self.prob, self.sw, self.ls, self.C = mixup_alpha, cutmix_alpha, prob, switch_prob, smoothing, num_classes

    def onehot(self, y, lam=1.0):
        off = self.ls / self.C
        t = torch.full((len(y), self.C), off, device=y.device)
        return t.scatter_(1, y[:, None], 1.0 - self.ls + off)

    def __call__(self, x, y):
        t = self.onehot(y)
        if np.random.rand() > self.prob:
            return x, t
        use_cut = self.ca > 0 and (self.ma <= 0 or np.random.rand() < self.sw)
        lam = float(np.random.beta(*(2 * [self.ca if use_cut else self.ma])))
        perm = torch.randperm(x.shape[0], device=x.device)
        if use_cut:
            H, W = x.shape[-2:]
            r = math.sqrt(1 - lam)
            ch, cw = int(H * r), int(W * r)
            cy, cx = np.random.randint(H), np.random.randint(W)
            y1, y2, x1, x2 = max(cy - ch // 2, 0), min(cy + ch // 2, H), max(cx - cw // 2, 0), min(cx + cw // 2, W)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
            lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
        else:
            x = lam * x + (1 - lam) * x[perm]
        return x, lam * t + (1 - lam) * t[perm]


def soft_ce(outs, target, exit_w):
    return sum(w * (-(target * F.log_softmax(o.float(), -1)).sum(-1)).mean() for o, w in zip(outs, exit_w))


@torch.no_grad()
def predict(model, loader, device, amp_dtype, use_amp, desc=None):
    model.eval()
    ys, logits = [], [[] for _ in model.exit_names]
    for x, y in tqdm(loader, desc=desc, leave=False, disable=desc is None, dynamic_ncols=True):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            outs = model(x.to(device, non_blocking=True))
        for i, o in enumerate(outs):
            logits[i].append(o.float().cpu())
        ys.append(y)
    y = torch.cat(ys)
    lg = [torch.cat(l) for l in logits]
    nll = F.cross_entropy(lg[-1], y).item()
    return y.numpy().astype(int), {n: F.softmax(l, -1).numpy() for n, l in zip(model.exit_names, lg)}, nll


def bootstrap_acc(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    hit = (y == pred).astype(float)
    b = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    args = get_args("config_finetune.yaml")
    seed_all(args.seed)
    device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dataset = os.path.basename(os.path.normpath(args.dataset_dir or args.train_dir))
    init = f"ft-{args.init_strategy}" if args.backbone else "scratch"
    name = (f"{args.attention.lower()}_{args.n_prelude}-{args.n_core}x{args.n_passes}-{args.n_coda}"
            f"_{init}_r{args.image_size}_s{args.seed}")
    out = os.path.join(args.output_dir, dataset, name)

    # ---- data (needed first: it defines num_classes) ----
    if args.summary_only:
        classes = [f"c{i}" for i in range(int(args.class_subset or 100))]
        train_loader = val_loader = test_loader = None
    else:
        os.makedirs(out, exist_ok=True)
        train_loader, val_loader, test_loader, classes = build_folder_loaders(
            args.dataset_dir, args.train_dir, args.test_dir, args.image_size, args.batch_size,
            args.num_workers, args.val_split, args.test_split, args.augment,
            "imagenet" if args.backbone else args.norm, args.class_subset, args.max_per_class, args.seed,
            pin_memory=device.type == "cuda")

    mcfg = CrackViTConfig(**{f.name: getattr(args, f.name) for f in fields(CrackViTConfig) if hasattr(args, f.name)})
    mcfg.num_classes = len(classes)
    model = LoopedCrackViT(mcfg)
    if args.backbone and not args.resume:
        from pretrained import load_pretrained
        load_pretrained(model, args.backbone, args.init_strategy)
    model = model.to(device)
    exit_names, exit_cost = model.exit_names, model.exit_cost
    print(f">>> {dataset} / {name}\n    {model.param_report()}\n    block applications per exit: {exit_cost}")
    if args.summary_only:
        return

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "model": mcfg.to_dict(), "classes": classes}, f, indent=2)

    exit_w = [1.0 if n == "final" else args.aux_weight for n in exit_names]
    mix = (MixupCutmix(args.mixup_alpha, args.cutmix_alpha, args.mixup_prob, args.mixup_switch_prob,
                       args.label_smoothing, len(classes)) if (args.mixup_alpha > 0 or args.cutmix_alpha > 0)
           else MixupCutmix(0, 0, 0.0, 0.5, args.label_smoothing, len(classes)))

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay, args.lr, args.layer_decay), lr=args.lr)
    accum = max(1, args.grad_accum)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    es = EarlyStopping(args.monitor, args.early_stop_patience, args.early_stop_min_delta)

    start_epoch, step, history = 1, 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if ck["model_cfg"] != mcfg.to_dict():
            raise SystemExit("--resume: checkpoint model config differs from the current one")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
        es.load_state_dict(ck["early_stop"]); step, history, start_epoch = ck["step"], ck["history"], ck["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    cols = (["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5", "val_f1_macro", "val_bal_acc"]
            + [f"val_acc_{e}" for e in exit_names[:-1]] + ["sec"])
    log_path = os.path.join(out, "log.csv")
    if not (args.resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)
    print(f"\nmonitor={args.monitor} patience={args.early_stop_patience} | layer_decay={args.layer_decay} | "
          f"mixup={args.mixup_alpha}/cutmix={args.cutmix_alpha} | amp={amp_dtype if use_amp else 'off'}\n")

    run_start, stopped_early = time.time(), False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); t0 = time.time()
        n = correct = 0; loss_sum = 0.0; lr = args.lr
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
        for it, (x, y) in enumerate(pbar):
            if it % accum == 0:
                lr = cosine_lr(step, total_steps, warmup, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr * g.get("lr_scale", 1.0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xm, target = mix(x, y)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                outs = model(xm)
            loss = soft_ce(outs, target, exit_w)
            if model.reg_loss is not None:
                loss = loss + model.reg_loss
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0 or (it + 1) == len(train_loader):
                if args.grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            loss_sum += loss.item() * y.size(0); n += y.size(0)
            correct += (outs[-1].argmax(-1) == y).sum().item()     # vs. original labels (approx. under mixup)
            pbar.set_postfix(loss=f"{loss_sum / n:.3f}", lr=f"{lr:.1e}")
        pbar.close()

        yv, pv, vloss = predict(model, val_loader, device, amp_dtype, use_amp)
        mv = M.compute_multiclass(yv, pv["final"])
        row = dict(epoch=epoch, lr=lr, train_loss=loss_sum / n, train_acc=correct / n, val_loss=vloss,
                   val_acc=mv["acc"], val_top5=mv["top5"], val_f1_macro=mv["f1_macro"], val_bal_acc=mv["bal_acc"],
                   sec=time.time() - t0)
        for e in exit_names[:-1]:
            row[f"val_acc_{e}"] = float((pv[e].argmax(1) == yv).mean())
        history.append(row)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{row[c]:.6g}" if isinstance(row[c], float) else row[c] for c in cols])

        improved, stop = es.update(row[args.monitor], epoch)
        ck = {"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "classes": classes, "epoch": epoch,
              "step": step, "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "history": history,
              "early_stop": es.state_dict()}
        torch.save(ck, os.path.join(out, "last.pt"))
        if improved:
            torch.save(ck, os.path.join(out, "best.pt"))
        if args.save_every and epoch % args.save_every == 0:
            torch.save(ck, os.path.join(out, f"epoch_{epoch:04d}.pt"))
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        ex = " ".join(f"{e}:{row[f'val_acc_{e}']:.4f}" for e in exit_names[:-1])
        print(f"epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | train loss {row['train_loss']:.4f} | val loss {vloss:.4f} "
              f"acc {mv['acc']:.4f} top5 {mv['top5']:.4f} f1m {mv['f1_macro']:.4f} {ex} | {row['sec']:.0f}s"
              f"{'  * best' if improved else f'  (no improv {es.bad}/{args.early_stop_patience or "-"})'}")
        if stop:
            stopped_early = True
            print(f"[early stopping] {args.monitor} did not improve for {es.bad} epochs "
                  f"(best {es.best_value:.4f} @ epoch {es.best_epoch})")
            break
    pbar = None
    del train_loader                    # free its persistent worker processes before the test loader starts
    import gc; gc.collect()
    total_min = (time.time() - run_start) / 60
    print(f"\ntraining done in {total_min:.1f} min | best {args.monitor}={es.best_value:.4f} @ epoch {es.best_epoch}")

    # ================================================================ test evaluation (best.pt)
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)["model"])
    yt, pt, _ = predict(model, test_loader, device, amp_dtype, use_amp, "test")
    mt = M.compute_multiclass(yt, pt["final"])
    lo, hi = bootstrap_acc(yt, pt["final"].argmax(1), args.bootstrap) if args.bootstrap else (float("nan"),) * 2
    print("\n" + "=" * 78 + f"\n  TEST — {dataset} — {name}\n" + "=" * 78)
    print("  " + " | ".join(f"{k} {v:.4f}" for k, v in mt.items() if k != "n" and v == v)
          + f"\n  acc 95% CI [{lo:.4f}, {hi:.4f}] (bootstrap)   n={mt['n']}")

    exit_df = pd.DataFrame([dict(exit=e, loops=i + 1, block_apps=exit_cost[e],
                                 **{k: v for k, v in M.compute_multiclass(yt, pt[e]).items()
                                    if k in ("acc", "top5", "f1_macro", "bal_acc", "nll", "ece")})
                            for i, e in enumerate(exit_names)])
    print("\n  ACCURACY vs LOOP COUNT (accuracy after each pass through the shared stack)")
    print(exit_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ee = []
    if len(exit_names) > 1:
        ee = M.early_exit_table_mc(yt, pt[exit_names[0]], pt["final"], exit_cost[exit_names[0]], exit_cost["final"],
                                   [None, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.0])
        print("\n  CONFIDENCE EARLY EXIT")
        print(pd.DataFrame(ee).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    gates = pd.DataFrame(model.gate_report())
    if len(gates):
        print("\n  LEARNED GATES PER PASS (init 0.01)")
        print(gates.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    cx = complexity(model, device, amp_dtype, use_amp, args.image_size)
    print("\n  COMPLEXITY: " + " | ".join(f"{k}={v:,.2f}" if isinstance(v, float) else f"{k}={v:,}" for k, v in cx.items()))

    # ---- write ----
    pred = pt["final"].argmax(1)
    with open(f"{out}/per_class.txt", "w") as f:
        f.write(classification_report(yt, pred, labels=list(range(len(classes))), target_names=classes,
                                      digits=4, zero_division=0))
    cm = confusion_matrix(yt, pred, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"{out}/confusion_matrix.csv")
    exit_df.to_csv(f"{out}/per_exit.csv", index=False)
    if ee:
        pd.DataFrame(ee).to_csv(f"{out}/early_exit.csv", index=False)
    if len(gates):
        gates.to_csv(f"{out}/gates.csv", index=False)
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val CE"); ax[0].set_title("Loss")
    ax[1].plot(h.epoch, h.val_acc, label="final")
    for e in exit_names[:-1]:
        ax[1].plot(h.epoch, h[f"val_acc_{e}"], "--", label=e)
    ax[1].set_title("Val accuracy per exit")
    ax[2].plot(exit_df.loops, exit_df.acc, "o-"); ax[2].set_xlabel("loop iterations"); ax[2].set_title("Test accuracy vs loop count")
    for a in ax:
        a.grid(alpha=.3)
        if a is not ax[2]:
            a.set_xlabel("epoch"); a.legend(fontsize=8)
    fig.suptitle(f"{dataset} / {name}", fontsize=10); fig.tight_layout(); fig.savefig(f"{out}/curves.png", dpi=150); plt.close(fig)

    res = dict(dataset=dataset, run=name, attention=args.attention, backbone=args.backbone or "scratch",
               init_strategy=args.init_strategy if args.backbone else "", n_prelude=args.n_prelude, n_core=args.n_core,
               n_coda=args.n_coda, n_passes=args.n_passes, image_size=args.image_size, seed=args.seed,
               epochs_run=len(history), best_epoch=es.best_epoch, stopped_early=stopped_early, train_min=total_min,
               acc_ci_lo=lo, acc_ci_hi=hi, **cx)
    res.update({f"test_{k}": v for k, v in mt.items()})
    for r in exit_df.itertuples():
        res[f"acc_loop{r.loops}"] = r.acc
    with open(f"{out}/results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    pd.DataFrame([res]).to_csv(f"{out}/results.csv", index=False)
    print(f"\nall outputs -> {out}")


if __name__ == "__main__":
    main()
