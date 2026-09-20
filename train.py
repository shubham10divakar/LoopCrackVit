"""
Train Looped CrackViT + GIPA on SDNET2018.

    python train.py --config config.yaml                       # STACK 6x2 GIPA_FULL (defaults)
    python train.py --config config.yaml --summary-only        # model summary, no data needed
    python train.py --config config.yaml --n-prelude 1 --n-core 4 --n-coda 1     # SANDWICH
    python train.py --config config.yaml --n-passes 1          # untied 6-block baseline
    python train.py --config config.yaml --attention MHSA      # does looping help GIPA more?
    python train.py --config config.yaml --resume runs/<run>/last.pt

Flags override the YAML. Everything is written to runs/<run_name>/:
    best.pt last.pt log.csv history.json config.json
    results.json results.csv (one row, for the ablation table)
    test_predictions.csv per_surface.csv per_exit.csv early_exit.csv gates.csv
    curves.png cm.png roc.png pr.png calibration.png early_exit.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import fields

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             precision_recall_curve)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *a, **k):
        return it

import metrics as M
from augment import MinorityBank, mix_batch, smote_batch
from data import CLASS_NAMES, build_loaders, describe_split, load_split
from model import CrackViTConfig, LoopedCrackViT

MINIMISE = {"val_loss"}


# ------------------------------------------------------------------ args
def str2bool(v):
    return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "y")


def nullable(cast):
    """CLI value converter: 'null' / 'none' -> None (e.g. --backbone null, --early-stop-patience null)."""
    return lambda x: None if str(x).lower() in ("none", "null") else cast(x)


def get_args(default_config="config.yaml"):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=default_config)
    pre, _ = p.parse_known_args()
    with open(pre.config, encoding="utf8") as f:
        cfg = {k.replace("-", "_"): v for k, v in (yaml.safe_load(f) or {}).items()}
    for k, v in cfg.items():                    # one flag per YAML key, typed from its default
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, type=str2bool, nargs="?", const=True, default=v)
        elif v is None:
            p.add_argument(flag, default=None)
        else:
            p.add_argument(flag, type=nullable(type(v)), default=v)
    args = p.parse_args()
    for k, v in vars(args).items():             # keys whose YAML default is null arrive as strings
        if cfg.get(k, 0) is None and isinstance(v, str):
            if v.lower() in ("none", "null", ""):
                setattr(args, k, None)
            else:
                for cast in (int, float):
                    try:
                        setattr(args, k, cast(v)); break
                    except ValueError:
                        pass
    return args


def run_name(a):
    init = f"ft-{a.init_strategy}" if a.backbone else a.stem
    tag = ((f"_cut{a.cutmix_alpha:g}" if a.cutmix_alpha > 0 else "") + (f"_mix{a.mixup_alpha:g}" if a.mixup_alpha > 0 else "")
           + (f"_{a.imbalance}" if a.imbalance != "off" else ""))
    return (f"{a.attention.lower()}_{a.n_prelude}-{a.n_core}x{a.n_passes}-{a.n_coda}"
            f"_{init}_bs{a.batch_size * a.grad_accum}_{a.split_mode}{tag}_s{a.seed}")


def pick_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                        "cpu" if name == "auto" else name)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def eta(history, total, epoch):
    """Remaining time from the mean of the last 3 epoch durations (upper bound; early stopping may end sooner)."""
    secs = [h["sec"] for h in history[-3:]]
    left = (total - epoch) * (sum(secs) / len(secs))
    return f"{int(left // 3600)}h{int(left % 3600 // 60):02d}m"


# ------------------------------------------------------------------ early stopping
class EarlyStopping:
    """Stops when `monitor` has not improved by > min_delta for `patience` epochs."""

    def __init__(self, monitor, patience, min_delta):
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.sign = -1.0 if monitor in MINIMISE else 1.0
        self.best, self.best_epoch, self.bad = -math.inf, 0, 0

    def update(self, value, epoch):
        """Returns (improved, should_stop)."""
        v = self.sign * value
        improved = v > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad = v, epoch, 0
        else:
            self.bad += 1
        return improved, self.patience is not None and self.bad >= self.patience

    @property
    def best_value(self):
        return self.sign * self.best

    def state_dict(self):
        return dict(best=self.best, best_epoch=self.best_epoch, bad=self.bad)

    def load_state_dict(self, s):
        self.best, self.best_epoch, self.bad = s["best"], s["best_epoch"], s["bad"]


def sync_after_resume(out, es, ck_epoch, resume_path, history, log_path, cols):
    """Make the run folder consistent after resuming from ANY epoch checkpoint.
    * best.pt is reset to the best epoch within the resumed history (a later, abandoned branch of the
      same run may have left a different best.pt behind);
    * log.csv is rewritten from the resumed history, so epochs after the resume point are not duplicated."""
    import shutil
    src = resume_path if es.best_epoch == ck_epoch else os.path.join(out, f"epoch_{es.best_epoch:04d}.pt")
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(out, "best.pt"))
    else:
        print(f"[warn] {src} not found, so best.pt cannot be re-synced; keep per-epoch checkpoints (--save-every 1) "
              f"to avoid this")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for h in history:
            w.writerow([f"{v:.6g}" if isinstance(v := h.get(c), float) else ("" if v is None else v) for c in cols])


# ------------------------------------------------------------------ optimisation helpers
def cosine_lr(step, total, warmup, base, min_lr):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


NEW_PARAM_TAGS = ("lam_", "gamma", "kappa", ".W", "tensor_head", "pass_emb", "inject", "head.")
NO_DECAY_TAGS = ("lam_", "gamma", "kappa", ".W", "inject", "table")


def layer_id(name, cfg):
    """Depth index of a parameter for layer-wise LR decay (0 = tokeniser ... top = head/new params)."""
    P, C = cfg.n_prelude, cfg.n_core
    top = P + C + cfg.n_coda + 1
    if any(t in name for t in NEW_PARAM_TAGS):          # new / task-specific params: full LR
        return top
    if name.startswith(("tokeniser", "cls_pos")):
        return 0
    for tag, off in (("prelude.", 1), ("core.", 1 + P), ("coda.", 1 + P + C)):
        if name.startswith(tag):
            return off + int(name.split(".")[1])
    return top


def param_groups(model, wd, lr=1.0, layer_decay=1.0):
    """AdamW groups. layer_decay < 1 => lr_scale = layer_decay ** (top - layer_id), so earlier
    pretrained layers move slower; new params and the head always get the full LR."""
    cfg = model.cfg
    top = cfg.n_prelude + cfg.n_core + cfg.n_coda + 1
    groups = {}
    for n, p in model.named_parameters():
        no_wd = p.ndim <= 1 or n.endswith(("cls", "pos")) or any(t in n for t in NO_DECAY_TAGS)
        lid = layer_id(n, cfg)
        g = groups.setdefault((lid, no_wd), {"params": [], "weight_decay": 0.0 if no_wd else wd,
                                             "lr_scale": layer_decay ** (top - lid)})
        g["params"].append(p)
    for g in groups.values():
        g["lr"] = lr * g["lr_scale"]
    return list(groups.values())


def multi_exit_loss(outs, y, exit_w, smoothing, sw):
    """Class-weighted, label-smoothed BCE summed over exits (final exit weight 1.0)."""
    t = y * (1 - smoothing) + 0.5 * smoothing
    w = sw[0] + (sw[1] - sw[0]) * y          # interpolates the class weight for soft (mixed) targets
    total = 0.0
    for o, ew in zip(outs, exit_w):
        total = total + ew * (F.binary_cross_entropy_with_logits(o.float(), t, reduction="none") * w).mean()
    return total


@torch.no_grad()
def predict(model, loader, device, amp_dtype, use_amp, desc=None):
    """-> y (N,), probs {exit: (N,)} of P(Cracked), plain-BCE loss of the final exit."""
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
    loss = F.binary_cross_entropy_with_logits(lg[-1], y).item()
    return y.numpy().astype(int), {n: torch.sigmoid(l).numpy() for n, l in zip(model.exit_names, lg)}, loss


# ------------------------------------------------------------------ plots
def plot_curves(hist, exit_names, path, title):
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    ax[0].plot(h.epoch, h.train_loss, label="train (weighted, all exits)"); ax[0].plot(h.epoch, h.val_loss, label="val BCE (final)")
    ax[0].set_title("Loss")
    for a, key, t in [(ax[1], "auc", "Val ROC-AUC"), (ax[2], "ap", "Val PR-AUC"), (ax[3], "f1", "Val F1 (Cracked @0.5)")]:
        a.plot(h.epoch, h[f"val_{key}"], label="final")
        if f"val_{key}_exit1" in h and len(exit_names) > 1:
            a.plot(h.epoch, h[f"val_{key}_exit1"], "--", label="exit1")
        a.set_title(t)
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_eval(y, p, thr, out, title):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y, (p >= thr).astype(int), display_labels=CLASS_NAMES,
                                            ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{title}\nTest confusion matrix @ thr={thr:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{out}/cm.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y, p, ax=ax, name="final exit"); ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_title("ROC (Cracked positive)", fontsize=10); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/roc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    PrecisionRecallDisplay.from_predictions(y, p, ax=ax, name="final exit")
    ax.axhline(y.mean(), color="k", ls=":", lw=1, label=f"prevalence {y.mean():.2f}")
    ax.set_title("Precision-Recall (Cracked)", fontsize=10); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/pr.png", dpi=150); plt.close(fig)

    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 0]
    ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 0]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k:"); ax.plot(xs, ys, "o-")
    ax.set_xlabel("predicted P(Cracked)"); ax.set_ylabel("observed fraction Cracked")
    ax.set_title("Reliability diagram", fontsize=10); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/calibration.png", dpi=150); plt.close(fig)


def plot_early_exit(rows, out, title):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df.avg_block_apps, df.f1, "o-", label="F1 (Cracked)")
    ax.plot(df.avg_block_apps, df.auc, "s--", label="ROC-AUC")
    for _, r in df.iterrows():
        ax.annotate(str(r.tau), (r.avg_block_apps, r.f1), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("average block applications per image"); ax.set_title(f"{title}\nearly-exit trade-off", fontsize=9)
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(f"{out}/early_exit.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ complexity
def complexity(model, device, amp_dtype, use_amp, size):
    info = model.param_report()
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x, max_pass=None)
        info["gmacs_final"] = fc.get_total_flops() / 2e9          # FLOPs / 2 = MACs (fwd, 1 image)
    except Exception as e:
        print(f"[complexity] flop count skipped: {e}")
    if device.type == "cuda":
        xb = torch.randn(32, 3, size, size, device=device)
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            for _ in range(5):
                model(xb)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(20):
                model(xb)
            torch.cuda.synchronize()
        info["throughput_img_s"] = 32 * 20 / (time.time() - t)
    return info


# ------------------------------------------------------------------ main
def main():
    args = get_args()
    args.imbalance = args.imbalance or "off"
    if args.imbalance not in ("off", "oversample", "smote"):
        raise SystemExit("--imbalance must be off | oversample | smote")
    if args.imbalance != "off" and args.class_weights:
        print(f"[note] --imbalance {args.imbalance} already rebalances the classes, so loss class weights are turned off")
        args.class_weights = False
    seed_all(args.seed)
    device = pick_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    mcfg = CrackViTConfig(**{f.name: getattr(args, f.name) for f in fields(CrackViTConfig) if hasattr(args, f.name)})
    model = LoopedCrackViT(mcfg)
    if args.backbone and not args.resume:
        from pretrained import load_pretrained
        load_pretrained(model, args.backbone, args.init_strategy)
    model = model.to(device)
    exit_names, exit_cost = model.exit_names, model.exit_cost
    name = run_name(args)
    print(f">>> RUN {name}")
    print(model.summary())
    if args.summary_only:
        return

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    out = os.path.join(args.output_dir, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"args": vars(args), "model": mcfg.to_dict()}, f, indent=2)

    # ---- data ----------------------------------------------------------------
    df = load_split(args.data_root, args.split_mode, args.seed, os.path.join(args.output_dir, "_splits"))
    if args.debug_subset:       # quick pipeline test: N images per split (both classes kept)
        df = df.groupby(["split", "label"], group_keys=False).head(args.debug_subset // 2)
    print(f"\nsplit_mode={args.split_mode}\n{describe_split(df)}")
    train_loader, val_loader, test_loader, (tr_df, va_df, te_df) = build_loaders(
        df, args.image_size, args.batch_size, args.num_workers, args.augment,
        pin_memory=device.type == "cuda", seed=args.seed, oversample=args.imbalance == "oversample",
        norm="imagenet" if (args.norm == "auto" and args.backbone) or args.norm == "imagenet" else "half")

    n_pos = int(tr_df["label"].sum()); n_neg = len(tr_df) - n_pos
    cw = torch.tensor([len(tr_df) / (2 * n_neg), len(tr_df) / (2 * n_pos)] if args.class_weights else [1., 1.],
                      device=device)
    print(f"class weights: Non-cracked={cw[0]:.3f}  Cracked={cw[1]:.3f}  (train prevalence {n_pos / len(tr_df):.3f})")
    exit_w = [1.0 if n == "final" else args.aux_weight for n in exit_names]
    bank = MinorityBank(args.smote_bank) if args.imbalance == "smote" else None
    both = args.cutmix_alpha > 0 and args.mixup_alpha > 0
    print(f"batch augmentation: cutmix_alpha={args.cutmix_alpha} mixup_alpha={args.mixup_alpha}"
          + (f" | {args.mixup_prob:.0%} of batches mixed" if (args.cutmix_alpha > 0 or args.mixup_alpha > 0) else " (no mixing)")
          + (f", cutmix share {args.mixup_switch_prob}" if both else "")
          + f" | imbalance={args.imbalance}"
          + (f" (target {args.smote_target:.0%} cracked per batch)" if bank else ""))

    # ---- optim ---------------------------------------------------------------
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay, args.lr, args.layer_decay), lr=args.lr)
    accum = max(1, args.grad_accum)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    plateau = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr)
        if args.scheduler == "plateau" else None)
    es = EarlyStopping(args.monitor, args.early_stop_patience, args.early_stop_min_delta)

    start_epoch, step, history = 1, 0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if ck["model_cfg"] != mcfg.to_dict():
            raise SystemExit("--resume: checkpoint model config differs from the current one")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
        if plateau and ck.get("plateau"): plateau.load_state_dict(ck["plateau"])
        es.load_state_dict(ck["early_stop"]); step, history = ck["step"], ck["history"]
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    cols = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_bal_acc", "val_auc",
            "val_ap", "val_f1", "val_f2", "val_precision", "val_recall", "val_mcc"]
    cols += [f"val_{k}_{e}" for e in exit_names[:-1] for k in ("auc", "f1")] + ["sec"]
    log_path = os.path.join(out, "log.csv")
    if args.resume:
        sync_after_resume(out, es, start_epoch - 1, args.resume, history, log_path, cols)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    print(f"\ntraining: monitor={args.monitor} patience={args.early_stop_patience} "
          f"min_delta={args.early_stop_min_delta} | scheduler={args.scheduler} | amp={amp_dtype if use_amp else 'off'}\n")

    # ---- train ---------------------------------------------------------------
    run_start = time.time()
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); t0 = time.time()
        n = correct = 0; loss_sum = 0.0; lr = opt.param_groups[0]["lr"]
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, dynamic_ncols=True)
        for it, (x, y) in enumerate(pbar):
            if plateau is None and it % accum == 0:
                lr = cosine_lr(step, total_steps, warmup, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr * g.get("lr_scale", 1.0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if bank is not None:
                x, y = smote_batch(x, y, bank, args.smote_target)
            x, y = mix_batch(x, y, args.mixup_alpha, args.cutmix_alpha, args.mixup_prob, args.mixup_switch_prob)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                outs = model(x)
            loss = multi_exit_loss(outs, y, exit_w, args.label_smoothing, cw)
            if model.reg_loss is not None:
                loss = loss + model.reg_loss
            scaler.scale(loss / accum).backward()
            if (it + 1) % accum == 0 or (it + 1) == len(train_loader):
                if args.grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            loss_sum += loss.item() * y.size(0)
            correct += ((outs[-1] > 0).float() == (y >= 0.5).float()).sum().item(); n += y.size(0)
            pbar.set_postfix(loss=f"{loss_sum / n:.4f}", acc=f"{correct / n:.4f}", lr=f"{lr:.1e}", gpu=f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G" if device.type == "cuda" else "cpu")
        pbar.close()

        # ---- validate (every exit) ----
        yv, pv, vloss = predict(model, val_loader, device, amp_dtype, use_amp, f"  val {epoch}")
        mv = M.compute_all(yv, pv["final"])
        row = dict(epoch=epoch, lr=lr, train_loss=loss_sum / n, train_acc=correct / n, val_loss=vloss,
                   **{f"val_{k}": mv[k] for k in ("acc", "bal_acc", "auc", "ap", "f1", "f2", "precision", "recall", "mcc")})
        for e in exit_names[:-1]:
            me = M.compute_all(yv, pv[e]); row[f"val_auc_{e}"], row[f"val_f1_{e}"] = me["auc"], me["f1"]
        row["sec"] = time.time() - t0
        history.append(row)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{row[c]:.6g}" if isinstance(row[c], float) else row[c] for c in cols])

        improved, stop = es.update(row[args.monitor], epoch)
        if plateau is not None:
            plateau.step(vloss)
        ck = {"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "epoch": epoch, "step": step,
              "optimizer": opt.state_dict(), "scaler": scaler.state_dict(), "history": history,
              "plateau": plateau.state_dict() if plateau else None, "early_stop": es.state_dict()}
        torch.save(ck, os.path.join(out, "last.pt"))
        if improved:
            torch.save(ck, os.path.join(out, "best.pt"))
        if args.save_every and epoch % args.save_every == 0:
            torch.save(ck, os.path.join(out, f"epoch_{epoch:04d}.pt"))
        with open(os.path.join(out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        ex = " ".join(f"{e}:auc={row[f'val_auc_{e}']:.4f}" for e in exit_names[:-1])
        print(f"epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f}"
              f" | val loss {vloss:.4f} auc {mv['auc']:.4f} ap {mv['ap']:.4f} f1 {mv['f1']:.4f}"
              f" rec {mv['recall']:.4f} prec {mv['precision']:.4f} mcc {mv['mcc']:.4f} {ex}"
              f" | {row['sec']:.0f}s | ETA {eta(history, args.epochs, epoch)} | best {args.monitor} {es.best_value:.4f}{'  * best' if improved else f'  (no improv {es.bad}/{args.early_stop_patience or "-"})'}")
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

    # ================================================================ final evaluation
    ck = torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    yv, pv, _ = predict(model, val_loader, device, amp_dtype, use_amp, "val")
    yt, pt, tloss = predict(model, test_loader, device, amp_dtype, use_amp, "test")
    pf_v, pf_t = pv["final"], pt["final"]

    # thresholds are chosen on VAL only, then applied to TEST
    thr_f, _ = M.best_f1_threshold(yv, pf_v, args.threshold_beta)
    thr_r = M.threshold_at_recall(yv, pf_v, args.target_recall)
    ops = {"thr_0.5": 0.5, f"thr_bestF{args.threshold_beta:g}(val)": thr_f,
           f"thr_recall{args.target_recall:g}(val)": thr_r}
    op_rows = [dict(operating_point=k, **M.compute_all(yt, pf_t, t)) for k, t in ops.items()]
    op_df = pd.DataFrame(op_rows)
    show = ["operating_point", "threshold", "acc", "bal_acc", "precision", "recall", "specificity", "f1", "f2",
            "mcc", "iou_crack", "fn", "fp"]
    print("\n" + "=" * 78 + f"\n  TEST — FINAL EXIT — {name}\n" + "=" * 78)
    m0 = op_rows[0]
    print(f"  ROC-AUC {m0['auc']:.4f} | PR-AUC {m0['ap']:.4f} | Brier {m0['brier']:.4f} | ECE {m0['ece']:.4f} | NLL {m0['nll']:.4f}")
    print(op_df[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ci = M.bootstrap_ci(yt, pf_t, 0.5, n_boot=args.bootstrap) if args.bootstrap else {}
    if ci:
        print("\n  95% bootstrap CI @0.5: " + " | ".join(f"{k} [{lo:.4f}, {hi:.4f}]" for k, (lo, hi) in ci.items()))

    # per exit
    exit_rows = [dict(exit=e, block_apps=exit_cost[e],
                      **{k: v for k, v in M.compute_all(yt, pt[e], thr_f).items()
                         if k in ("auc", "ap", "acc", "precision", "recall", "f1", "mcc")}) for e in exit_names]
    exit_df = pd.DataFrame(exit_rows)
    print("\n  PER-EXIT (does the extra pass help?; threshold = val-tuned F1 of the final exit)")
    print(exit_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # per surface
    surf = te_df["surface"].to_numpy()
    srows = []
    for s in ("Decks", "Pavements", "Walls"):
        m = surf == s
        if m.sum():
            mm = M.compute_all(yt[m], pf_t[m], thr_f)
            srows.append(dict(surface=s, N=int(m.sum()), cracked=int(yt[m].sum()),
                              **{k: mm[k] for k in ("auc", "ap", "acc", "precision", "recall", "f1", "mcc")}))
    surf_df = pd.DataFrame(srows)
    print("\n  PER-SURFACE (val-tuned F1 threshold)")
    print(surf_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # confidence early exit
    ee_rows = []
    if len(exit_names) > 1:
        ee_rows = M.early_exit_table(yt, pt[exit_names[0]], pf_t, exit_cost[exit_names[0]], exit_cost["final"],
                                     [None, 0.99, 0.97, 0.95, 0.925, 0.90, 0.85, 0.80, 0.0], thr_f)
        print("\n  CONFIDENCE EARLY EXIT (accuracy vs. average compute)")
        print(pd.DataFrame(ee_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    gates = pd.DataFrame(model.gate_report())
    if len(gates):
        print("\n  LEARNED GATES PER PASS (init 0.01; growth => term is used)")
        print(gates.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cx = complexity(model, device, amp_dtype, use_amp, args.image_size)
    print("\n  COMPLEXITY: " + " | ".join(f"{k}={v:,.2f}" if isinstance(v, float) else f"{k}={v:,}" for k, v in cx.items()))

    # ---- write everything ----
    plot_curves(history, exit_names, f"{out}/curves.png", name)
    plot_eval(yt, pf_t, thr_f, out, name)
    if ee_rows:
        plot_early_exit(ee_rows, out, name)
    op_df.to_csv(f"{out}/operating_points.csv", index=False)
    exit_df.to_csv(f"{out}/per_exit.csv", index=False)
    surf_df.to_csv(f"{out}/per_surface.csv", index=False)
    if ee_rows:
        pd.DataFrame(ee_rows).to_csv(f"{out}/early_exit.csv", index=False)
    if len(gates):
        gates.to_csv(f"{out}/gates.csv", index=False)
    pred = te_df[["path", "surface", "label"]].copy()
    for e in exit_names:
        pred[f"p_{e}"] = pt[e]
    pred.to_csv(f"{out}/test_predictions.csv", index=False)

    res = dict(run=name, attention=args.attention, split_mode=args.split_mode, n_prelude=args.n_prelude,
               n_core=args.n_core, n_coda=args.n_coda, n_passes=args.n_passes,
               batch_size=args.batch_size * accum, epochs_run=len(history), best_epoch=es.best_epoch,
               stopped_early=stopped_early, train_min=total_min, test_loss=tloss, **cx,
               thr_val_f1=thr_f, thr_val_recall=thr_r)
    res.update({f"test_{k}": v for k, v in m0.items() if k not in ("threshold",)})
    for k in ("precision", "recall", "specificity", "f1", "f2", "mcc", "iou_crack", "fn", "fp"):
        res[f"test_{k}@valF1thr"] = op_rows[1][k]
    for r in exit_rows[:-1]:
        res.update({f"{r['exit']}_auc": r["auc"], f"{r['exit']}_f1": r["f1"]})
    for k, (lo, hi) in ci.items():
        res[f"ci_{k}_lo"], res[f"ci_{k}_hi"] = lo, hi
    for r in srows:
        res[f"{r['surface']}_auc"], res[f"{r['surface']}_f1"] = r["auc"], r["f1"]
    with open(f"{out}/results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    pd.DataFrame([res]).to_csv(f"{out}/results.csv", index=False)
    print(f"\nall outputs -> {out}")


if __name__ == "__main__":
    main()
