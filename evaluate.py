"""
Evaluate a saved SDNET2018 run on the TEST split and write the paper metrics table.

    python evaluate.py --run runs/gipa_full_0-6x2-0_conv_bs16_random_smote_s42

Use this when training ended before train.py's final evaluation (or to re-score another
checkpoint with --ckpt). Rebuilds the model from config.json, uses the cached split in
runs/_splits, tunes the decision thresholds on VAL only and applies them to TEST.

Writes to <run>/eval/:
    metrics_table.md / .csv   ROC-AUC, PR-AUC, crack recall/precision, specificity, MCC, Brier,
                              ECE, parameters (M), block applications (final exit) + 95% bootstrap CI
    operating_points.csv      all metrics at thr 0.5 / val-best-F1 / val-recall target
    per_exit.csv              exit1 vs final
    test_predictions.csv      P(Cracked) per test image and exit
    metrics.json              everything above in one file
Predictions are cached in <run>/eval/_preds_<ckpt>.npz (delete it, or pass --no-cache, to recompute).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *a, **k):
        return it

import metrics as M
from data import CrackDataset, build_transforms, load_split
from model import CrackViTConfig, LoopedCrackViT

# (table label, key in metrics.compute_all, format)
TABLE = [("ROC-AUC", "auc", ".4f"), ("PR-AUC (average precision)", "ap", ".4f"),
         ("Crack recall (sensitivity)", "recall", ".4f"), ("Crack precision", "precision", ".4f"),
         ("Specificity", "specificity", ".4f"), ("MCC", "mcc", ".4f"),
         ("Brier score", "brier", ".4f"), ("Expected calibration error", "ece", ".4f")]
THRESHOLD_FREE = {"auc", "ap", "brier", "ece"}


def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/gipa_full_0-6x2-0_conv_bs16_random_smote_s42")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=1000, help="stratified bootstrap resamples (0 = no CI)")
    ap.add_argument("--target-recall", type=float, default=None, help="default: value from the run config")
    ap.add_argument("--amp", action="store_true", help="bf16/fp16 autocast like train.py (default fp32)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


@torch.no_grad()
def predict(model, df, tf, device, bs, workers, amp, desc):
    dl = DataLoader(CrackDataset(df, tf), batch_size=bs, shuffle=False, num_workers=workers,
                    pin_memory=device.type == "cuda")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    logits = [[] for _ in model.exit_names]
    for x, _ in tqdm(dl, desc=desc, dynamic_ncols=True):
        with torch.autocast(device.type, dtype=dtype, enabled=amp and device.type == "cuda"):
            outs = model(x.to(device, non_blocking=True))
        for i, o in enumerate(outs):
            logits[i].append(o.float().cpu())
    return {n: torch.sigmoid(torch.cat(l)).numpy() for n, l in zip(model.exit_names, logits)}


def main():
    args = get_args()
    run = os.path.normpath(args.run)
    out = os.path.join(run, "eval")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)

    cfg = json.load(open(os.path.join(run, "config.json")))
    targs = cfg["args"]
    ck = torch.load(os.path.join(run, args.ckpt), map_location="cpu", weights_only=False)
    model = LoopedCrackViT(CrackViTConfig(**ck.get("model_cfg", cfg["model"])))
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    exit_names, exit_cost = model.exit_names, model.exit_cost
    params = model.param_report()
    print(f">>> {os.path.basename(run)} | {args.ckpt} (epoch {ck.get('epoch')}) | device {device}")

    norm = "imagenet" if targs.get("norm") == "imagenet" or (targs.get("norm") == "auto" and targs.get("backbone")) \
        else "half"
    df = load_split(targs["data_root"], targs["split_mode"], targs["seed"],
                    os.path.join(os.path.dirname(run), "_splits"))
    va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("val", "test"))

    cache = os.path.join(out, f"_preds_{os.path.splitext(args.ckpt)[0]}.npz")
    if os.path.exists(cache) and not args.no_cache:
        z = np.load(cache)
        pv, pt = ({e: z[f"{s}_{e}"] for e in exit_names} for s in ("val", "test"))
        print(f"loaded cached predictions from {cache}")
    else:
        tf = build_transforms(targs["image_size"], False, norm)
        pv = predict(model, va, tf, device, args.batch_size, args.num_workers, args.amp, "val")
        pt = predict(model, te, tf, device, args.batch_size, args.num_workers, args.amp, "test")
        np.savez(cache, **{f"val_{e}": pv[e] for e in exit_names}, **{f"test_{e}": pt[e] for e in exit_names})
    yv, yt = va["label"].to_numpy().astype(int), te["label"].to_numpy().astype(int)

    # thresholds tuned on VAL only, applied to TEST
    beta = targs.get("threshold_beta", 1.0)
    target = args.target_recall or targs.get("target_recall", 0.95)
    thr_f, _ = M.best_f1_threshold(yv, pv["final"], beta)
    thr_r = M.threshold_at_recall(yv, pv["final"], target)
    ops = {"thr_0.5": 0.5, f"thr_bestF{beta:g}(val)": thr_f, f"thr_recall{target:g}(val)": thr_r}
    op_rows = [dict(operating_point=k, **M.compute_all(yt, pt["final"], t)) for k, t in ops.items()]
    m05, mf = op_rows[0], op_rows[1]

    keys = tuple(k for _, k, _ in TABLE)
    ci05 = M.bootstrap_ci(yt, pt["final"], 0.5, keys=keys, n_boot=args.bootstrap) if args.bootstrap else {}
    cif = M.bootstrap_ci(yt, pt["final"], thr_f, keys=keys, n_boot=args.bootstrap) if args.bootstrap else {}

    # ---- the paper table ----
    fmt_ci = lambda ci, k, f: f"[{ci[k][0]:{f}}, {ci[k][1]:{f}}]" if ci else ""
    rows = []
    for label, k, f in TABLE:
        rows.append({"Metric": label,
                     "@0.5": f"{m05[k]:{f}}", "95% CI @0.5": fmt_ci(ci05, k, f),
                     f"@val-F1 thr ({thr_f:.3f})": "same" if k in THRESHOLD_FREE else f"{mf[k]:{f}}",
                     "95% CI @val-F1 thr": "" if k in THRESHOLD_FREE else fmt_ci(cif, k, f)})
    rows.append({"Metric": "Parameters (M)", "@0.5": f"{params['total_params'] / 1e6:.2f}"})
    rows.append({"Metric": "Block applications (final exit)", "@0.5": str(exit_cost["final"])})
    table = pd.DataFrame(rows).fillna("")
    table.to_csv(os.path.join(out, "metrics_table.csv"), index=False)

    n_pos = int(yt.sum())
    md = [f"# Test metrics: {os.path.basename(run)}", "",
          f"Checkpoint `{args.ckpt}` (epoch {ck.get('epoch')}), split `{targs['split_mode']}` seed {targs['seed']}, "
          f"test N = {len(yt)} ({n_pos} cracked, prevalence {yt.mean():.3f}). Positive class = Cracked.",
          f"Thresholds are tuned on VAL and applied to TEST. ECE uses 15 equal-width bins. "
          f"95% CIs: {args.bootstrap} stratified bootstrap resamples." if args.bootstrap else "", "",
          table.to_markdown(index=False, disable_numparse=True), ""]
    with open(os.path.join(out, "metrics_table.md"), "w", encoding="utf8") as f:
        f.write("\n".join(md))

    # ---- supporting files ----
    pd.DataFrame(op_rows).to_csv(os.path.join(out, "operating_points.csv"), index=False)
    exit_df = pd.DataFrame([dict(exit=e, block_apps=exit_cost[e],
                                 **{k: v for k, v in M.compute_all(yt, pt[e], thr_f).items()
                                    if k in ("auc", "ap", "precision", "recall", "specificity", "mcc", "brier", "ece")})
                            for e in exit_names])
    exit_df.to_csv(os.path.join(out, "per_exit.csv"), index=False)
    pred = te[["path", "surface", "label"]].copy()
    for e in exit_names:
        pred[f"p_{e}"] = pt[e]
    pred.to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(dict(run=os.path.basename(run), ckpt=args.ckpt, epoch=ck.get("epoch"),
                       total_params=params["total_params"], params_M=params["total_params"] / 1e6,
                       block_apps_final=exit_cost["final"], exit_cost=exit_cost,
                       thr_val_f1=thr_f, thr_val_recall=thr_r,
                       test_at_0_5=m05, test_at_val_f1=mf, test_at_val_recall=op_rows[2],
                       ci_at_0_5=ci05, ci_at_val_f1=cif, per_exit=exit_df.to_dict("records")),
                  f, indent=2, default=float)

    print("\n" + "\n".join(md[2:]))
    print("per exit (val-F1 threshold):\n" + exit_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nall outputs -> {out}")


if __name__ == "__main__":
    main()
