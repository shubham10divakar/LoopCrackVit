"""
Collect results.csv files and report mean +- std over seeds per configuration.

    python aggregate.py --root runs_bench            # multi-class benchmarks
    python aggregate.py --root runs_ft               # SDNET fine-tune runs
    python aggregate.py --root runs                  # SDNET from-scratch runs

Writes <root>/summary_all_runs.csv (every run) and <root>/summary_mean_std.csv / .md
(one row per configuration: params, GMACs, block applications, metrics as mean +- std, n seeds).
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

KEYS = ["dataset", "attention", "backbone", "init_strategy", "n_prelude", "n_core", "n_coda", "n_passes",
        "image_size", "split_mode"]
METRICS = ["test_acc", "test_top5", "test_f1_macro", "test_bal_acc", "test_ece",              # multi-class
           "acc_loop1", "acc_loop2", "acc_loop3", "acc_loop4",
           "test_auc", "test_ap", "test_f1", "test_recall", "test_precision", "test_mcc",     # SDNET
           "test_f1@valF1thr", "test_recall@valF1thr", "test_mcc@valF1thr", "exit1_auc"]
COSTS = ["total_params", "block_apps_final", "gmacs_final", "throughput_img_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs_bench")
    a = ap.parse_args()
    files = glob.glob(os.path.join(a.root, "**", "results.csv"), recursive=True)
    if not files:
        raise SystemExit(f"no results.csv under {a.root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(a.root, "summary_all_runs.csv"), index=False)
    keys = [k for k in KEYS if k in df.columns]
    df[keys] = df[keys].fillna("")
    metrics = [m for m in METRICS if m in df.columns and df[m].notna().any()]
    costs = [c for c in COSTS if c in df.columns]

    rows = []
    for k, g in df.groupby(keys, dropna=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["n_seeds"] = len(g)
        for c in costs:
            r[c] = g[c].iloc[0]
        for m in metrics:
            mu, sd = g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else float("nan")
            r[m] = f"{mu:.4f} ± {sd:.4f}" if len(g) > 1 else f"{mu:.4f}"
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.root, "summary_mean_std.csv"), index=False)
    with open(os.path.join(a.root, "summary_mean_std.md"), "w", encoding="utf8") as f:
        f.write(out.to_markdown(index=False) if _has_tabulate() else out.to_string(index=False))
    print(out.to_string(index=False))
    print(f"\nwrote {a.root}/summary_mean_std.csv|.md and summary_all_runs.csv")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
