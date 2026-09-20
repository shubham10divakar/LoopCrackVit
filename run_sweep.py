"""
Run the controlled comparison: datasets x variants x seeds, one after another.
Finished commands are recorded in <output_dir>/_sweep_done.txt, so re-running skips them
(safe to Ctrl-C and resume). Same recipe / resolution / epochs / augmentation for every variant.

    python run_sweep.py --datasets cub200 aircraft --seeds 42 43 44
    python run_sweep.py --datasets cub200 --variants ours plain12 --seeds 42 --dry-run
    python run_sweep.py --task sdnet --seeds 42 43 44                       # SDNET fine-tune sweep
    python run_sweep.py --datasets cifar100 --config config_scratch_cifar.yaml --scratch   # from scratch
    python run_sweep.py --datasets cub200 --extra "--image-size 384 --batch-size 16 --grad-accum 4"

Variants
    ours          looped 6x2, GIPA_FULL, DeiT-S init  (avg of layers i and i+6)      <- the claim
    loop_mhsa     looped 6x2, plain attention        (isolates GIPA from looping)
    shallow6      6 untied MHSA blocks               (parameter-matched shallow ViT)
    plain12       12 untied MHSA blocks              (= DeiT-S, the standard baseline)
    untied_gipa6  6 untied GIPA blocks               (isolates looping from GIPA)
    sandwich      1 + 4x2 + 1, GIPA_FULL
    ours_k3/k4    looped Kx, GIPA_FULL               (accuracy-vs-loop-count curve)
Then:  python aggregate.py --root runs_bench
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

VARIANTS = {
    "ours": "",
    "loop_mhsa": "--attention MHSA",
    "shallow6": "--attention MHSA --n-core 6 --n-passes 1 --pass-embed false",
    "plain12": "--attention MHSA --n-core 12 --n-passes 1 --pass-embed false",
    "untied_gipa6": "--n-core 6 --n-passes 1 --pass-embed false",
    "sandwich": "--n-prelude 1 --n-core 4 --n-coda 1",
    "ours_k3": "--n-passes 3",
    "ours_k4": "--n-passes 4",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["benchmark", "sdnet"], default="benchmark")
    ap.add_argument("--datasets", nargs="*", default=[], help="names under --datasets-root (benchmark task)")
    ap.add_argument("--datasets-root", default="datasets")
    ap.add_argument("--variants", nargs="+", default=["ours", "loop_mhsa", "shallow6", "plain12"],
                    choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--config", default=None)
    ap.add_argument("--scratch", action="store_true", help="from scratch (adds --backbone null)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--extra", default="", help="flags appended to every run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    script = "train_finetune.py" if a.task == "benchmark" else "train.py"
    config = a.config or ("config_finetune.yaml" if a.task == "benchmark" else "config_sdnet_finetune.yaml")
    out = a.output_dir or ("runs_bench" if a.task == "benchmark" else "runs_ft")
    os.makedirs(out, exist_ok=True)
    done_file = os.path.join(out, "_sweep_done.txt")
    done = set(open(done_file).read().splitlines()) if os.path.exists(done_file) else set()

    targets = [f"--dataset-dir {a.datasets_root}/{d}" for d in a.datasets] if a.task == "benchmark" else [""]
    if a.task == "benchmark" and not targets:
        sys.exit("--datasets is required for the benchmark task")
    cmds = []
    for t in targets:
        for v in a.variants:
            for s in a.seeds:
                cmds.append(f"{sys.executable} {script} --config {config} {t} {VARIANTS[v]} --seed {s} "
                            f"--output-dir {out} {'--backbone null' if a.scratch else ''} {a.extra}".replace("  ", " ").strip())
    print(f"{len(cmds)} runs ({len([c for c in cmds if c in done])} already done)")
    for i, c in enumerate(cmds, 1):
        if c in done:
            continue
        print(f"\n[{i}/{len(cmds)}] {c}")
        if a.dry_run:
            continue
        if subprocess.call(c, shell=True) == 0:
            with open(done_file, "a") as f:
                f.write(c + "\n")
        else:
            print(f"!! run failed, continuing: {c}")


if __name__ == "__main__":
    main()
