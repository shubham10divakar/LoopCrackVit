# Benchmarks: FGVC-Aircraft, CUB-200, Flowers-102, Food-101, CIFAR-100, …

Entry point: `train_finetune.py` (multi-class). Same model code as SDNET.

## Strategy: fine-tune from pretrained, do not train from scratch

CUB-200 has ~6k training images and FGVC-Aircraft ~6.7k. ViTs lack CNN inductive biases, so from scratch they
land far below published numbers, and published numbers are almost all fine-tuned from ImageNet backbones.
A from-scratch ViT would look weak for reasons unrelated to the looping idea, and pre-training on ImageNet
yourself is not realistic here. So:

1. **Initialise the looped blocks from a pretrained ViT** (timm DeiT-S) and fine-tune on each dataset. The numbers
   become comparable to the literature.
2. **Run controlled baselines with the identical recipe** (same script, resolution, epochs, augmentation, seeds):
   the plain non-looped ViT and a parameter-matched shallow ViT. The looped-vs-non-looped delta is the claim.
3. **One small from-scratch experiment** (CIFAR-100) with equal compute, to show the architecture helps on its own,
   independent of pretrained weights.
4. **Report** parameters, FLOPs, number of loop iterations, accuracy vs loop count, and mean ± std over 3 seeds.

The same mapping is available for SDNET (`config_sdnet_finetune.yaml`).

### How the pretrained weights are mapped (`pretrained.py`)

DeiT-S has 12 layers (dim 384, 6 heads, mlp 4.0, patch 16); the looped model has fewer *unique* blocks.

| strategy | rule | STACK 6×2 result |
|---|---|---|
| `avg` (default, Relaxed-Recursive-Transformers style) | each shared core block = **average** of the source layers it stands in for, one per pass | block *i* ← mean(layer *i*, layer *i+6*) |
| `pick` | evenly spaced source layers | blocks ← layers 0, 2, 4, 6, 8, 10 |

General rule for `avg` (source depth D, prelude P, core C, passes K, coda Q, L = D−P−Q):
prelude *j* ← layer *j*; coda *j* ← layer *D−Q+j*; core *i* ← mean over passes *p* of layer
`P + floor((p·C + i)·L/(C·K))`. It reproduces the baselines exactly:

| variant | unique blocks | init |
|---|---|---|
| plain12 (12 untied) | 12 | identity copy = DeiT-S (verified: max abs feature difference vs timm = 0.0) |
| shallow6 (6 untied) | 6 | layers 0, 2, 4, 6, 8, 10 |
| looped 6×2 | 6 | layers (i, i+6) averaged |
| sandwich 1+4×2+1 | 6 | prelude ← 0; core ← (1,6) (2,7) (3,8) (4,9); coda ← 11 |

Not in the checkpoint and therefore freshly initialised: GIPA gates (0.01, so the model starts as a plain ViT),
structure-tensor head, predictor `W`, pass embedding, and the task head. These new parameters, and the head,
always train at the full learning rate; pretrained layers get layer-wise LR decay (`--layer-decay`, 0.65 for
benchmarks, 0.75 for SDNET), earlier layers slower.

Position embeddings are bicubically interpolated when `--image-size` differs from 224.
The pretrained model needs `stem: patch`, `qkv_bias: true`, `patch_norm: false`, `head_type: linear`, ImageNet
normalisation. `config_finetune.yaml` and `config_sdnet_finetune.yaml` already set these.

## Datasets

```powershell
python downloads.py --dataset cub200 aircraft flowers102 food101 cifar100 plantdoc   # automatic
python downloads.py --dataset all
python downloads.py --dataset plantvillage --kaggle-dataset abdallahalidev/plantvillage-dataset   # needs Kaggle token
python downloads.py --dataset cassava                          # after accepting the competition rules
python downloads.py --root D:/datasets --dataset cub200        # custom output root
```

Result: `datasets/<name>/train/<class>/*.jpg` and, when the dataset ships an official split, `test/<class>/*.jpg`.
Raw archives are cached in `datasets/_raw/`; images are hard-linked (`--copy` for real copies). Re-running skips
finished datasets.

| `--dataset` | classes | size | notes |
|---|---|---|---|
| `cub200` | 200 | ~6k train / ~5.8k test | Caltech tarball, official split |
| `aircraft` | 100 (variants) | ~6.7k / ~3.3k | torchvision, official split |
| `flowers102` | 102 | ~2k / ~6k | train+val merged as train |
| `food101` | 101 | 75.8k / 25.3k | large |
| `cifar100` | 100 | 50k / 10k | 32×32; the from-scratch experiment |
| `plantdoc`, `plantvillage`, `cassava`, `appleleaf` | – | – | plant-disease sets from the Topo repo |

Any folder of `<class>/*.jpg` also works: `--dataset-dir path/to/dir` or `--train-dir … --test-dir …`.
Cache datasets as a Kaggle Dataset to avoid path/IO problems when moving to Kaggle.

**Splits.** A validation split (`--val-split 0.1`) is always held out of `train/` for early stopping and model
selection. If the dataset has no official `test/`, `--test-split 0.2` of the classes is held out as test too. The
official test set is only used for the final report.

## Running

```powershell
# ours: looped 6x2 GIPA, DeiT-S init
python train_finetune.py --dataset-dir datasets/cub200

# controlled baselines (same script, same recipe)
python train_finetune.py --dataset-dir datasets/cub200 --attention MHSA --n-core 12 --n-passes 1 --pass-embed false   # plain ViT (DeiT-S)
python train_finetune.py --dataset-dir datasets/cub200 --attention MHSA --n-core 6  --n-passes 1 --pass-embed false   # parameter-matched shallow ViT
python train_finetune.py --dataset-dir datasets/cub200 --attention MHSA                                              # looped, plain attention
python train_finetune.py --dataset-dir datasets/cub200 --n-core 6 --n-passes 1 --pass-embed false                     # 6 untied GIPA blocks

# accuracy vs loop count: one run trains K passes and reports accuracy after every pass
python train_finetune.py --dataset-dir datasets/cub200 --n-passes 4

# resolution (state it, and run ALL baselines at the same size)
python train_finetune.py --dataset-dir datasets/cub200 --image-size 384 --batch-size 8 --grad-accum 8

# quick test of the whole pipeline
python train_finetune.py --dataset-dir datasets/cifar100 --class-subset 10 --max-per-class 100 --epochs 2 --output-dir runs_debug
```

### The whole comparison, resumable

```powershell
python run_sweep.py --datasets cub200 aircraft --seeds 42 43 44                         # ours, loop_mhsa, shallow6, plain12
python run_sweep.py --datasets cub200 --variants ours ours_k3 ours_k4 sandwich untied_gipa6 --seeds 42 43 44
python run_sweep.py --datasets cub200 --extra "--image-size 384 --batch-size 8 --grad-accum 8"
python run_sweep.py --datasets cub200 --variants ours --seeds 42 --dry-run               # print commands only
python aggregate.py --root runs_bench
```

`run_sweep.py` records finished commands in `runs_bench/_sweep_done.txt`, so it can be interrupted and restarted.

### From-scratch experiment (independent of pretrained weights)

```powershell
python downloads.py --dataset cifar100
python run_sweep.py --datasets cifar100 --config config_scratch_cifar.yaml --scratch --variants ours loop_mhsa shallow6 untied_gipa6 --seeds 42 43 44
python aggregate.py --root runs_bench
```

Small 32×32 setting (patch 4 → 8×8 tokens, dim 256, 100 epochs). Use equal epochs for every variant.
Tiny-ImageNet works the same way once arranged as `train/<class>` and `test/<class>` folders.

## Outputs (`runs_bench/<dataset>/<run>/`)

`best.pt last.pt log.csv history.json config.json results.json results.csv`,
`per_exit.csv` (**accuracy vs loop count**), `early_exit.csv`, `per_class.txt` (precision/recall/F1 per class),
`confusion_matrix.csv`, `gates.csv`, `curves.png` (loss, per-exit val accuracy, test accuracy vs loops).
Run folder: `<attention>_<prelude>-<core>x<passes>-<coda>_<ft-avg|scratch>_r<resolution>_s<seed>`.

Metrics: top-1, top-5, macro-F1, weighted-F1, balanced accuracy, macro precision/recall, MCC, NLL, ECE,
one-vs-rest ROC-AUC (≤ 60 classes), 95 % bootstrap CI on accuracy, plus parameters, GMACs, block applications
and throughput. Early stopping monitors `val_acc` on the held-out validation split
(`--monitor val_top5|val_f1_macro|val_bal_acc|val_loss` also work).

## What to report

| item | where it comes from |
|---|---|
| parameters, unique blocks, block applications | `results.csv`: `total_params`, `unique_blocks`, `block_apps_final` |
| FLOPs | `gmacs_final` (multiply-accumulates of one forward pass; ×2 for FLOPs) |
| loop iterations and accuracy vs loop count | `acc_loop1 … acc_loopK` in `results.csv`, `per_exit.csv`, `curves.png` |
| mean ± std over 3 seeds | `aggregate.py` → `summary_mean_std.csv / .md` |
| looped vs non-looped | `ours` vs `plain12` and `shallow6` rows |
| GIPA vs looping alone | `ours` vs `loop_mhsa` (GIPA effect) and `ours` vs `untied_gipa6` (looping effect) |
| from-scratch evidence | the CIFAR-100 sweep |

Framing: *pretrained ViT → looped version keeps or improves accuracy with fewer unique parameters / adaptive depth
(early exits)*, backed by the from-scratch ablation.

## Practical notes

* Training cost on the RTX 3060 (dim 384, batch 32, 224 px): GIPA ≈ 100 img/s and 4.8 GiB, plain attention
  ≈ 390 img/s and 1.2 GiB. At 384 px GIPA is ≈ 15 img/s at batch 8 (7.9 GiB). CUB (~5k images after the val split)
  is therefore about a minute per epoch at 224 px with GIPA. FGVC papers often use 448 px; 224 or 384 is fine if
  stated and applied to every baseline.
* AMP (bfloat16 on Ampere) and gradient accumulation are on by default (`--amp`, `--grad-accum`).
* The recipe is fixed across variants: 50 epochs, lr 1e-4, layer decay 0.65, mixup 0.8 / cutmix 1.0, RandAugment,
  label smoothing 0.1, drop-path 0.1, early-stop patience 15.
* Other timm backbones work if dim / heads / mlp ratio match (`--backbone vit_small_patch16_224`); for ViT-B set
  `--dim 768 --num-heads 12` (and expect much higher memory use).
