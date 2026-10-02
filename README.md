# Looped CrackViT + GIPA

PyTorch training code for the **looped CrackViT with Geometry-Informed Predictive Attention (GIPA)**,
ported from the Kaggle TensorFlow notebook (`loopedcrackvit.ipynb`) and organised like the
TopoLoop-ViT repo. It trains and evaluates on

1. **SDNET2018 concrete-crack classification** (the problem statement) — from scratch *or* fine-tuned from a pretrained ViT, and
2. **fine-grained / standard benchmarks** (CUB-200, FGVC-Aircraft, Flowers-102, Food-101, CIFAR-100, any folder of images) to show the architecture holds up against the literature.

```
stem -> [CLS]+pos -> prelude -> [ core blocks ] x n_passes -> coda -> head
                                       |                        ^
                                       +-- after every pass: coda -> head   (early exits, one shared head)

STACK     n_prelude 0  n_core 6  n_coda 0  n_passes 2   6 unique blocks, 12 applications
SANDWICH  n_prelude 1  n_core 4  n_coda 1  n_passes 2   6 unique blocks, 10 applications
UNTIED    n_prelude 0  n_core 6  n_coda 0  n_passes 1   baseline
```

Detailed guides: **[docs/SDNET.md](docs/SDNET.md)** (crack problem: data, metrics, early stopping, outputs) ·
**[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** (pretrained-init strategy, datasets, controlled baselines, sweeps, what to report).

---

## Files

| file | what |
|---|---|
| `model.py` | `LoopedCrackViT`: conv/patch stem, GIPA and MHSA attention, pass embedding, per-pass gates, early exits |
| `pretrained.py` | maps a timm ViT (DeiT-S …) into the looped blocks (`avg` = Relaxed-Recursive style, `pick`) |
| `data.py` | SDNET pipeline (leak-free group split) and the multi-class image-folder pipeline |
| `metrics.py` | binary crack metrics, bootstrap CIs, threshold tuning, multi-class metrics, early-exit tables |
| `augment.py` | batch-level CutMix, MixUp and SMOTE-style minority synthesis for the SDNET trainer |
| `train.py` | **SDNET2018** training / fine-tuning + full evaluation |
| `train_finetune.py` | **multi-class benchmarks** training / fine-tuning + full evaluation |
| `evaluate.py` | re-scores a saved SDNET run on the test split → paper metrics table with bootstrap CIs |
| `downloads.py` | downloads and prepares CUB-200, Aircraft, Flowers-102, Food-101, CIFAR-100, PlantDoc, … |
| `run_sweep.py` | datasets × variants × seeds runner (resumable) |
| `aggregate.py` | mean ± std over seeds → CSV / markdown tables |
| `config.yaml` | SDNET, from scratch (matches the Kaggle notebook) |
| `config_sdnet_finetune.yaml` | SDNET, pretrained DeiT-S init |
| `config_finetune.yaml` | benchmarks, pretrained DeiT-S init |
| `config_scratch_cifar.yaml` | small from-scratch experiment (CIFAR-100) |

Every YAML key is also a CLI flag (underscores → dashes): `--n-core 4`, `--attention MHSA`, `--pass-embed false`.
Flags override the YAML.

---

## Install

Verified on this machine (Windows 11, Python 3.14, RTX 3060 12 GB, torch 2.14+cu126):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install scikit-learn pandas matplotlib pyyaml tqdm timm tabulate
pip install kaggle            # only for Kaggle-hosted datasets (plantvillage, cassava, appleleaf)
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```

`timm` downloads the pretrained weights from the Hugging Face hub on first use (internet needed once; cached after).
The optional `HF_TOKEN` env var only raises download rate limits.

---

## Quick start

```powershell
# 0. sanity: parameter counts, block applications (no data needed)
python train.py --summary-only
python train_finetune.py --summary-only --class-subset 200

# 1. SDNET, from scratch (headline STACK 6x2 GIPA, early stopping on val ROC-AUC)
python train.py

# 2. SDNET, fine-tuned from DeiT-S (recommended for the paper)
python train.py --config config_sdnet_finetune.yaml

# 3. benchmarks: prepare data, then fine-tune the looped model
python downloads.py --dataset cub200 aircraft
python train_finetune.py --dataset-dir datasets/cub200

# 4. the controlled comparison, 3 seeds each, then the tables
python run_sweep.py --datasets cub200 aircraft --seeds 42 43 44
python aggregate.py --root runs_bench
```

Pipeline smoke tests (≈1–2 min): `python train.py --debug-subset 600 --epochs 3 --output-dir runs_debug`
and `python train_finetune.py --dataset-dir datasets/cifar100 --class-subset 10 --max-per-class 100 --epochs 2 --output-dir runs_debug`.

---

## Reproduce the Kaggle notebook runs on this desktop

The notebook cells (`loopedcrackvit.ipynb`) train from scratch with Adam, ReduceLROnPlateau, a random patch-level
split, batch 16, 60 epochs, and early stopping on `val_final_auc` (patience 10). This command matches that setup:

```powershell
# STACK 6x2, GIPA_FULL (the "STACK LOOP" cell)
python train.py --split-mode random --scheduler plateau --weight-decay 0 --grad-clip 0 --early-stop-min-delta 0 --early-stop-patience 10 --monitor val_auc --lr 0.0003 --batch-size 16 --epochs 60

# SANDWICH 1 + 4x2 + 1 (the "SANDWICH LOOP" cell): same flags plus
python train.py --split-mode random --scheduler plateau --weight-decay 0 --grad-clip 0 --early-stop-min-delta 0 --early-stop-patience 10 --monitor val_auc --lr 0.0003 --batch-size 16 --epochs 60 --n-prelude 1 --n-core 4 --n-coda 1
```

Output goes to `runs/gipa_full_0-6x2-0_conv_bs16_random_s42/` (sandwich: `gipa_full_1-4x2-1_conv_bs16_random_s42/`).

| flag | matches in the notebook |
|---|---|
| `--split-mode random` | patch-level `train_test_split` 70/15/15, seed 42 (the default `group` split avoids leakage; use it for the paper) |
| `--scheduler plateau` | `ReduceLROnPlateau(factor 0.5, patience 4, min_lr 1e-6)` on the validation loss |
| `--weight-decay 0 --grad-clip 0` | plain Adam, no gradient clipping |
| `--early-stop-patience 10 --early-stop-min-delta 0 --monitor val_auc` | `EarlyStopping` on `val_final_auc`, patience 10 |
| `--lr 0.0003 --batch-size 16 --epochs 60` | `LR_INIT`, `BATCH_SIZE`, `EPOCHS` |

Model defaults in `config.yaml` already match the notebook (GIPA_FULL, conv stem, dim 256, 8 heads, MLP ratio 2.0,
dropout 0.1, pass embedding and per-pass gates on, aux weight 0.3, label smoothing 0.05, balanced class weights).

Expect small differences from the Kaggle numbers: the split uses the same sklearn calls and seed but a differently
ordered file list, so the exact images differ; augmentation is torchvision (rotations and shifts fill with black
instead of Keras "nearest", inputs normalised to −1..1 instead of 0..1); initialisation and random streams differ.
The Kaggle log shows about 9 minutes per epoch on a T4, and this GPU should be similar, so a full 60-epoch run can
take up to about 9 hours unless early stopping ends it sooner.

---

## Model variants (all flags of `model.py`)

| flag | default | meaning |
|---|---|---|
| `--attention` | `GIPA_FULL` | `MHSA`, `GIPA_S`, `GIPA_G`, `GIPA_I`, `GIPA_GI`, `GIPA_FULL` |
| `--n-prelude / --n-core / --n-coda / --n-passes` | 0 / 6 / 0 / 2 | untied prelude, shared core, untied coda, times the core is applied |
| `--pass-embed` | true | zero-init learned vector per pass (turn **off** for `--n-passes 1` baselines) |
| `--per-pass-gates` | true | GIPA gates get one row per pass; q/k/v, tensor head, W stay shared |
| `--input-injection` | false | re-add the loop input on passes ≥ 2 |
| `--aux-weight` | 0.3 | loss weight on early exits (final exit = 1.0) |
| `--stem` | conv | `conv` (from scratch, crack-friendly) or `patch` (needed for pretrained init) |
| `--coherence-reg` | 0.0 | orientation-smoothness penalty (try 1e-3) |
| `--drop-path` | 0.0 / 0.1 | stochastic depth over the unique blocks |

GIPA adds `lam_c, lam_a, lam_i, gamma, kappa` gates (init 0.01, so a pretrained ViT starts as a plain ViT and
*learns* whether to use geometry). The learned values are printed and saved in `gates.csv`.

---

## Early stopping, resuming, checkpoints (both trainers)

| flag | default | meaning |
|---|---|---|
| `--monitor` | `val_auc` (SDNET) / `val_acc` (benchmarks) | metric on the **validation split, final exit**. SDNET: `val_auc val_ap val_f1 val_f2 val_mcc val_bal_acc val_recall val_loss`; benchmarks: `val_acc val_top5 val_f1_macro val_bal_acc val_loss` |
| `--early-stop-patience` | 10 / 8 / 15 | stop after N epochs without improvement (`null` = never) |
| `--early-stop-min-delta` | 0.0005 / 0 | minimum improvement that counts |
| `--save-every N` | **1** | keep a numbered checkpoint `epoch_NNNN.pt` every N epochs (1 = every epoch, 0 = only `best.pt` / `last.pt`) |
| `--resume <checkpoint>` | – | restores model, optimizer, AMP scaler, scheduler, early-stop counter and history |

`best.pt` (best monitored epoch) is always the model used for the final test evaluation; the test set is never
used for early stopping, model selection or threshold tuning.

### Resume from any epoch

Every epoch is saved by default, so you can restart training from whichever epoch you want:

```powershell
# what is there?
dir runs_ft\gipa_full_0-6x2-0_ft-avg_bs32_group_s42\epoch_*.pt

# continue from epoch 12 (SDNET; benchmarks work the same with train_finetune.py)
python train.py --config config_sdnet_finetune.yaml --resume runs_ft/gipa_full_0-6x2-0_ft-avg_bs32_group_s42/epoch_0012.pt

# continue from the newest epoch / from the best epoch
python train.py --config config_sdnet_finetune.yaml --resume runs_ft/gipa_full_0-6x2-0_ft-avg_bs32_group_s42/last.pt
python train.py --config config_sdnet_finetune.yaml --resume runs_ft/gipa_full_0-6x2-0_ft-avg_bs32_group_s42/best.pt
```

* Resuming from epoch N continues at epoch N+1 with the same learning-rate schedule position, optimizer state and
  early-stopping counters. Use exactly the flags of the original run (a changed model config is refused; a changed
  learning rate or batch size is not caught). To branch with different settings, change `--seed` or an
  augmentation flag so the run gets a new folder name.
* Resuming from an older epoch than the newest one is safe: `log.csv` is rewritten from that epoch's history (no
  duplicated epochs) and `best.pt` is reset to the best epoch within it (copied from `epoch_NNNN.pt`). Epochs after
  the resume point are overwritten as training proceeds.
* `--epochs` must be larger than the epoch you resume from, otherwise it goes straight to the final test evaluation.
* **Disk:** each checkpoint holds the weights and the Adam state. Measured 45 MB per epoch for the from-scratch
  SDNET model (3.9M parameters); the DeiT-S-sized fine-tune models (about 11M parameters) come to roughly 3× that
  (an estimate from parameter count, not measured), i.e. a few GB per 50-epoch run. Use `--save-every 5` (or 0)
  to save less.
* Not restored: the random-number state, so a resumed run is not bit-identical to an uninterrupted one.

### Evaluate a saved run (SDNET)

If training stopped before the final test evaluation (no `results.json` in the run folder), or you want to score
another checkpoint, `evaluate.py` rebuilds the model from `config.json`, reuses the cached split in `runs/_splits`,
tunes thresholds on VAL only and scores TEST:

```powershell
python evaluate.py --run runs/gipa_full_0-6x2-0_conv_bs16_random_smote_s42

# another checkpoint / on the CPU while the GPU is busy / skip the bootstrap CIs
python evaluate.py --run runs/gipa_full_0-6x2-0_conv_bs16_random_smote_s42 --ckpt epoch_0051.pt
python evaluate.py --run runs/gipa_full_0-6x2-0_conv_bs16_random_smote_s42 --device cpu
python evaluate.py --run runs/gipa_full_0-6x2-0_conv_bs16_random_smote_s42 --bootstrap 0
```

Writes to `<run>/eval/`: `metrics_table.md` / `.csv` (ROC-AUC, PR-AUC, crack recall and precision, specificity,
MCC, Brier score, ECE, parameters (M), block applications at the final exit; threshold-dependent metrics at 0.5 and
at the val-tuned F1 threshold, each with a 95% stratified-bootstrap CI), `operating_points.csv`, `per_exit.csv`,
`test_predictions.csv` and `metrics.json`. Inference runs in fp32 unless you pass `--amp`. Predictions are cached in
`eval/_preds_<ckpt>.npz`; pass `--no-cache` to recompute them.

---

## Batch augmentation and class imbalance (SDNET trainer)

CutMix, MixUp and two imbalance options are flags of `train.py`. All are **off by default**, so existing commands
behave as before. They apply to training batches only; validation and test images are never mixed.

| flag | default | meaning |
|---|---|---|
| `--cutmix-alpha A` | 0 (off) | CutMix: paste a random box from another image, label weighted by box area. `1.0` is the standard value |
| `--mixup-alpha A` | 0 (off) | MixUp: blend two images and their labels. `0.2` is a common start |
| `--mixup-prob P` | 1.0 | probability that a batch is mixed at all |
| `--mixup-switch-prob P` | 0.5 | when both are on: probability of using CutMix (otherwise MixUp) |
| `--imbalance off\|oversample\|smote` | `off` | see below |
| `--smote-target F` | 0.5 | `smote`: fraction of each batch that should be cracked after synthesis |
| `--smote-bank N` | 256 | `smote`: how many recent cracked images are kept to interpolate between |

* `oversample` draws images with probability inversely proportional to their class size, so batches are about 50 %
  cracked (some cracked images repeat within an epoch).
* `smote` is **SMOTE-style, in pixel space**. Classic SMOTE interpolates between a minority sample and one of its
  nearest neighbours in feature space; on raw 224×224×3 images that is impractical, so this blends two random
  cracked images from the rolling bank (hard label 1) and appends the synthetic images to the batch until the
  cracked fraction reaches `--smote-target`. At most half a batch is added (a batch can grow to 1.5× the size, so
  memory rises accordingly).
* With `oversample` or `smote` the loss class weights are switched off automatically (a note is printed), since
  the classes are already rebalanced; use `--class-weights false` yourself if you combine them with something else.
* Runs get a tag in the folder name (`..._cut1_mix0.2_smote_s42`) so they do not overwrite each other.
* Caveat for cracks: CutMix labels follow box *area*, but a hairline crack covers few pixels, so a pasted box can
  remove the crack while the label still says "partly cracked". Treat both as regularisers to be judged on the
  validation split, not as guaranteed improvements; run them as ablations against the plain baseline.

```powershell
# fine-tune with CutMix (+ a little MixUp)
python train.py --config config_sdnet_finetune.yaml --cutmix-alpha 1.0 --mixup-alpha 0.2

# CutMix only
python train.py --config config_sdnet_finetune.yaml --cutmix-alpha 1.0

# SMOTE-style minority synthesis (50 % cracked per batch)
python train.py --config config_sdnet_finetune.yaml --imbalance smote

# balanced sampling instead
python train.py --config config_sdnet_finetune.yaml --imbalance oversample

# everything together
python train.py --config config_sdnet_finetune.yaml --cutmix-alpha 1.0 --mixup-alpha 0.2 --imbalance smote --smote-target 0.5

# from scratch (config.yaml) works the same way
python train.py --cutmix-alpha 1.0 --imbalance oversample
```

The multi-class benchmark trainer (`train_finetune.py`) already has `--mixup-alpha`, `--cutmix-alpha`, `--mixup-prob`
and `--mixup-switch-prob` (on by default in `config_finetune.yaml`); it has no SMOTE or oversampling option since the
benchmark datasets are roughly balanced.

---

## Speed and memory (measured on the RTX 3060 12 GB, shared with the desktop)

| model | batch | resolution | peak GPU memory | training throughput |
|---|---|---|---|---|
| looped 6×2 **GIPA**, dim 384 | 32 | 224 | 4.8 GiB | ~100 img/s |
| looped 6×2 GIPA, dim 384 | 16 | 224 | 2.5 GiB | ~94 img/s |
| looped 6×2 MHSA, dim 384 | 32 | 224 | 1.2 GiB | ~390 img/s |
| looped 6×2 GIPA, dim 384 | 8 | 384 | 7.9 GiB | ~15 img/s |

* GIPA is about **4× slower to train than plain attention** in this implementation (fp32 bias maps of shape
  `B×heads×197×197` at every block application). Budget for it; 224 px is practical, 384 px is very slow.
* SDNET from scratch (dim 256) ran at ~4.7 it/s with batch 16 in the smoke test ≈ 8 min per epoch on the full
  training split; early stopping usually ends the run well before 60 epochs.
* Out of memory → lower `--batch-size` and raise `--grad-accum` (effective batch = product).

---

## Troubleshooting

| symptom | fix |
|---|---|
| `torch.cuda.is_available()` is `False` | CPU wheel installed: `pip uninstall torch torchvision`, reinstall from the cu126 index |
| `ModuleNotFoundError: timm` | `pip install timm` |
| `pretrained init needs stem='patch'` | use `--config config_finetune.yaml` / `config_sdnet_finetune.yaml`, or `--stem patch` |
| `set dim/num_heads/mlp_ratio to match` | the backbone (DeiT-S) is dim 384 / 6 heads / mlp 4.0; use those or another backbone with matching sizes |
| `--resume: checkpoint model config differs` | resume with exactly the flags used to start the run |
| `MemoryError` / `DataLoader worker exited unexpectedly` | Windows spawns a full Python+torch process per worker (~1 GB commit each). Default is `--num-workers 4` (val/test loaders use 2 and the train workers are freed before testing); lower it to 2 if the commit charge is tight |
| Kaggle downloads fail | put your token at `~/.kaggle/kaggle.json`; competitions need their rules accepted once on the website |

---

## Provenance and known differences from the Kaggle notebook

* PyTorch port; the notebook model is reproduced (3.91M parameters for STACK 6×2, 10-block sandwich, same gate
  parametrisation and early exit). A 12-block MHSA model loaded from DeiT-S reproduces timm's features exactly
  (max abs difference 0.0), so the pretrained mapping is verified.
* Cracked is class **1** (the notebook had Cracked = 0, so its sigmoid was P(Non-cracked)).
* Default SDNET split is **grouped by source photo** (no neighbouring-patch leakage); `--split-mode random`
  reproduces the notebook's patch-level split.
* Decision thresholds are tuned on the validation split and applied to test (the notebook tuned on test).
* AdamW + warmup/cosine instead of Adam + ReduceLROnPlateau (`--scheduler plateau` restores the latter).
* Not ported: attention-rollout and orientation-field visualisations.
