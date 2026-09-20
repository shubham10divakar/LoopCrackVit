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
| `train.py` | **SDNET2018** training / fine-tuning + full evaluation |
| `train_finetune.py` | **multi-class benchmarks** training / fine-tuning + full evaluation |
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
| `--save-every N` | 0 | also keep numbered checkpoints `epoch_NNNN.pt` |
| `--resume runs/<run>/last.pt` | – | restores model, optimizer, AMP scaler, scheduler, early-stop counter and history |

`best.pt` (best monitored epoch) is always the model used for the final test evaluation; the test set is never
used for early stopping, model selection or threshold tuning.

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
