# SDNET2018 crack classification

Binary task: **Cracked (1)** vs **Non-cracked (0)** on 256×256 patches of Decks, Pavements and Walls
(56,092 images, ~15 % cracked). Entry point: `train.py`.

## Data

Expected folder (already extracted next to the code):

```
Structural Defects Network (SDNET) 2018 archive/{Decks,Pavements,Walls}/{Cracked,Non-cracked}/<photo>-<patch>.jpg
```

Set `data_root` in the YAML (or `--data-root`).

| surface | Cracked | Non-cracked |
|---|---|---|
| Decks | 2,025 | 11,595 |
| Pavements | 2,608 | 21,726 |
| Walls | 3,851 | 14,287 |

**Splits** (70 / 15 / 15, stratified on surface × label, cached in `<output_dir>/_splits/split_<mode>_seed<seed>.csv`
so every run of the same seed sees the same images):

* `--split-mode group` (default): all patches of one source photo stay in one split. File names are
  `<photo>-<patch>.jpg`, and neighbouring patches of one photo are near-duplicates, so a random split leaks.
* `--split-mode random`: patch-level split like the Kaggle notebook. Use it only to reproduce those numbers.

Class imbalance is handled with balanced class weights on the loss (`--class-weights true`).

## Commands

```powershell
python train.py --summary-only                              # params / block applications
python train.py --debug-subset 600 --epochs 3 --output-dir runs_debug   # 1-2 min pipeline test

python train.py                                             # STACK 6x2 GIPA_FULL, from scratch (config.yaml)
python train.py --config config_sdnet_finetune.yaml         # same model, DeiT-S initialised

# comparison runs (same batch size and split seed everywhere)
python train.py --n-passes 1 --pass-embed false             # untied 6-block baseline
python train.py --n-prelude 1 --n-core 4 --n-coda 1         # SANDWICH 1 + 4x2 + 1
python train.py --n-core 3                                  # STACK 3x2 (same compute as untied 6)
python train.py --attention MHSA                            # looped MHSA
python train.py --attention MHSA --n-passes 1 --pass-embed false   # untied MHSA
python train.py --attention GIPA_S                          # GIPA component ablations (also GIPA_G/I/GI)

python train.py --scheduler plateau                         # ReduceLROnPlateau like the notebook
python train.py --monitor val_ap --early-stop-patience 12   # early-stop on PR-AUC
python train.py --monitor val_f2                            # favour crack recall
python train.py --early-stop-patience null                  # no early stopping
python train.py --batch-size 8 --grad-accum 2               # low memory
python train.py --resume runs/<run>/last.pt

python run_sweep.py --task sdnet --variants ours loop_mhsa shallow6 plain12 --seeds 42 43 44
python aggregate.py --root runs_ft
```

Fine-tune runs go to `runs_ft/`, from-scratch runs to `runs/`. Run folder name:
`<attention>_<prelude>-<core>x<passes>-<coda>_<init>_bs<effective batch>_<split>_s<seed>`.

## Early stopping

Validation is computed after every epoch on the final exit. Training stops when `--monitor`
(default `val_auc`) has not improved by more than `--early-stop-min-delta` for `--early-stop-patience`
epochs. `best.pt` is what the test evaluation uses. See the README for all monitor options.

## Metrics reported

Accuracy alone is misleading at 15 % prevalence, so the report leads with ranking metrics and crack-class metrics.

| group | metrics |
|---|---|
| threshold-free | ROC-AUC, PR-AUC (average precision) |
| crack class | precision, recall (sensitivity; missed cracks = `fn`), specificity, NPV, F1, F2, IoU (`tp/(tp+fp+fn)`), FPR, FNR |
| balanced | balanced accuracy, macro-F1, MCC, Cohen's kappa |
| calibration | Brier score, ECE (15 bins), NLL |
| uncertainty | 95 % stratified bootstrap CIs (`--bootstrap 1000`) for AUC, AP, F1, recall, precision, MCC |
| slices | per-surface (Decks / Pavements / Walls) table |
| looping | per-exit table (does the second pass help?), confidence-gated early-exit trade-off, learned gate values per pass |
| cost | parameters, unique blocks, block applications, GMACs, inference throughput |

**Operating points** (all thresholds chosen on the *validation* split, then applied to test):
`thr_0.5`, `thr_bestF1(val)` (`--threshold-beta 2` favours recall), and `thr_recall0.95(val)` (`--target-recall`),
the threshold that still catches 95 % of cracks with the fewest false alarms.

## Output files (`runs/<run>/`)

| file | content |
|---|---|
| `best.pt`, `last.pt` | best-monitor checkpoint; latest (for `--resume`) |
| `log.csv`, `history.json` | per-epoch train/val metrics |
| `config.json` | args + model config |
| `results.json`, `results.csv` | one row with everything (for ablation tables; used by `aggregate.py`) |
| `operating_points.csv` | test metrics at each threshold |
| `per_exit.csv`, `early_exit.csv`, `per_surface.csv`, `gates.csv` | the tables printed at the end of training |
| `test_predictions.csv` | per-image probability at every exit (for error analysis) |
| `curves.png cm.png roc.png pr.png calibration.png early_exit.png` | plots |

## Reading the results

* **Per-exit table:** `exit1` is the model after one pass (6 block applications), `final` after two (12). If
  `final` does not beat `exit1`, the second pass is not helping.
* **Gates:** each `lam_*` starts at 0.01; values that grow above ~0.05 mean the model uses that GIPA term.
  Compare pass 1 vs pass 2 rows to see whether the passes specialise.
* **Early exit:** `avg_block_apps` vs `f1` shows how much compute can be saved at a given confidence threshold.

## Fine-tuning vs from scratch

Ten thousand-ish cracked patches is small for a ViT, and published numbers are almost always from ImageNet-pretrained
backbones. `config_sdnet_finetune.yaml` initialises the six shared blocks from DeiT-S (layers *i* and *i+6*
averaged), uses ImageNet normalisation, layer-wise LR decay 0.75 and drop-path 0.1. See
[BENCHMARKS.md](BENCHMARKS.md) for how the mapping works and which baselines to run alongside it.

## CutMix, MixUp, SMOTE-style oversampling, resuming from any epoch

`--cutmix-alpha`, `--mixup-alpha`, `--imbalance off|oversample|smote` and `--smote-target` are documented in the
main README ("Batch augmentation and class imbalance"), together with "Resume from any epoch". A checkpoint is now
saved every epoch by default (`--save-every 1`), so `--resume runs_ft/<run>/epoch_0012.pt` restarts from epoch 12.

```powershell
python train.py --config config_sdnet_finetune.yaml --cutmix-alpha 1.0 --mixup-alpha 0.2 --imbalance smote
python train.py --config config_sdnet_finetune.yaml --resume runs_ft/<run>/epoch_0012.pt
```
