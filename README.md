# LoopCrackViT: looped CrackViT with Geometry-Informed Predictive Attention (GIPA)

PyTorch code for **LoopCrackViT**, a compact vision transformer (3.91M parameters, trained from scratch) for binary
concrete-crack classification on SDNET2018. Six shared transformer blocks with GIPA attention are applied twice;
a shared head after each pass gives an early exit.

```
conv stem -> [CLS]+pos -> [ 6 GIPA blocks ] x 2 passes -> head      (head also after pass 1 = early exit)
```

More detail: [docs/SDNET.md](docs/SDNET.md) (data, metrics, outputs) and
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) (pretrained init, multi-class benchmarks).

---

## Paper protocol: balanced SDNET2018

The paper uses the **class-balanced SDNET2018 subset**, following the CrackNeXt protocol. Within each surface (deck,
pavement, wall), a random set of Non-cracked images the same size as the Cracked set is kept. The balanced set is
then split 70/15/15, stratified on surface × label (seed 42):

|            | Non-cracked | Cracked | Total  |
|------------|------------:|--------:|-------:|
| Train      | 5,939       | 5,938   | 11,877 |
| Validation | 1,272       | 1,273   | 2,545  |
| Test       | 1,273       | 1,273   | 2,546  |
| **Total**  | **8,484**   | **8,484** | **16,968** (of 56,092) |

The split is cached in `runs/_splits/split_balanced_seed42.csv`. A data summary is printed at the start of every run:
per-split counts, crack share, and how many images undersampling dropped.

### 1. Train (3 seeds)

```powershell
python train.py --split-mode balanced --scheduler plateau --weight-decay 0 --grad-clip 0 --early-stop-min-delta 0 --early-stop-patience 10 --monitor val_auc --lr 0.0003 --batch-size 16 --epochs 60 --seed 42
```

Repeat with `--seed 43` and `--seed 44`. Each run writes `runs/gipa_full_0-6x2-0_conv_bs16_balanced_s<seed>/`.
Everything printed to the console (data summary, per-epoch lines, test tables, errors) is also saved to
`<run>/train_log.txt`; a resumed run appends to it.
The classes are already 50/50, so leave `--imbalance` off.

### 2. Paper outputs (written automatically at the end of training)

`<run>/paper/` holds a short table and four figures. Every thresholded number comes from **one** threshold, chosen
on validation (best F1), and **one** test confusion matrix, so the numbers all agree:

| file | content |
|---|---|
| `metrics.md` / `.csv` / `.json` | accuracy, macro precision / recall / F1, ROC-AUC, PR-AUC, crack recall and precision, specificity, MCC, Brier, ECE, confusion counts, threshold, parameters, block applications |
| `fig_training_curves.png` | train / validation loss and validation ROC-AUC / MCC per epoch |
| `fig_roc_pr.png` | ROC and precision-recall curves with the operating point |
| `fig_reliability.png` | reliability diagram (15 bins) with Brier and ECE |
| `fig_confusion.png` | test confusion matrix |

### 3. Same outputs from a saved checkpoint

For a run that was interrupted, or for another checkpoint (e.g. the last epoch):

```powershell
python evaluate.py --run runs/gipa_full_0-6x2-0_conv_bs16_balanced_s42                    # best.pt -> <run>/paper/
python evaluate.py --run runs/gipa_full_0-6x2-0_conv_bs16_balanced_s42 --ckpt last.pt     # -> <run>/paper_last/
```

`evaluate.py` also writes the longer tables (bootstrap CIs, operating points, per exit) to `<run>/eval/`.

### 4. Mean ± s.d. over seeds

```powershell
python paper.py runs/gipa_full_0-6x2-0_conv_bs16_balanced_s42 runs/gipa_full_0-6x2-0_conv_bs16_balanced_s43 runs/gipa_full_0-6x2-0_conv_bs16_balanced_s44 --out runs/paper_mean_std.md
```

Other split modes (`--split-mode`): `random` uses the full imbalanced dataset (56,092 images, 15.1% cracked) with
a patch-level split; `group` (default) keeps every patch of a source photo in the same split. Results on different
split modes are not comparable.

---

## Install

Tested on Windows 11, Python 3.14, RTX 3060 12 GB, torch 2.14+cu126.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install scikit-learn pandas matplotlib pyyaml tqdm timm tabulate
```

Put the dataset at `Structural Defects Network (SDNET) 2018 archive/{Decks,Pavements,Walls}/{Cracked,Non-cracked}/`
or set `data_root` in `config.yaml`. Smoke test (about 1 min):
`python train.py --split-mode balanced --debug-subset 64 --epochs 1 --bootstrap 0 --output-dir runs_debug`.

---

## Files

| file | what |
|---|---|
| `model.py` | `LoopedCrackViT`: conv stem, GIPA / MHSA attention, pass embedding, per-pass gates, early exits |
| `data.py` | SDNET2018 scan, split modes (`balanced`, `random`, `group`), loaders, data summary |
| `train.py` | SDNET2018 training + final test evaluation + paper outputs |
| `evaluate.py` | re-scores a saved run / checkpoint on the test split |
| `paper.py` | paper metrics table and figures; mean ± s.d. over seeds |
| `metrics.py` | metrics, threshold tuning, bootstrap CIs, early-exit tables |
| `augment.py` | CutMix, MixUp, SMOTE-style minority synthesis |
| `gradcam.py` | Grad-CAM figures and deletion / insertion faithfulness ([gradcam/README.md](gradcam/README.md)) |
| `config.yaml` | SDNET2018 from-scratch configuration; every key is also a CLI flag (`n_core` → `--n-core`) |
| `train_finetune.py`, `pretrained.py`, `downloads.py`, `run_sweep.py`, `aggregate.py`, `config_*.yaml` | pretrained init and multi-class benchmarks ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)) |
| `weights/` | released weights ([weights/README.md](weights/README.md)); the current SDNET2018 release is from a `random`-split run with `--imbalance smote`, not the balanced protocol |

---

## Model and ablation flags

| flag | default | meaning |
|---|---|---|
| `--attention` | `GIPA_FULL` | `MHSA`, `GIPA_S`, `GIPA_G`, `GIPA_I`, `GIPA_GI`, `GIPA_FULL` |
| `--n-prelude / --n-core / --n-coda / --n-passes` | 0 / 6 / 0 / 2 | untied prelude, shared core, untied coda, passes over the core |
| `--pass-embed`, `--per-pass-gates` | true | pass embedding; one gate row per pass |
| `--aux-weight` | 0.3 | loss weight of the early exit (final exit = 1.0) |
| `--imbalance off\|oversample\|smote` | off | rebalancing for the imbalanced split modes (not needed with `balanced`) |
| `--cutmix-alpha`, `--mixup-alpha` | 0 | batch mixing (off) |

Ablations from the paper: `--attention MHSA`, `--n-passes 1`, `--n-core 3`, and the sandwich
`--n-prelude 1 --n-core 4 --n-coda 1`.

---

## Checkpoints and resuming

`best.pt` (best validation ROC-AUC) is the model evaluated on test; the test set is never used for early stopping,
checkpoint choice or thresholds. `last.pt` and `epoch_NNNN.pt` (every epoch, `--save-every`) are kept too.

```powershell
python train.py <same flags as the original run> --resume runs/<run>/last.pt      # or any epoch_NNNN.pt
```

Resuming restores the model, optimizer, scheduler, early-stop state and history; use exactly the original flags.
Each checkpoint is about 45 MB.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `torch.cuda.is_available()` is `False` | CPU wheel installed; reinstall torch from the cu126 index |
| `MemoryError` / `DataLoader worker exited unexpectedly` | Windows starts a full process per worker; use `--num-workers 2` |
| out of GPU memory | lower `--batch-size` and raise `--grad-accum` (effective batch = product) |
| `--resume: checkpoint model config differs` | resume with the flags used to start the run |
