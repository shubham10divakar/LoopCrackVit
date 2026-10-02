"""
SDNET2018 data pipeline (binary: Cracked = 1 / Non-cracked = 0).

Layout: <root>/{Decks,Pavements,Walls}/{Cracked,Non-cracked}/<image>-<patch>.jpg

SDNET files are patches cut from larger photos ("7001-115.jpg" = photo 7001, patch 115).
A random patch-level split therefore leaks near-identical neighbouring patches between
train and test. split_mode:
    group   (default) - all patches of one source photo stay in the same split
    random            - patch-level split, identical in spirit to the Kaggle notebook
    balanced          - CrackNeXt protocol: per surface, undersample Non-cracked to the Cracked count
                        (~17k images, 50/50), then the patch-level random split. A separate
                        benchmark, not comparable to metrics on the full imbalanced set.
All are stratified on surface x label. The split is cached to a CSV so every run of the
same seed/mode sees exactly the same images.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

SURFACES = ["Decks", "Pavements", "Walls"]
LABELS = {"Non-cracked": 0, "Cracked": 1}
CLASS_NAMES = ["Non-cracked", "Cracked"]
NORMS = {"half": ((0.5,) * 3, (0.5,) * 3),                       # from-scratch default
         "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}  # required for pretrained ViTs


def scan_sdnet(root):
    rows = []
    for s in SURFACES:
        for lab, y in LABELS.items():
            folder = os.path.join(root, s, lab)
            if not os.path.isdir(folder):
                raise FileNotFoundError(folder)
            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append(dict(path=os.path.join(folder, f), label=y, surface=s,
                                     group=f"{s}_{f.rsplit('-', 1)[0]}"))
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    df["strat"] = df["surface"] + "_" + df["label"].astype(str)
    return df


def undersample(df, seed=42):
    """Within each surface, randomly keep as many Non-cracked images as there are Cracked ones
    (the CrackNeXt SDNET2018 protocol). Balancing happens BEFORE the train/val/test split."""
    keep = []
    for _, g in df.groupby("surface"):
        pos, neg = g[g["label"] == 1], g[g["label"] == 0]
        keep += [pos, neg.sample(n=min(len(pos), len(neg)), random_state=seed)]
    return pd.concat(keep).sort_values("path").reset_index(drop=True)


def make_split(df, mode="group", seed=42, val_frac=0.15, test_frac=0.15):
    if mode == "balanced":      # per-surface undersampling to 50/50, then the patch-level random split
        return make_split(undersample(df, seed), "random", seed, val_frac, test_frac)
    if mode == "random":
        tr, tmp = train_test_split(df, test_size=val_frac + test_frac, random_state=seed,
                                   stratify=df["strat"])
        va, te = train_test_split(tmp, test_size=test_frac / (val_frac + test_frac),
                                  random_state=seed, stratify=tmp["strat"])
        out = df.copy()
        out["split"] = "train"
        out.loc[va.index, "split"] = "val"
        out.loc[te.index, "split"] = "test"
        return out
    if mode != "group":
        raise ValueError("split_mode must be 'group', 'random' or 'balanced'")
    k = int(round(1 / min(val_frac, test_frac)))          # 15% -> ~7 folds
    folds = np.full(len(df), -1)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, idx) in enumerate(sgkf.split(df, df["strat"], df["group"])):
        folds[idx] = i
    out = df.copy()
    out["split"] = "train"
    out.loc[folds == 0, "split"] = "test"
    out.loc[folds == 1, "split"] = "val"
    return out


def load_split(root, mode, seed, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"split_{mode}_seed{seed}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        if os.path.exists(df["path"].iloc[0]):
            return df
    df = make_split(scan_sdnet(root), mode, seed)
    df.to_csv(cache, index=False)
    return df


def build_transforms(size, augment=True, norm="half"):
    ev = [v2.Resize((size, size), antialias=True)]
    tail = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(*NORMS[norm])]
    if not augment:
        return v2.Compose(ev + tail)
    return v2.Compose(ev + [
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
        v2.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        v2.ColorJitter(brightness=(0.8, 1.2)),
    ] + tail)


class CrackDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].to_numpy().astype("float32")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = self.transform(im.convert("RGB"))
        return img, self.labels[i]


def loader_kw(workers, persistent, pin_memory):
    """Windows spawns a full Python+torch process per worker (~1+ GB commit each), so keep the
    total small: train loader persistent, val loader 2 workers, test loader not persistent."""
    return dict(num_workers=workers, pin_memory=pin_memory, persistent_workers=persistent and workers > 0,
                prefetch_factor=4 if workers > 0 else None)


def build_loaders(df, image_size, batch_size, num_workers, augment=True, pin_memory=True, seed=42,
                  norm="half", oversample=False):
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    gen = torch.Generator().manual_seed(seed)
    if oversample:      # every image drawn with probability ~ 1/(size of its class): batches are ~50% cracked
        counts = tr["label"].value_counts()
        wts = torch.tensor(tr["label"].map(lambda c: 1.0 / counts[c]).to_numpy(), dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=len(tr), replacement=True, generator=gen)
        order = dict(sampler=sampler)
    else:
        order = dict(shuffle=True, generator=gen)
    train_loader = DataLoader(CrackDataset(tr, build_transforms(image_size, augment, norm)),
                              batch_size=batch_size, drop_last=True, **order,
                              **loader_kw(num_workers, True, pin_memory))
    ev_tf = build_transforms(image_size, False, norm)
    val_loader = DataLoader(CrackDataset(va, ev_tf), batch_size=batch_size * 2, shuffle=False,
                            **loader_kw(min(2, num_workers), True, pin_memory))
    test_loader = DataLoader(CrackDataset(te, ev_tf), batch_size=batch_size * 2, shuffle=False,
                             **loader_kw(min(2, num_workers), False, pin_memory))
    return train_loader, val_loader, test_loader, (tr, va, te)


def describe_split(df):
    t = df.groupby(["split", "surface", "label"]).size().unstack(fill_value=0)
    t.columns = ["Non-cracked", "Cracked"]
    return t


def data_summary(df, mode, root=None):
    """Printable dataset report: per split x surface counts with totals and crack prevalence.
    For split_mode 'balanced' it also shows the original folder counts and what undersampling dropped."""
    def table(d, by):
        t = d.groupby(by + ["label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        t.columns = ["Non-cracked", "Cracked"]
        t["Total"] = t.sum(1)
        t["Cracked %"] = (100 * t["Cracked"] / t["Total"]).round(1)
        return t

    order = ["train", "val", "test"]
    lines = [f"=== DATA  split_mode={mode} ===", "per split x surface:",
             table(df, ["split", "surface"]).reindex(order, level=0).to_string(), "", "per split:"]
    per = table(df, ["split"]).reindex(order)
    per["% of data"] = (100 * per["Total"] / len(df)).round(1)
    lines += [per.to_string(), "",
              f"TOTAL {len(df):,} images: {int((df.label == 0).sum()):,} Non-cracked + "
              f"{int(df.label.sum()):,} Cracked ({100 * df.label.mean():.1f}% cracked)"]
    if mode == "balanced" and root:
        full = scan_sdnet(root)
        orig, kept = table(full, ["surface"]), table(df, ["surface"])
        cmp = orig[["Non-cracked", "Cracked", "Total"]].add_prefix("orig ").join(kept[["Non-cracked", "Cracked", "Total"]].add_prefix("kept "))
        cmp["dropped Non-cracked"] = cmp["orig Non-cracked"] - cmp["kept Non-cracked"]
        lines += ["", "undersampling (per surface, Non-cracked cut to the Cracked count, before splitting):",
                  cmp.to_string(),
                  f"kept {len(df):,} of {len(full):,} images ({100 * len(df) / len(full):.1f}%); "
                  f"dropped {len(full) - len(df):,} Non-cracked"]
    return "\n".join(lines)


# =============================================================================
# Multi-class image-folder benchmarks (CUB-200, FGVC-Aircraft, Flowers, Food, CIFAR-100 ...)
# Layout produced by downloads.py:  datasets/<name>/train/<class>/*.jpg [+ test/<class>/*.jpg]
# =============================================================================
class FolderDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), y


def folder_transforms(size, augment="rand", norm="imagenet"):
    tail = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(*NORMS[norm])]
    ev = v2.Compose([v2.Resize(int(round(size / 0.875)), antialias=True), v2.CenterCrop(size)] + tail)
    if augment == "none":
        return v2.Compose([v2.Resize((size, size), antialias=True)] + tail), ev
    tr = [v2.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True), v2.RandomHorizontalFlip()]
    if augment == "rand":
        tr.append(v2.RandAugment(num_ops=2, magnitude=9))
    elif augment == "trivial":
        tr.append(v2.TrivialAugmentWide())
    return v2.Compose(tr + tail), ev


def build_folder_loaders(dataset_dir=None, train_dir=None, test_dir=None, image_size=224, batch_size=32,
                         num_workers=8, val_split=0.1, test_split=0.2, augment="rand", norm="imagenet",
                         num_classes=None, max_per_class=None, seed=42, pin_memory=True):
    """-> train_loader, val_loader, test_loader, class_names.
    val is ALWAYS held out of train (stratified); early stopping never sees the test set.
    If the dataset ships no test/ folder, `test_split` of train is held out as test too."""
    import random
    from collections import defaultdict
    from torchvision.datasets import ImageFolder

    if dataset_dir:
        tr_d, te_d = os.path.join(dataset_dir, "train"), os.path.join(dataset_dir, "test")
        train_dir = train_dir or (tr_d if os.path.isdir(tr_d) else dataset_dir)
        test_dir = test_dir or (te_d if os.path.isdir(te_d) else None)
    if not train_dir:
        raise SystemExit("give --dataset-dir (or --train-dir)")
    rng = random.Random(seed)
    base = ImageFolder(train_dir)
    classes = base.classes[:num_classes] if num_classes else base.classes
    lab = {c: i for i, c in enumerate(classes)}
    by = defaultdict(list)
    for p, i in base.samples:
        if base.classes[i] in lab:
            by[base.classes[i]].append(p)
    tr, va, te = [], [], []
    for c in classes:
        ps = sorted(by[c])
        rng.shuffle(ps)
        if max_per_class:
            ps = ps[:max_per_class]
        n_te = 0 if test_dir else max(1, int(round(len(ps) * test_split)))
        n_va = max(1, int(round(len(ps) * val_split)))
        te += [(p, lab[c]) for p in ps[:n_te]]
        va += [(p, lab[c]) for p in ps[n_te:n_te + n_va]]
        tr += [(p, lab[c]) for p in ps[n_te + n_va:]]
    if test_dir:
        tb = ImageFolder(test_dir)
        te = [(p, lab[tb.classes[i]]) for p, i in tb.samples if tb.classes[i] in lab]
    tr_tf, ev_tf = folder_transforms(image_size, augment, norm)
    g = torch.Generator().manual_seed(seed)
    loaders = (DataLoader(FolderDataset(tr, tr_tf), batch_size=batch_size, shuffle=True,
                          drop_last=len(tr) > batch_size, generator=g, **loader_kw(num_workers, True, pin_memory)),
               DataLoader(FolderDataset(va, ev_tf), batch_size=batch_size * 2, shuffle=False,
                          **loader_kw(min(2, num_workers), True, pin_memory)),
               DataLoader(FolderDataset(te, ev_tf), batch_size=batch_size * 2, shuffle=False,
                          **loader_kw(min(2, num_workers), False, pin_memory)))
    print(f"[data] {len(classes)} classes | train {len(tr)} | val {len(va)} (held out of train) | "
          f"test {len(te)} ({'official test/' if test_dir else f'{test_split:.0%} of train'})")
    return (*loaders, classes)
