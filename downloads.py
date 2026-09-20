"""
Download and prepare classification datasets into a plain ImageFolder layout
that train_finetune.py / data.py understand:

    datasets/<name>/train/<class>/*.jpg
    datasets/<name>/test/<class>/*.jpg      (when the dataset ships a test split)

For datasets with no official test split, only train/ is written and train.py's
--val-split holds out a validation set. Use any prepared dir with:

    python train_finetune.py --dataset-dir datasets/<name>

Usage
-----
    python downloads.py --dataset all
    python downloads.py --dataset plantvillage plantdoc
    python downloads.py --dataset cassava --kaggle-dataset <owner>/<slug>
    python downloads.py --dataset appleleaf --kaggle-dataset <owner>/<slug>
    python downloads.py --root /path/to/datasets --dataset cub200

Datasets
--------
  plantvillage  PlantVillage leaf disease, up to 38 classes   Kaggle dataset (foldered)
  plantdoc      PlantDoc, 27 classes, real train/test          GitHub zip, automatic
  cassava       Cassava Leaf Disease, 5 classes                Kaggle competition (csv labels)
  appleleaf     Apple leaf disease                             Kaggle dataset (foldered)
  cub200        Caltech-UCSD Birds-200-2011, 200 classes       Caltech tarball, automatic
  aircraft      FGVC-Aircraft, 100 variants                    torchvision, automatic
  flowers102    Oxford Flowers-102, 102 classes                torchvision, automatic
  food101       Food-101, 101 classes                          torchvision, automatic
  cifar100      CIFAR-100, 100 classes (from-scratch experiment) torchvision, automatic

Raw downloads are cached under datasets/_raw/<name>/. Per-class images are
hardlinked (not copied) into datasets/<name>/... when the two live on the same
drive; pass --copy to force real copies. Re-running is idempotent: a dataset
whose output folder already has files is skipped.

Kaggle datasets/competitions
-----------------------------
    pip install kaggle
    # token at https://www.kaggle.com/settings -> "Create New Token"
    # save it to ~/.kaggle/kaggle.json  (chmod 600)
Competition datasets (cassava) also require accepting the competition rules on
the Kaggle website once, from the account that owns the token.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INVALID_CHARS = '<>:"/\\|?*'


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sanitize(name: str) -> str:
    for c in INVALID_CHARS:
        name = name.replace(c, "_")
    return name.strip() or "unnamed"


def _link_or_copy(src: Path, dst: Path, link: bool):
    if dst.exists():
        return
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _export(image_files, labels, class_names, out_dir: Path, link: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, label in zip(image_files, labels):
        path = Path(path)
        dst_dir = out_dir / _sanitize(str(class_names[label]))
        dst_dir.mkdir(exist_ok=True)
        _link_or_copy(path, dst_dir / path.name, link)
    n = sum(1 for p in out_dir.rglob("*") if p.is_file())
    print(f"[downloads] {out_dir}: {n} images across {len(set(labels))} classes")


def _copy_class_tree(src_root: Path, out_dir: Path, link: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        dst_dir = out_dir / _sanitize(cls_dir.name)
        dst_dir.mkdir(exist_ok=True)
        for f in cls_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                _link_or_copy(f, dst_dir / f.name, link)
                n += 1
    print(f"[downloads] {out_dir}: {n} images from {src_root}")


def _already_done(out_dir: Path) -> bool:
    return out_dir.exists() and any(p.is_file() for p in out_dir.rglob("*"))


def _looks_like_class_root(d: Path, min_classes=2, min_images=1) -> bool:
    """True if d has >= min_classes subdirs that each contain image files."""
    if not d.is_dir():
        return False
    good = 0
    for sub in d.iterdir():
        if sub.is_dir() and any(f.suffix.lower() in IMG_EXTS for f in sub.iterdir() if f.is_file()):
            good += 1
            if good >= min_classes:
                return True
    return good >= min_classes


def _find_class_root(root: Path):
    """Find the directory that best looks like a `<class>/*.jpg` root."""
    if not root.exists():
        return None
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    best, best_n = None, -1
    for d in candidates:
        if _looks_like_class_root(d):
            n = sum(1 for s in d.iterdir() if s.is_dir())
            if n > best_n:
                best, best_n = d, n
    return best


def _find_split_roots(root: Path):
    """Find (train_dir, test_dir) if a train/ + test/ (or val/) layout exists."""
    if not root.exists():
        return None
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        names = {c.name.lower(): c for c in d.iterdir() if c.is_dir()}
        tr = names.get("train")
        te = names.get("test") or names.get("val") or names.get("valid")
        if tr and _looks_like_class_root(tr):
            return tr, (te if (te and _looks_like_class_root(te)) else None)
    return None


def _kaggle_or_die():
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The `kaggle` CLI is required for this dataset.\n"
            "  pip install kaggle\n"
            "  create a token at https://www.kaggle.com/settings ('Create New Token')\n"
            "  save it to ~/.kaggle/kaggle.json  (chmod 600)")


def _kaggle_dataset_download(slug: str, raw_root: Path):
    _kaggle_or_die()
    raw_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_root), "--unzip"], check=True)


def _kaggle_competition_download(comp: str, raw_root: Path):
    _kaggle_or_die()
    raw_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", comp, "-p", str(raw_root)], check=True)
    for z in raw_root.glob("*.zip"):
        subprocess.run(["unzip", "-n", "-q", str(z), "-d", str(raw_root)], check=True)


def _export_foldered(raw_root: Path, out_root: Path, link: bool, help_text: str):
    """Generic exporter for a Kaggle/GitHub dataset that is already foldered."""
    split = _find_split_roots(raw_root)
    if split is not None:
        train_dir, test_dir = split
        _copy_class_tree(train_dir, out_root / "train", link)
        if test_dir is not None:
            _copy_class_tree(test_dir, out_root / "test", link)
        return
    class_root = _find_class_root(raw_root)
    if class_root is not None:
        _copy_class_tree(class_root, out_root / "train", link)
        return
    raise RuntimeError(f"Couldn't find a class-folder layout under {raw_root}.\n{help_text}")


# --------------------------------------------------------------------------- #
# PlantVillage  (Kaggle, foldered)
# --------------------------------------------------------------------------- #
PLANTVILLAGE_HELP = (
    "PlantVillage: pass a Kaggle dataset slug with --kaggle-dataset, e.g.\n"
    "  python downloads.py --dataset plantvillage --kaggle-dataset abdallahalidev/plantvillage-dataset\n"
    "The exporter picks the folder with the most class sub-directories (use the\n"
    "'color' variant if the dataset ships color/grayscale/segmented).")


def prepare_plantvillage(raw_root: Path, out_root: Path, link: bool, args):
    if not _already_done(raw_root):
        slug = args.kaggle_dataset or "abdallahalidev/plantvillage-dataset"
        print(f"[downloads] plantvillage via Kaggle dataset: {slug}")
        _kaggle_dataset_download(slug, raw_root)
    # prefer a 'color' root if present (PlantVillage ships color/grayscale/segmented)
    color = next((p for p in raw_root.rglob("*") if p.is_dir() and p.name.lower() == "color"
                  and _looks_like_class_root(p)), None)
    if color is not None:
        _copy_class_tree(color, out_root / "train", link)
        return
    _export_foldered(raw_root, out_root, link, PLANTVILLAGE_HELP)


# --------------------------------------------------------------------------- #
# PlantDoc  (GitHub zip, real train/test)
# --------------------------------------------------------------------------- #
PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"


def prepare_plantdoc(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets.utils import download_and_extract_archive
    if not _already_done(raw_root):
        raw_root.mkdir(parents=True, exist_ok=True)
        download_and_extract_archive(PLANTDOC_URL, download_root=str(raw_root),
                                      filename="plantdoc-master.zip")
    _export_foldered(raw_root, out_root, link,
                     "PlantDoc: expected train/ and test/ folders inside the extracted repo.")


# --------------------------------------------------------------------------- #
# Cassava Leaf Disease  (Kaggle competition, csv labels)
# --------------------------------------------------------------------------- #
CASSAVA_HELP = (
    "Cassava Leaf Disease is a Kaggle *competition* dataset. Accept the rules once at\n"
    "  https://www.kaggle.com/c/cassava-leaf-disease-classification/rules\n"
    "then re-run. Override the competition slug with --kaggle-dataset if needed.")


def prepare_cassava(raw_root: Path, out_root: Path, link: bool, args):
    comp = args.kaggle_dataset or "cassava-leaf-disease-classification"
    if not _already_done(raw_root):
        print(f"[downloads] cassava via Kaggle competition: {comp}")
        try:
            _kaggle_competition_download(comp, raw_root)
        except Exception as e:
            raise RuntimeError(CASSAVA_HELP) from e

    # If the download is already foldered somewhere, just use it.
    split = _find_split_roots(raw_root) or _find_class_root(raw_root)
    csv_path = next(raw_root.rglob("train.csv"), None)
    img_dir = next((p for p in raw_root.rglob("train_images") if p.is_dir()), None)
    if csv_path is None or img_dir is None:
        if split is not None:
            _export_foldered(raw_root, out_root, link, CASSAVA_HELP)
            return
        raise RuntimeError(f"Couldn't find train.csv + train_images/ under {raw_root}.\n{CASSAVA_HELP}")

    # map label id -> disease name
    label_map = {}
    mapping_json = next(raw_root.rglob("label_num_to_disease_map.json"), None)
    if mapping_json is not None:
        with open(mapping_json) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}

    image_files, labels, names = [], [], {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            lid = int(row["label"])
            image_files.append(img_dir / row["image_id"])
            labels.append(lid)
            names[lid] = _sanitize(label_map.get(lid, f"class_{lid}"))
    class_names = [names[i] for i in range(max(names) + 1)]
    _export(image_files, labels, class_names, out_root / "train", link)


# --------------------------------------------------------------------------- #
# Apple leaf disease  (Kaggle, foldered)
# --------------------------------------------------------------------------- #
APPLELEAF_HELP = (
    "Apple leaf disease: pass a Kaggle dataset slug with --kaggle-dataset, e.g.\n"
    "  python downloads.py --dataset appleleaf --kaggle-dataset nirmalsankalana/apple-leaf-disease-dataset\n"
    "Any dataset laid out as <class>/*.jpg (optionally under train/ + test/) works.")


def prepare_appleleaf(raw_root: Path, out_root: Path, link: bool, args):
    if not _already_done(raw_root):
        if not args.kaggle_dataset:
            raise RuntimeError(APPLELEAF_HELP)
        print(f"[downloads] appleleaf via Kaggle dataset: {args.kaggle_dataset}")
        _kaggle_dataset_download(args.kaggle_dataset, raw_root)
    _export_foldered(raw_root, out_root, link, APPLELEAF_HELP)


# --------------------------------------------------------------------------- #
# Caltech-UCSD Birds-200-2011
# --------------------------------------------------------------------------- #
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB_MANUAL_HELP = (
    "Automatic CUB-200-2011 download failed (the Caltech Data mirror may have moved). "
    "Download CUB_200_2011.tgz yourself from "
    "https://data.caltech.edu/records/65de6-vp158 and extract it so that "
    "{base}/images exists, then re-run this script.")


def prepare_cub200(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets.utils import download_and_extract_archive
    base = raw_root / "CUB_200_2011"
    if not (base / "images").exists():
        raw_root.mkdir(parents=True, exist_ok=True)
        try:
            download_and_extract_archive(CUB_URL, download_root=str(raw_root), filename="CUB_200_2011.tgz")
        except Exception as e:
            raise RuntimeError(CUB_MANUAL_HELP.format(base=base)) from e
    if not (base / "images").exists():
        raise RuntimeError(CUB_MANUAL_HELP.format(base=base))

    images = {}
    with open(base / "images.txt") as f:
        for line in f:
            iid, rel = line.strip().split(" ", 1)
            images[iid] = rel
    is_train = {}
    with open(base / "train_test_split.txt") as f:
        for line in f:
            iid, flag = line.strip().split(" ", 1)
            is_train[iid] = flag == "1"

    for iid, rel in images.items():
        cls, name = rel.split("/", 1)
        out_split = "train" if is_train[iid] else "test"
        dst_dir = out_root / out_split / _sanitize(cls)
        dst_dir.mkdir(parents=True, exist_ok=True)
        _link_or_copy(base / "images" / rel, dst_dir / name, link)
    for split in ("train", "test"):
        d = out_root / split
        n = sum(1 for p in d.rglob("*") if p.is_file())
        print(f"[downloads] {d}: {n} images")


# --------------------------------------------------------------------------- #
# FGVC-Aircraft
# --------------------------------------------------------------------------- #
def prepare_aircraft(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets import FGVCAircraft
    for split, out_split in [("trainval", "train"), ("test", "test")]:
        ds = FGVCAircraft(str(raw_root), split=split, annotation_level="variant", download=True)
        _export(ds._image_files, ds._labels, ds.classes, out_root / out_split, link)


# --------------------------------------------------------------------------- #
# Oxford Flowers-102
# --------------------------------------------------------------------------- #
def prepare_flowers102(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets import Flowers102
    class_names = [f"{i:03d}" for i in range(1, 103)]
    train_ds = Flowers102(str(raw_root), split="train", download=True)
    val_ds = Flowers102(str(raw_root), split="val", download=True)
    test_ds = Flowers102(str(raw_root), split="test", download=True)
    _export(list(train_ds._image_files) + list(val_ds._image_files),
            list(train_ds._labels) + list(val_ds._labels),
            class_names, out_root / "train", link)
    _export(test_ds._image_files, test_ds._labels, class_names, out_root / "test", link)


# --------------------------------------------------------------------------- #
# Food-101
# --------------------------------------------------------------------------- #
def prepare_food101(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets import Food101
    for split, out_split in [("train", "train"), ("test", "test")]:
        ds = Food101(str(raw_root), split=split, download=True)
        _export(ds._image_files, ds._labels, ds.classes, out_root / out_split, link)


# --------------------------------------------------------------------------- #
# CIFAR-100  (small from-scratch experiment; 32x32 PNGs, 50k train / 10k test)
# --------------------------------------------------------------------------- #
def prepare_cifar100(raw_root: Path, out_root: Path, link: bool, args):
    from torchvision.datasets import CIFAR100
    for split, out_split in [(True, "train"), (False, "test")]:
        ds = CIFAR100(str(raw_root), train=split, download=True)
        base = out_root / out_split
        for c in ds.classes:
            (base / c).mkdir(parents=True, exist_ok=True)
        for i, (img, y) in enumerate(ds):
            img.save(base / ds.classes[y] / f"{i:05d}.png")
        print(f"[downloads] {base}: {len(ds)} images across {len(ds.classes)} classes")


DATASETS = {
    "plantvillage": prepare_plantvillage,
    "plantdoc": prepare_plantdoc,
    "cassava": prepare_cassava,
    "appleleaf": prepare_appleleaf,
    "cub200": prepare_cub200,
    "aircraft": prepare_aircraft,
    "flowers102": prepare_flowers102,
    "food101": prepare_food101,
    "cifar100": prepare_cifar100,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=list(DATASETS) + ["all"], default=["all"])
    p.add_argument("--root", type=str, default="datasets", help="output root: <root>/<name>/{train,test}")
    p.add_argument("--raw-root", type=str, default=None, help="raw download cache (default: <root>/_raw)")
    p.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    p.add_argument("--clean-raw", action="store_true", help="delete a dataset's raw cache after exporting it")
    p.add_argument("--kaggle-dataset", type=str, default=None,
                   help="Kaggle dataset slug (or competition slug for cassava) to use for the selected dataset")
    args = p.parse_args()

    names = list(DATASETS) if "all" in args.dataset else args.dataset
    root = Path(args.root)
    raw_root = Path(args.raw_root) if args.raw_root else root / "_raw"
    link = not args.copy

    results = {}
    for name in names:
        print(f"\n=== {name} ===")
        out_dir = root / name
        if _already_done(out_dir):
            print(f"[downloads] {out_dir} already has files, skipping (delete it to re-export)")
            results[name] = "already done"
            continue
        try:
            DATASETS[name](raw_root / name, out_dir, link, args)
        except Exception as e:
            print(f"[downloads] {name} FAILED: {e}")
            results[name] = f"failed: {e}"
            continue
        results[name] = "ok"
        if args.clean_raw:
            shutil.rmtree(raw_root / name, ignore_errors=True)

    print("\n=== summary ===")
    for name, status in results.items():
        print(f"  {name:14s} {status}")


if __name__ == "__main__":
    main()
