"""
Grad-CAM explanation figures for a trained SDNET2018 LoopedCrackViT.

    python gradcam.py --run runs/gipa_full_0-6x2-0_conv_bs16_random_smote_s42 --out gradcam

Panels per image:
    image | Grad-CAM conv stem (28x28) | Grad-CAM pass 1 (exit1) | Grad-CAM pass 2 (final)

Grad-CAM on a ViT: the head reads only [CLS], so the patch tokens at the OUTPUT of the last
block get no gradient. We therefore take the INPUT of the last core block (its attention is
what moves patch evidence into [CLS]), once per pass, and back-propagate the logit of the
exit belonging to that pass.

Faithfulness (no pixel masks in SDNET, so localisation IoU is impossible): deletion / insertion
curves (Petsiuk et al., RISE 2018) on correctly classified cracked test images, Grad-CAM vs a
random map. Lower deletion AUC and higher insertion AUC = the map points at what the model uses.

The decision threshold is tuned on VAL (best F1), exactly as in train.py.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import metrics as M
from data import NORMS, CrackDataset, build_transforms, load_split
from model import CrackViTConfig, LoopedCrackViT

CASES = {"TP": "true positive", "FN": "false negative (missed crack)",
         "FP": "false positive", "TN": "true negative"}
STEM_LAYER = 17          # ReLU after the 3rd stride-2 conv block -> 28x28 feature map


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/gipa_full_0-6x2-0_conv_bs16_random_smote_s42")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", default="gradcam")
    ap.add_argument("--per-case", type=int, default=2, help="images per surface x case")
    ap.add_argument("--n-faith", type=int, default=200, help="images for deletion/insertion (0 = skip)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


# ---------------------------------------------------------------- model / data
def load_model(run, ckpt, device):
    cfg = json.load(open(os.path.join(run, "config.json")))
    ck = torch.load(os.path.join(run, ckpt), map_location="cpu", weights_only=False)
    model = LoopedCrackViT(CrackViTConfig(**ck.get("model_cfg", cfg["model"])))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg["args"], ck.get("epoch")


@torch.no_grad()
def predict(model, df, tf, device, bs):
    dl = DataLoader(CrackDataset(df, tf), batch_size=bs, shuffle=False, num_workers=0)
    p1, pf = [], []
    for x, _ in tqdm(dl, desc="predict", leave=False):
        outs = model(x.to(device))
        p1.append(torch.sigmoid(outs[0].float()).cpu()); pf.append(torch.sigmoid(outs[-1].float()).cpu())
    return torch.cat(p1).numpy(), torch.cat(pf).numpy()


# ---------------------------------------------------------------- explanations
class Explainer:
    def __init__(self, model):
        self.m = model
        self.tokens, self.stem = [], None
        last = model.core[-1]
        last.register_forward_pre_hook(self._tok_hook)
        model.tokeniser.stem[STEM_LAYER].register_forward_hook(self._stem_hook)

    def _tok_hook(self, mod, args):
        x = args[0]
        if x.requires_grad:
            x.retain_grad()
        self.tokens.append(x)

    def _stem_hook(self, mod, inp, out):
        if not out.requires_grad:               # plain no-grad forward (deletion/insertion)
            return None
        out = out.clone()                       # ReLU is in-place; keep our own copy in the graph
        out.retain_grad()
        self.stem = out
        return out

    @staticmethod
    def _cam(act, grad, chw):
        """act/grad (B, C, H, W) -> (B, H, W) in [0, 1]."""
        w = grad.mean((2, 3), keepdim=True) if chw else grad
        cam = F.relu((w * act).sum(1))
        mx = cam.flatten(1).max(1)[0].clamp_min(1e-12)[:, None, None]
        return cam / mx

    def __call__(self, x):
        """x (B,3,H,W) normalised -> dict of maps (B, h, w) numpy + probs."""
        self.tokens = []
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            outs = self.m(x)
            res = {}
            g = int(len(self.tokens[0][0, 1:]) ** 0.5)
            for p, (name, logit) in enumerate(zip(("pass1", "pass2"), (outs[0], outs[-1]))):
                for t in self.tokens:
                    t.grad = None
                self.stem.grad = None
                logit.float().sum().backward(retain_graph=True)
                t = self.tokens[p]
                act = t[:, 1:].detach().float().transpose(1, 2).reshape(len(x), -1, g, g)
                grd = t.grad[:, 1:].float().transpose(1, 2).reshape(len(x), -1, g, g)
                res[name] = self._cam(act, grd, True)
                if name == "pass2":                                      # stem CAM w.r.t. the final logit
                    res["stem"] = self._cam(self.stem.detach().float(), self.stem.grad.float(), True)
        res = {k: v.detach().cpu().numpy() for k, v in res.items()}
        res["p1"] = torch.sigmoid(outs[0].detach().float()).cpu().numpy()
        res["pf"] = torch.sigmoid(outs[-1].detach().float()).cpu().numpy()
        return res


def upsample(m, size):
    t = torch.from_numpy(m)[None, None].float()
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


# ---------------------------------------------------------------- drawing
def show_img(ax, img):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])


def overlay(ax, img, cam):
    show_img(ax, img)
    ax.imshow(upsample(cam, img.shape[0]), cmap="jet", alpha=0.45, vmin=0, vmax=1)


COLS = ["Input", "Grad-CAM\nconv stem", "Grad-CAM\npass 1 (exit 1)", "Grad-CAM\npass 2 (final)"]


def draw_rows(rows, path, title=None):
    """rows: list of dicts with img, stem, pass1, pass2, p1, pf, label text."""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 2.1, n * 2.25 + (0.5 if title else 0.2)), squeeze=False)
    for r, row in enumerate(rows):
        a = axes[r]
        show_img(a[0], row["img"])
        overlay(a[1], row["img"], row["stem"])
        overlay(a[2], row["img"], row["pass1"])
        overlay(a[3], row["img"], row["pass2"])
        a[0].set_ylabel(row["label"], fontsize=8)
        a[2].set_xlabel(f"p(crack) = {row['p1']:.2f}", fontsize=8)
        a[3].set_xlabel(f"p(crack) = {row['pf']:.2f}", fontsize=8)
        if r == 0:
            for c, t in enumerate(COLS):
                a[c].set_title(t, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def denorm(x, norm):
    m, s = (torch.tensor(v)[:, None, None] for v in NORMS[norm])
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------- faithfulness
@torch.no_grad()
def deletion_insertion(model, x, maps, steps, bs):
    """x (N,3,H,W) normalised; maps (N,H,W) saliency at input size. Returns (del_curve, ins_curve) (steps+1,)."""
    N, _, H, W = x.shape
    base = F.avg_pool2d(F.pad(x, (5, 5, 5, 5), mode="reflect"), 11, 1)           # blurred image
    order = torch.from_numpy(maps).reshape(N, -1).argsort(1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(H * W).expand(N, -1))
    rank = rank.reshape(N, 1, H, W).to(x.device)
    dele, ins = [], []
    for k in range(steps + 1):
        top = rank < int(round(k / steps * H * W))
        for out, a, b in ((dele, base, x), (ins, x, base)):           # top pixels taken from a, rest from b
            img = torch.where(top, a, b)
            p = torch.cat([torch.sigmoid(model(img[i:i + bs])[-1].float()) for i in range(0, N, bs)])
            out.append(p.mean().item())
    return np.array(dele), np.array(ins)


# ---------------------------------------------------------------- main
def main():
    args = get_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    rng = np.random.default_rng(args.seed)
    model, targs, epoch = load_model(args.run, args.ckpt, device)
    norm = "imagenet" if targs.get("norm") == "imagenet" or (targs.get("norm") == "auto" and targs.get("backbone")) else "half"
    tf = build_transforms(targs["image_size"], False, norm)
    df = load_split(targs["data_root"], targs["split_mode"], targs["seed"],
                    os.path.join(os.path.dirname(os.path.normpath(args.run)), "_splits"))
    va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("val", "test"))
    os.makedirs(args.out, exist_ok=True)

    # threshold on VAL, predictions on TEST
    cache = os.path.join(args.out, f"_preds_{os.path.basename(os.path.normpath(args.run))}_{args.ckpt}.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        pv, te["p_exit1"], te["p_final"] = c["pv"], c["p1"], c["pf"]
    else:
        _, pv = predict(model, va, tf, device, args.batch_size)
        te["p_exit1"], te["p_final"] = predict(model, te, tf, device, args.batch_size)
        np.savez(cache, pv=pv, p1=te["p_exit1"].to_numpy(), pf=te["p_final"].to_numpy())
    thr, _ = M.best_f1_threshold(va["label"].to_numpy(), pv, 1.0)
    y, pr = te["label"].to_numpy(), (te["p_final"] >= thr).astype(int)
    te["pred"] = pr
    te["case"] = np.select([(y == 1) & (pr == 1), (y == 1) & (pr == 0), (y == 0) & (pr == 1)], ["TP", "FN", "FP"], "TN")
    te.to_csv(os.path.join(args.out, "test_predictions.csv"), index=False)
    summ = {"checkpoint": os.path.join(args.run, args.ckpt), "epoch": epoch, "threshold_val_bestF1": thr,
            "test_auc_final": roc_auc_score(y, te["p_final"]), "test_auc_exit1": roc_auc_score(y, te["p_exit1"]),
            "test_counts": te["case"].value_counts().to_dict()}
    print(json.dumps(summ, indent=2))

    # ---- choose examples: TP/TN = confident, FN/FP = most confidently wrong; sampled from the top 30
    expl = Explainer(model)
    picks = []
    for s in sorted(te["surface"].unique()):
        for c in CASES:
            d = te[(te["surface"] == s) & (te["case"] == c)]
            if d.empty:
                continue
            d = d.sort_values("p_final", ascending=c in ("FN", "TN")).head(30)
            picks.append(d.iloc[np.sort(rng.choice(len(d), min(args.per_case, len(d)), replace=False))])
    picks = pd.concat(picks)
    ds = CrackDataset(picks, tf)
    rows = []
    for i in tqdm(range(len(picks)), desc="grad-cam"):
        x = ds[i][0][None].to(device)
        e = expl(x)
        r = picks.iloc[i]
        row = {k: e[k][0] for k in ("stem", "pass1", "pass2", "p1", "pf")}
        row.update(img=denorm(ds[i][0], norm), surface=r["surface"], case=r["case"], path=r["path"],
                   label=f"{r['surface']} | {r['case']}\n{'Cracked' if r['label'] else 'Non-cracked'}")
        rows.append(row)

    by = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
    surfaces = sorted({r["surface"] for r in rows})
    # main paper figure: 2 cracked (TP) per surface
    draw_rows([r for s in surfaces for r in by(surface=s, case="TP")[:2]],
              os.path.join(args.out, "fig_gradcam_cracked"),
              "Correctly detected cracks (TP): Grad-CAM evidence concentrates on the crack")
    draw_rows([r for s in surfaces for r in by(surface=s, case="TN")[:1]] +
              [r for s in surfaces for r in by(surface=s, case="TP")[:1]],
              os.path.join(args.out, "fig_gradcam_cracked_vs_intact"),
              "Intact (TN, top) vs cracked (TP, bottom)")
    draw_rows([r for s in surfaces for c in ("FN", "FP") for r in by(surface=s, case=c)[:1]],
              os.path.join(args.out, "fig_gradcam_failures"),
              "Failure cases: missed cracks (FN) and false alarms (FP)")
    if args.n_faith <= 0:
        return

    # ---- faithfulness on TP test images
    tp = te[te["case"] == "TP"]
    tp = tp.iloc[np.sort(rng.choice(len(tp), min(args.n_faith, len(tp)), replace=False))]
    ds = CrackDataset(tp, tf)
    xs, cams = [], {k: [] for k in ("stem", "pass1", "pass2", "random")}
    S = targs["image_size"]
    for i in tqdm(range(0, len(tp), 16), desc="faithfulness maps"):
        x = torch.stack([ds[j][0] for j in range(i, min(i + 16, len(tp)))]).to(device)
        e = expl(x)
        for k in ("stem", "pass1", "pass2"):
            cams[k] += [upsample(m, S) for m in e[k]]
        cams["random"] += [upsample(rng.random(e["pass2"][0].shape), S) for _ in range(len(x))]
        xs.append(x.detach())
    x = torch.cat(xs)
    names = {"pass2": "Grad-CAM pass 2 (final)", "pass1": "Grad-CAM pass 1", "stem": "Grad-CAM conv stem",
             "random": "random map"}
    res, curves = [], {}
    for k in tqdm(names, desc="deletion/insertion"):
        d, i = deletion_insertion(model, x, np.stack(cams[k]), args.steps, args.batch_size)
        curves[k] = (d, i)
        res.append({"map": names[k], "deletion_auc(lower=better)": np.trapezoid(d, dx=1 / args.steps),
                    "insertion_auc(higher=better)": np.trapezoid(i, dx=1 / args.steps), "n_images": len(tp)})
    fdf = pd.DataFrame(res)
    fdf.to_csv(os.path.join(args.out, "faithfulness.csv"), index=False)
    pd.DataFrame({"fraction": np.linspace(0, 1, args.steps + 1),
                  **{f"{k}_{n}": c[j] for k, c in curves.items() for j, n in enumerate(("deletion", "insertion"))}}
                 ).to_csv(os.path.join(args.out, "faithfulness_curves.csv"), index=False)
    print(fdf.to_string(index=False))
    fr = np.linspace(0, 1, args.steps + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, (d, i) in curves.items():
        ls = "--" if k == "random" else "-"
        ax[0].plot(fr, d, ls, label=names[k]); ax[1].plot(fr, i, ls, label=names[k])
    for a, t in zip(ax, ("Deletion (lower AUC = more faithful)", "Insertion (higher AUC = more faithful)")):
        a.set_title(t, fontsize=10); a.set_xlabel("fraction of pixels removed / inserted"); a.set_ylabel("mean p(crack)")
        a.grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig_faithfulness.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    summ["faithfulness"] = res
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
