"""
Metrics for SDNET2018 crack classification (positive class = Cracked, ~15% prevalence).

Accuracy is nearly meaningless at this imbalance, so the headline numbers are the
threshold-free ranking metrics (ROC-AUC, PR-AUC/AP) plus the crack-class operating-point
metrics a structural inspector cares about: recall (sensitivity, i.e. missed cracks =
FN), precision, F1/F2, specificity, MCC, IoU, and calibration (Brier, ECE).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, cohen_kappa_score,
                             confusion_matrix, log_loss, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ece_score(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def compute_all(y, p, thr=0.5):
    """y: {0,1} (1 = Cracked); p: P(Cracked). Returns a flat dict of floats."""
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    d = lambda a, b: float(a / b) if b else 0.0
    prec, rec, spec = d(tp, tp + fp), d(tp, tp + fn), d(tn, tn + fp)
    fb = lambda beta: d((1 + beta ** 2) * prec * rec, beta ** 2 * prec + rec)
    two = len(np.unique(y)) > 1
    return dict(
        threshold=float(thr), n=int(len(y)), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        acc=d(tp + tn, len(y)), bal_acc=0.5 * (rec + spec),
        precision=prec, recall=rec, specificity=spec, npv=d(tn, tn + fn),
        f1=fb(1), f2=fb(2), f1_macro=0.5 * (fb(1) + _f1_neg(tn, fp, fn)),
        mcc=float(matthews_corrcoef(y, pred)) if two else 0.0,
        kappa=float(cohen_kappa_score(y, pred)) if two else 0.0,
        iou_crack=d(tp, tp + fp + fn),
        fpr=d(fp, fp + tn), fnr=d(fn, fn + tp),
        auc=float(roc_auc_score(y, p)) if two else float("nan"),
        ap=float(average_precision_score(y, p)) if two else float("nan"),
        brier=float(brier_score_loss(y, p)), ece=ece_score(y, p),
        nll=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
    )


def _f1_neg(tn, fp, fn):
    p = tn / (tn + fn) if (tn + fn) else 0.0
    r = tn / (tn + fp) if (tn + fp) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def best_f1_threshold(y, p, beta=1.0):
    """Threshold maximising F-beta on (y, p). Tune on VAL, then apply to TEST."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-12)
    i = int(np.argmax(f))
    return float(thr[i]), float(f[i])


def threshold_at_recall(y, p, target=0.95):
    """Highest threshold that still reaches `target` recall (fewest false alarms)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def bootstrap_ci(y, p, thr, keys=("auc", "ap", "f1", "recall", "precision", "mcc"),
                 n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    acc = {k: [] for k in keys}
    for _ in range(n_boot):                                  # stratified resample
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = compute_all(y[idx], p[idx], thr)
        for k in keys:
            acc[k].append(m[k])
    return {k: (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
            for k, v in acc.items()}


def early_exit_table(y, p_exit1, p_final, cost1, cost_final, taus, thr=0.5):
    """Confidence-gated exit: stop at exit1 if max(p, 1-p) >= tau, else use the final exit."""
    conf = np.maximum(p_exit1, 1 - p_exit1)
    rows = []
    for tau in taus:
        stop = np.zeros_like(conf, bool) if tau is None else conf >= tau
        pm = np.where(stop, p_exit1, p_final)
        m = compute_all(y, pm, thr)
        frac = float(stop.mean())
        rows.append(dict(tau="final_only" if tau is None else tau, exited_early=frac,
                         avg_block_apps=frac * cost1 + (1 - frac) * cost_final,
                         acc=m["acc"], auc=m["auc"], f1=m["f1"], recall=m["recall"], mcc=m["mcc"]))
    return rows


# ----------------------------------------------------------------------------- multi-class
def compute_multiclass(y, probs, auc_max_classes=60):
    """y: (N,) int; probs: (N, C) softmax. Top-1/top-5, macro/weighted F1, balanced acc, MCC, NLL, ECE."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    y, probs = np.asarray(y).astype(int), np.asarray(probs, dtype=float)
    C = probs.shape[1]
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, 1)[:, :min(5, C)]
    out = dict(n=int(len(y)), acc=float((pred == y).mean()),
               top5=float((top5 == y[:, None]).any(1).mean()),
               bal_acc=float(recall_score(y, pred, average="macro", zero_division=0)),
               f1_macro=float(f1_score(y, pred, average="macro", zero_division=0)),
               f1_weighted=float(f1_score(y, pred, average="weighted", zero_division=0)),
               precision_macro=float(precision_score(y, pred, average="macro", zero_division=0)),
               recall_macro=float(recall_score(y, pred, average="macro", zero_division=0)),
               mcc=float(matthews_corrcoef(y, pred)),
               nll=float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1)).mean()),
               ece=ece_score((pred == y).astype(float), probs.max(1)), auc_ovr=float("nan"))
    if C <= auc_max_classes and len(np.unique(y)) == C:
        out["auc_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", labels=list(range(C))))
    return out


def early_exit_table_mc(y, p_exit1, p_final, cost1, cost_final, taus):
    conf = p_exit1.max(1)
    rows = []
    for tau in taus:
        stop = np.zeros_like(conf, bool) if tau is None else conf >= tau
        pm = np.where(stop[:, None], p_exit1, p_final)
        frac = float(stop.mean())
        rows.append(dict(tau="final_only" if tau is None else tau, exited_early=frac,
                         avg_block_apps=frac * cost1 + (1 - frac) * cost_final,
                         acc=float((pm.argmax(1) == y).mean())))
    return rows
