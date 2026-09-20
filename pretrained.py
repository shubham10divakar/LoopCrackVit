"""
Initialise LoopedCrackViT from an ImageNet-pretrained timm ViT (DeiT-S / ViT-S / ViT-B ...).

Why: ViTs trained from scratch on small datasets (CUB ~6k, Aircraft ~6.7k, SDNET cracks)
land far below published numbers, and published numbers are fine-tuned from ImageNet
weights. Starting from pretrained weights makes results comparable to the literature.

The looped model has fewer UNIQUE blocks than the 12-layer source, so layers are collapsed:

  strategy "avg"   (Relaxed-Recursive-Transformers style)  [default]
      Each shared core block starts as the AVERAGE of the source layers it will stand in
      for (one per pass). STACK 6x2: block i <- mean(layer i, layer i+6). Prelude / coda
      blocks copy the first / last source layers.
  strategy "pick"
      Take evenly spaced source layers: 6 unique blocks <- layers 0,2,4,6,8,10.

General rule for "avg": source depth D, P prelude, C core, K passes, Q coda.
    prelude j <- layer j                      coda j <- layer D-Q+j
    core i    <- mean over p<K of layer  P + floor((p*C + i) * L/(C*K)),  L = D-P-Q
Untied baselines fall out of the same rule: (C=12,K=1) is the full DeiT-S (identity copy);
(C=6,K=1) is the parameter-matched shallow ViT (layers 0,2,...,10).

New parameters (GIPA gates / structure-tensor head / predictor W, pass embedding, task
head) are NOT in the checkpoint. The gates start at ~0.01 so the model begins as
(approximately) the pretrained content-attention ViT and learns to use geometry.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F

BLOCK_KEYS = ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
              "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
              "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")


def interpolate_pos_embed(pos, num_patches_new, num_prefix=1):
    """Bicubic resize of the patch position grid (e.g. 224 -> 384 px)."""
    old = pos.shape[1] - num_prefix
    if old == num_patches_new:
        return pos
    dim, prefix, grid = pos.shape[-1], pos[:, :num_prefix], pos[:, num_prefix:]
    go, gn = int(math.sqrt(old)), int(math.sqrt(num_patches_new))
    grid = grid.reshape(1, go, go, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(gn, gn), mode="bicubic", align_corners=False)
    return torch.cat([prefix, grid.permute(0, 2, 3, 1).reshape(1, gn * gn, dim)], 1)


def source_plan(cfg, depth):
    """For every UNIQUE block (prelude, core, coda) -> list of source layer indices to average."""
    P, C, Q, K = cfg.n_prelude, cfg.n_core, cfg.n_coda, cfg.n_passes
    L = depth - P - Q
    if L < 1:
        raise ValueError(f"source depth {depth} too small for prelude={P} coda={Q}")
    plan = [[j] for j in range(P)]
    plan += [sorted({P + int((p * C + i) * L / (C * K)) for p in range(K)}) for i in range(C)]
    plan += [[depth - Q + j] for j in range(Q)]
    return plan


def pick_plan(cfg, depth):
    n = cfg.n_prelude + cfg.n_core + cfg.n_coda
    return [[int(u * depth / n)] for u in range(n)]


def load_pretrained(model, backbone="deit_small_patch16_224", strategy="avg", pretrained=True):
    """Copy timm weights into `model` (a LoopedCrackViT). Returns a report dict."""
    import timm

    cfg = model.cfg
    if cfg.stem != "patch":
        raise ValueError("pretrained init needs stem='patch' (a conv stem has no ImageNet equivalent)")
    src = timm.create_model(backbone, pretrained=pretrained).state_dict()
    depth = 1 + max(int(k.split(".")[1]) for k in src if k.startswith("blocks."))
    src_dim = src["cls_token"].shape[-1]
    if src_dim != cfg.dim:
        raise ValueError(f"{backbone} has dim {src_dim} but the model has dim {cfg.dim}; "
                         f"set dim/num_heads/mlp_ratio to match (DeiT-S: 384 / 6 / 4.0)")
    if src["blocks.0.mlp.fc1.weight"].shape[0] != int(cfg.dim * cfg.mlp_ratio):
        raise ValueError("mlp_ratio does not match the backbone")
    if not cfg.qkv_bias:
        raise ValueError("set qkv_bias: true (timm ViTs have a qkv bias)")

    if strategy not in ("avg", "pick"):
        raise ValueError("strategy must be 'avg' or 'pick'")
    plan = (source_plan if strategy == "avg" else pick_plan)(cfg, depth)
    unique = list(model.prelude) + list(model.core) + list(model.coda)
    names = ([f"prelude.{i}" for i in range(cfg.n_prelude)] + [f"core.{i}" for i in range(cfg.n_core)]
             + [f"coda.{i}" for i in range(cfg.n_coda)])
    rep = {"backbone": backbone, "strategy": strategy, "src_depth": depth, "mapping": {}, "new": []}

    with torch.no_grad():
        model.tokeniser.proj.weight.copy_(src["patch_embed.proj.weight"])
        model.tokeniser.proj.bias.copy_(src["patch_embed.proj.bias"])
        model.cls_pos.cls.copy_(src["cls_token"])
        model.cls_pos.pos.copy_(interpolate_pos_embed(src["pos_embed"], model.cls_pos.pos.shape[1] - 1))
        for blk, name, layers in zip(unique, names, plan):
            sd = blk.state_dict()
            for k in BLOCK_KEYS:
                sd[k] = torch.stack([src[f"blocks.{l}.{k}"] for l in layers]).mean(0)
            blk.load_state_dict(sd, strict=False)       # GIPA extras stay at their own init
            rep["mapping"][name] = layers
        model.head.norm.weight.copy_(src["norm.weight"])
        model.head.norm.bias.copy_(src["norm.bias"])
    rep["new"] = ["head.out (task head)", "pass_embed", "GIPA gates / tensor_head / W (if GIPA)"]
    print(f"[pretrained] {backbone} ({depth} layers) -> {len(unique)} unique blocks, strategy={strategy}")
    print("             " + "  ".join(f"{n}<-{l}" for n, l in rep["mapping"].items()))
    return rep
