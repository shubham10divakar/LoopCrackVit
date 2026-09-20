"""
Looped CrackViT + GIPA  (PyTorch port of the Kaggle TensorFlow cells).

    stem -> [CLS]+pos -> prelude -> [ core ] x n_passes -> coda -> head
                                        |                    ^
                                        +-- after every pass: coda -> head  (early exits,
                                            ONE head shared by all exits)

    STACK     : n_prelude=0, n_core=6, n_coda=0, n_passes=2   (6 blocks x 2 = 12 applications)
    SANDWICH  : n_prelude=1, n_core=4, n_coda=1, n_passes=2   (1 + 4x2 + 1  = 10 applications)

GIPA = Geometry-Informed Predictive Attention:
    logit_ij = content(q.k) + lam_G*G_ij (orientation coherence + displacement
               alignment, anisotropy-gated) + lam_I*P_ij (predictive info) + S_ij
               (anisotropic continuity prior)
q/k/v, the structure-tensor head and predictor W are shared across passes; the scalar
gates (lam_c, lam_a, lam_i, gamma, kappa) have one row per pass (per_pass_gates).

The model outputs one LOGIT per exit: (B,) for binary (positive class = Cracked),
(B, num_classes) for the multi-class benchmarks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_INIT = 0.01


@dataclass
class CrackViTConfig:
    image_size: int = 224
    stem: str = "conv"              # conv | patch
    patch_size: int = 16            # only for stem == "patch"
    dim: int = 256
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attention: str = "GIPA_FULL"    # MHSA | GIPA_S | GIPA_G | GIPA_I | GIPA_GI | GIPA_FULL
    coherence_reg: float = 0.0
    n_prelude: int = 0
    n_core: int = 6
    n_coda: int = 0
    n_passes: int = 2
    pass_embed: bool = True
    per_pass_gates: bool = True
    input_injection: bool = False
    # --- pretrained-ViT compatibility (DeiT/ViT-S: dim 384, 6 heads, mlp 4, patch stem) ---
    qkv_bias: bool = False          # timm ViTs have a qkv bias
    patch_norm: bool = True         # LayerNorm after the patch projection (timm: none)
    head_type: str = "mlp"          # mlp (notebook head) | linear (LN -> Linear, like DeiT)
    drop_path: float = 0.0          # stochastic depth, linear over the UNIQUE blocks
    num_classes: int = 1            # 1 = binary (single logit, SDNET); >1 = multi-class benchmarks

    def to_dict(self):
        return asdict(self)


ATTENTION_FLAGS = {          # (geometry, info, structure)
    "GIPA_S":    (False, False, True),
    "GIPA_G":    (True,  False, False),
    "GIPA_I":    (False, True,  False),
    "GIPA_GI":   (True,  True,  False),
    "GIPA_FULL": (True,  True,  True),
}


# ---------------------------------------------------------------- tokenisers
class ConvStem(nn.Module):
    """/16 conv tokeniser (224 -> 14x14). Stride-2 convs, not max-pooling: MaxPool keeps
    the brightest pixel and would erase a thin dark crack."""

    def __init__(self, dim):
        super().__init__()

        def blk(cin, cout, stride):
            return [nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]

        self.stem = nn.Sequential(
            *blk(3, 32, 1), *blk(32, 32, 2),
            *blk(32, 64, 1), *blk(64, 64, 2),
            *blk(64, 128, 1), *blk(128, 128, 2),
            *blk(128, dim, 2))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(self.stem(x).flatten(2).transpose(1, 2))


class PatchEmbed(nn.Module):
    def __init__(self, patch, dim, norm=True):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)
        self.norm = nn.LayerNorm(dim, eps=1e-6) if norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class CLSPos(nn.Module):
    """Prepend a learnable [CLS], add learnable position embeddings. Applied once."""

    def __init__(self, num_patches, dim, dropout):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1).to(x.dtype), x], 1)
        return self.drop(x + self.pos.to(x.dtype))


class PassEmbedding(nn.Module):
    """One zero-init learned vector per pass, added to every token at the pass start."""

    def __init__(self, n_passes, dim):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(n_passes, dim))

    def forward(self, h, p):
        return h + self.table[p].to(h.dtype)[None, None, :]


class InputInjection(nn.Module):
    """h <- h + g * x0 on passes >= 2 (per-channel zero-init gate)."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, h, x0):
        return h + self.gate.to(h.dtype)[None, None, :] * x0


# ---------------------------------------------------------------- attention
class MHSA(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, num_passes=1, qkv_bias=False, **_):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop, self.proj_drop = dropout, nn.Dropout(dropout)
        self.store, self.last_attn = False, None

    def forward(self, x, pass_idx=0, sink=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.store:      # explicit path so the weights can be inspected
            a = torch.softmax((q.float() @ k.float().transpose(-1, -2)) * self.dh ** -0.5, -1)
            self.last_attn = a
            out = a.to(v.dtype) @ v
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, C)))


class GIPA(nn.Module):
    """Geometry-Informed Predictive Attention. Expects N = 1 + grid*grid tokens."""

    def __init__(self, dim, heads, dropout=0.0, grid=14, use_geometry=True, use_info=True,
                 use_structure=True, coherence_reg=0.0, num_passes=1, qkv_bias=False):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.scale = self.dh ** -0.5
        self.use_geometry, self.use_info, self.use_structure = use_geometry, use_info, use_structure
        self.coherence_reg, self.num_passes, self.grid = coherence_reg, num_passes, grid

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop, self.proj_drop = nn.Dropout(dropout), nn.Dropout(dropout)

        if use_geometry or use_structure:
            self.tensor_head = nn.Linear(dim, 3)
            nn.init.trunc_normal_(self.tensor_head.weight, std=0.02)
            with torch.no_grad():
                self.tensor_head.bias.copy_(torch.tensor([0.5, 0.0, 0.5]))

        g = grid
        rr, cc = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        pos = np.stack([rr.ravel(), cc.ravel()], -1).astype("float32")
        delta = pos[None, :, :] - pos[:, None, :]               # p_j - p_i
        d2 = (delta ** 2).sum(-1)
        nrm = np.sqrt(d2) + 1e-8
        dx, dy = delta[..., 0] / nrm, delta[..., 1] / nrm
        disp = np.stack([dx ** 2 - dy ** 2, 2.0 * dx * dy], -1)
        disp[d2 == 0] = 0.0
        reg = lambda n, a: self.register_buffer(n, torch.from_numpy(a.astype("float32")), persistent=False)
        reg("disp", disp)
        reg("dist2n", d2 / (d2.max() + 1e-8))
        reg("adj", (d2 == 1.0))

        G, H = num_passes, heads
        mk = lambda v: nn.Parameter(torch.full((G, H), float(v)))
        if use_geometry:
            self.lam_c, self.lam_a = mk(GATE_INIT), mk(GATE_INIT)
        if use_info:
            self.W = nn.Parameter(torch.eye(self.dh).repeat(H, 1, 1))
            self.lam_i = mk(GATE_INIT)
        if use_structure:
            self.gamma, self.kappa = mk(-6.0), mk(0.0)

        self.store = False
        self.last_attn = self.last_u = self.last_alpha = None

    def _row(self, p, pass_idx):
        return p[pass_idx if self.num_passes > 1 else 0].float()[None, :, None, None]

    def _orientation(self, x32):
        """u: (B,P,2) unit double-angle (cos2t, sin2t); alpha: (B,P) in [0,1):
        ~0 isotropic (rough concrete), ~1 oriented ridge (crack)."""
        a, b, c = self.tensor_head(x32[:, 1:]).unbind(-1)
        u_raw = torch.stack([a - c, 2.0 * b], -1)
        n = torch.sqrt((u_raw ** 2).sum(-1) + 1e-12)
        return u_raw / (n[..., None] + 1e-6), n / (n + F.softplus(a + c) + 1e-6)

    def _info_bias(self, z):
        """Directional predictive information: -log residual energy of z_j ~ W z_i."""
        Wz = torch.einsum("bhnd,hde->bhne", z, self.W.float())
        zz, ww = (z * z).sum(-1), (Wz * Wz).sum(-1)
        e = (ww[..., :, None] + zz[..., None, :] - 2.0 * Wz @ z.transpose(-1, -2)).clamp_min(0.0)
        e = e / (e.mean(-1, keepdim=True) + 1e-6)
        return -torch.log(e + 0.1)

    def forward(self, x, pass_idx=0, sink=None):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = (q.float() @ k.float().transpose(-1, -2)) * self.scale
            pad = lambda t: F.pad(t, (1, 0, 1, 0))
            u = alpha = align = gate = None
            if self.use_geometry or self.use_structure:
                u, alpha = self._orientation(x.float())
                align = 0.5 * (torch.einsum("pqk,bpk->bpq", self.disp, u)
                               + torch.einsum("pqk,bqk->bpq", self.disp, u))
                gate = torch.sqrt(alpha[:, :, None] * alpha[:, None, :] + 1e-8)
            if self.use_geometry:
                coh = torch.einsum("bpk,bqk->bpq", u, u)
                G = (self._row(self.lam_c, pass_idx) * coh[:, None]
                     + self._row(self.lam_a, pass_idx) * align[:, None]) * gate[:, None]
                logits = logits + pad(G)
            if self.use_structure:
                g_h = F.softplus(self._row(self.gamma, pass_idx))
                k_h = torch.sigmoid(self._row(self.kappa, pass_idx))
                relax = 1.0 - k_h * gate[:, None] * align[:, None] ** 2
                logits = logits + pad(-g_h * self.dist2n[None, None] * relax)
            if self.use_info:
                logits = logits + self._row(self.lam_i, pass_idx) * self._info_bias(v.float())
            attn = torch.softmax(logits, -1)
            if self.store:
                self.last_attn, self.last_u, self.last_alpha = attn.detach(), u, alpha
            out = self.attn_drop(attn) @ v.float()
        out = self.proj_drop(self.proj(out.to(x.dtype).transpose(1, 2).reshape(B, N, C)))

        if self.coherence_reg > 0.0 and u is not None and sink is not None and self.training:
            c = torch.einsum("bpk,bqk->bpq", u, u)
            w = self.adj[None] * alpha[:, :, None] * alpha[:, None, :]
            sink.append(self.coherence_reg * ((w * (1 - c)).sum((1, 2)) / (w.sum((1, 2)) + 1e-6)).mean())
        return out


def make_attention(cfg: CrackViTConfig, num_passes: int, grid: int):
    if cfg.attention == "MHSA":
        return MHSA(cfg.dim, cfg.num_heads, cfg.dropout, num_passes, cfg.qkv_bias)
    geo, info, struct = ATTENTION_FLAGS[cfg.attention]
    return GIPA(cfg.dim, cfg.num_heads, cfg.dropout, grid, geo, info, struct,
                cfg.coherence_reg, num_passes, cfg.qkv_bias)


# ---------------------------------------------------------------- blocks & model
class MLP(nn.Module):
    """fc1 -> GELU -> drop -> fc2 -> drop  (key names match timm: mlp.fc1 / mlp.fc2)."""

    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.fc1, self.fc2, self.drop = nn.Linear(dim, hidden), nn.Linear(hidden, dim), nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


def drop_path(x, p, training):
    if p == 0.0 or not training:
        return x
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(1.0 - p)
    return x * mask / (1.0 - p)


class Block(nn.Module):
    def __init__(self, cfg, num_passes, grid, dp=0.0):
        super().__init__()
        self.dp = dp
        self.norm1 = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.attn = make_attention(cfg, num_passes, grid)
        self.norm2 = nn.LayerNorm(cfg.dim, eps=1e-6)
        self.mlp = MLP(cfg.dim, int(cfg.dim * cfg.mlp_ratio), cfg.dropout)

    def forward(self, x, pass_idx=0, sink=None):
        x = x + drop_path(self.attn(self.norm1(x), pass_idx, sink), self.dp, self.training)
        return x + drop_path(self.mlp(self.norm2(x)), self.dp, self.training)


class Head(nn.Module):
    """final_norm -> [CLS] -> fc -> gelu -> dropout -> 1 logit. Shared by every exit."""

    def __init__(self, dim, num_classes=1, head_type="mlp"):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc = nn.Linear(dim, dim) if head_type == "mlp" else None
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.norm(x)[:, 0]
        if self.fc is not None:
            x = self.drop(F.gelu(self.fc(x)))
        x = self.out(x)
        return x.squeeze(-1) if x.shape[-1] == 1 else x


class LoopedCrackViT(nn.Module):
    def __init__(self, cfg: CrackViTConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.stem == "conv":
            self.tokeniser, grid = ConvStem(cfg.dim), cfg.image_size // 16
        else:
            self.tokeniser, grid = PatchEmbed(cfg.patch_size, cfg.dim, cfg.patch_norm), cfg.image_size // cfg.patch_size
        self.cls_pos = CLSPos(grid * grid, cfg.dim, cfg.dropout)
        core_passes = cfg.n_passes if cfg.per_pass_gates else 1
        n_u = cfg.n_prelude + cfg.n_core + cfg.n_coda
        dpr = [cfg.drop_path * i / max(1, n_u - 1) for i in range(n_u)]
        self.prelude = nn.ModuleList(Block(cfg, 1, grid, dpr[i]) for i in range(cfg.n_prelude))
        self.core = nn.ModuleList(Block(cfg, core_passes, grid, dpr[cfg.n_prelude + i]) for i in range(cfg.n_core))
        self.coda = nn.ModuleList(Block(cfg, 1, grid, dpr[cfg.n_prelude + cfg.n_core + i]) for i in range(cfg.n_coda))
        self.pass_emb = PassEmbedding(cfg.n_passes, cfg.dim) if cfg.pass_embed else None
        self.inject = InputInjection(cfg.dim) if cfg.input_injection else None
        self.head = Head(cfg.dim, cfg.num_classes, cfg.head_type)
        self.reg_loss = None

        for m in self.modules():                      # Keras-Dense-style init
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, GIPA) and hasattr(m, "tensor_head"):
                nn.init.trunc_normal_(m.tensor_head.weight, std=0.02)
                with torch.no_grad():
                    m.tensor_head.bias.copy_(torch.tensor([0.5, 0.0, 0.5]))

    # names / costs -----------------------------------------------------------
    @property
    def exit_names(self):
        return [f"exit{p + 1}" for p in range(self.cfg.n_passes - 1)] + ["final"]

    @property
    def exit_cost(self):
        c = self.cfg
        return {n: c.n_prelude + (p + 1) * c.n_core + c.n_coda for p, n in enumerate(self.exit_names)}

    @property
    def n_unique_blocks(self):
        c = self.cfg
        return c.n_prelude + c.n_core + c.n_coda

    def set_store(self, flag: bool):
        for m in self.modules():
            if isinstance(m, (GIPA, MHSA)):
                m.store = flag

    # forward -----------------------------------------------------------------
    def forward(self, x, max_pass: int | None = None):
        """Returns a list of logits (B,), one per exit up to `max_pass` passes."""
        sink = []
        h = self.cls_pos(self.tokeniser(x))
        for b in self.prelude:
            h = b(h, 0, sink)
        x0, outs = h, []
        for p in range(max_pass or self.cfg.n_passes):
            if self.inject is not None and p > 0:
                h = self.inject(h, x0)
            if self.pass_emb is not None:
                h = self.pass_emb(h, p)
            for b in self.core:
                h = b(h, p, sink)
            z = h
            for b in self.coda:
                z = b(z, 0, sink)
            outs.append(self.head(z))
        self.reg_loss = torch.stack(sink).sum() if sink else None
        return outs

    # diagnostics -------------------------------------------------------------
    @torch.no_grad()
    def gate_report(self):
        """Learned per-(block, pass) gate magnitudes. Init is 0.01: growth => term is used."""
        rows = []
        for name, m in self.named_modules():
            if not isinstance(m, GIPA):
                continue
            for g in range(m.num_passes):
                r = {"block": name.rsplit(".attn", 1)[0], "pass": g + 1 if m.num_passes > 1 else "shared"}
                if m.use_geometry:
                    r["lam_c"] = m.lam_c[g].abs().mean().item()
                    r["lam_a"] = m.lam_a[g].abs().mean().item()
                if m.use_info:
                    r["lam_i"] = m.lam_i[g].abs().mean().item()
                if m.use_structure:
                    r["gamma_softplus"] = F.softplus(m.gamma[g]).mean().item()
                    r["kappa_sigmoid"] = torch.sigmoid(m.kappa[g]).mean().item()
                rows.append(r)
        return rows

    def param_report(self):
        total = sum(p.numel() for p in self.parameters())
        core = sum(p.numel() for p in self.core.parameters())
        return {"total_params": total, "unique_blocks": self.n_unique_blocks,
                "block_apps_final": self.exit_cost["final"],
                "core_params": core,
                "untied_equivalent_params": total + core * (self.cfg.n_passes - 1)}
