"""Configurable LightECGNet family for the dual-dataset upgrade (Phase 1). `lightecgnetv2.py` is NOT modified.

Default configuration == LightECGNetV2 exactly (same parameter names, bit-identical logits when the V2 state dict is loaded);
every deviation is an explicit flag so that each difference between the paper (text + Fig. 2-5) and the recovered V2 code can be tested separately.

Paper-vs-V2 flags (see docs/PAPER_VS_V2_ARCHITECTURE.md):
  la_bias            V2 True (162 params); False gives the 144 parameters quoted in the paper text
  la_relu_after_fc2  Fig. 3 labels FC2 "ReLU" before the sigmoid; V2 has none (default False)
  stem_proj_bn       V2 has BatchNorm between the stem 1x1 conv and PReLU; Fig. 4 shows conv + PReLU only
  block_variant      'v2' (dilated depthwise + 1x1 pointwise) | 'two_ds' (two dilated depthwise-separable layers) | 'two_conv' (two standard dilated convs, Fig. 5 literal)
  post_add_relu      V2 applies ReLU after the residual add; Fig. 5 shows the sum as the block output
  head_dropout       V2 0.4; paper Table III / Fig. 2: 0.5
  pool_fusion        'concat' (text and V2: 512-d) | 'sum' (Fig. 2 draws an element-wise sum; 256-d head input)
Unconfirmed in the paper and therefore NOT claimed as paper-faithful: TCN kernel sizes, block dropout rates, skip-projection form, BN placement.
"""
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from .lightecgnetv2 import DilatedTCNBlock, DSConv1d, LeadAttention, SEBlock1d


@dataclass
class ArchConfig:
    n_leads: int = 12
    la_bias: bool = True
    la_relu_after_fc2: bool = False
    stem_proj_bn: bool = True
    block_variant: str = "v2"
    post_add_relu: bool = True
    head_dropout: float = 0.4
    pool_fusion: str = "concat"
    kernels: tuple = (13, 13, 13, 13, 9, 9)
    block_dropouts: tuple = (0.1, 0.1, 0.2, 0.2, 0.3, 0.3)
    dilations: tuple = (1, 2, 4, 8, 1, 2)
    channels: tuple = (64, 64, 128, 128, 256, 256)
    hidden: int = 256

    def to_dict(self):
        return asdict(self)


class TwoLayerBlock(nn.Module):
    """Fig. 5 reading: (dilated conv -> BN -> ReLU) x 2 -> SE -> dropout -> (+ residual) [-> ReLU]. `kind` selects what 'Dilated Conv1d' means (not shown in the paper)."""

    def __init__(self, cin, cout, k, d, drop, kind, post_add_relu):
        super().__init__()
        pad = (k - 1) * d // 2

        def conv(i, o):
            if kind == "two_ds":
                return nn.Sequential(nn.Conv1d(i, i, k, padding=pad, dilation=d, groups=i, bias=False), nn.Conv1d(i, o, 1, bias=False))
            return nn.Conv1d(i, o, k, padding=pad, dilation=d, bias=False)
        self.c1, self.b1 = conv(cin, cout), nn.BatchNorm1d(cout)
        self.c2, self.b2 = conv(cout, cout), nn.BatchNorm1d(cout)
        self.se, self.drop, self.relu = SEBlock1d(cout), nn.Dropout(drop), nn.ReLU()
        self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout)) if cin != cout else nn.Identity()
        self.post = post_add_relu

    def forward(self, x):
        r = self.skip(x)
        x = self.relu(self.b1(self.c1(x)))
        x = self.relu(self.b2(self.c2(x)))
        x = self.drop(self.se(x)) + r
        return self.relu(x) if self.post else x


class LeadAttentionCfg(LeadAttention):
    def __init__(self, n_leads=12, reduction=2, bias=True, relu_after_fc2=False):
        super().__init__(n_leads, reduction)
        if not bias:
            self.fc1 = nn.Linear(n_leads, self.fc1.out_features, bias=False)
            self.fc2 = nn.Linear(self.fc1.out_features, n_leads, bias=False)
        self.relu2 = relu_after_fc2

    def forward(self, x):
        h = self.fc2(torch.relu(self.fc1(x.mean(-1))))
        if self.relu2:
            h = torch.relu(h)
        return x * torch.sigmoid(h).unsqueeze(-1)


def _head(cfg, n_out):
    d_in = 512 if cfg.pool_fusion == "concat" else 256
    return nn.Sequential(nn.Linear(d_in, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.head_dropout), nn.Linear(cfg.hidden, n_out))


class LightECGNet(nn.Module):
    """Backbone (+ optional single head). Parameter names equal LightECGNetV2's for the default configuration."""

    def __init__(self, num_classes=45, cfg=None, include_head=True):
        super().__init__()
        cfg = cfg or ArchConfig()
        self.cfg = cfg
        n = cfg.n_leads
        self.lead_attn = LeadAttentionCfg(n, 2, cfg.la_bias, cfg.la_relu_after_fc2)
        self.stem_k3, self.stem_k7, self.stem_k15 = DSConv1d(n, 16, 3), DSConv1d(n, 16, 7), DSConv1d(n, 32, 15)
        self.stem_proj = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.PReLU()) if cfg.stem_proj_bn else nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.PReLU())
        self.stem_pool = nn.AvgPool1d(2)
        names = ["tcn1a", "tcn1b", "tcn2a", "tcn2b", "tcn3a", "tcn3b"]
        cin = 64
        for nm, k, d, dr, co in zip(names, cfg.kernels, cfg.dilations, cfg.block_dropouts, cfg.channels):
            blk = DilatedTCNBlock(cin, co, k, d, dr) if cfg.block_variant == "v2" else TwoLayerBlock(cin, co, k, d, dr, cfg.block_variant, cfg.post_add_relu)
            setattr(self, nm, blk)
            cin = co
        self.pool1, self.pool2 = nn.AvgPool1d(2), nn.AvgPool1d(2)
        self.head = _head(cfg, num_classes) if include_head else None
        if cfg.block_variant == "v2" and not cfg.post_add_relu:
            raise ValueError("block_variant='v2' always applies the post-add ReLU")

    def features(self, x):
        x = self.lead_attn(x)
        x = torch.cat([self.stem_k3(x), self.stem_k7(x), self.stem_k15(x)], dim=1)
        x = self.stem_pool(self.stem_proj(x))
        x = self.pool1(self.tcn1b(self.tcn1a(x)))
        x = self.pool2(self.tcn2b(self.tcn2a(x)))
        fmap = self.tcn3b(self.tcn3a(x))
        if self.cfg.pool_fusion == "concat":
            pooled = torch.cat([fmap.max(dim=-1).values, fmap.mean(dim=-1)], dim=1)
        else:
            pooled = fmap.max(dim=-1).values + fmap.mean(dim=-1)
        return fmap, pooled

    def forward_features(self, x):
        fmap, pooled = self.features(x)
        out = {"feature_map": fmap, "pooled": pooled}
        if self.head is not None:
            hidden = self.head[1](self.head[0](pooled))
            out.update(hidden=hidden, logits=self.head[3](self.head[2](hidden)))
        return out

    def forward(self, x):
        return self.forward_features(x)["logits"]


class DualHeadLightECGNet(nn.Module):
    """Shared backbone + dataset-specific heads. forward(x, head) returns the logits of ONE head; no label space is shared."""

    def __init__(self, heads=None, cfg=None):
        super().__init__()
        heads = heads or {"physionet": 45, "ptbxl": 5}
        self.backbone = LightECGNet(include_head=False, cfg=cfg)
        self.heads = nn.ModuleDict({k: _head(self.backbone.cfg, c) for k, c in heads.items()})

    def pooled(self, x):
        return self.backbone.features(x)[1]

    def forward(self, x, head):
        p = self.pooled(x)
        h = self.heads[head]
        return h[3](h[2](h[1](h[0](p))))

    def forward_all(self, x):
        p = self.pooled(x)
        return {k: h[3](h[2](h[1](h[0](p)))) for k, h in self.heads.items()}


def v2_state_to_dual(state, head="physionet"):
    """Map a LightECGNetV2 state dict onto DualHeadLightECGNet keys (backbone.* and heads.<head>.*)."""
    out = {}
    for k, v in state.items():
        out[(f"heads.{head}." + k[len("head."):]) if k.startswith("head.") else "backbone." + k] = v
    return out


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
