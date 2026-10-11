"""LightECGNetV2 as defined in the original training notebook
(HonsExAI/LightECGNet_v2_FIXED_faster.ipynb, cells 10-11).

Input  : (B, 12, 5000) float32, channels-first (lead order I..V6).
Output : (B, 45) raw logits (multi-label; apply sigmoid).

`forward_features` exposes the intermediate tensors without altering the
computation; `forward` is built on it, so logits are identical by construction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, padding=(kernel - 1) // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class SEBlock1d(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)

    def forward(self, x):
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(-1)))))
        return x * s.unsqueeze(-1)


class LeadAttention(SEBlock1d):
    """Per-sample gating over the 12 leads (same form as SE, reduction=2)."""
    def __init__(self, n_leads=12, reduction=2):
        nn.Module.__init__(self)
        mid = max(n_leads // reduction, 4)
        self.fc1 = nn.Linear(n_leads, mid)
        self.fc2 = nn.Linear(mid, n_leads)


class DilatedTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, drop):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, padding=(kernel - 1) * dilation // 2,
                            dilation=dilation, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1d(out_ch)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.skip = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch))
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        r = self.skip(x)
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        x = self.drop(self.se(x))
        return self.relu(x + r)


class LightECGNetV2(nn.Module):
    def __init__(self, num_classes=45, n_leads=12):
        super().__init__()
        self.lead_attn = LeadAttention(n_leads, reduction=2)
        self.stem_k3 = DSConv1d(n_leads, 16, 3)
        self.stem_k7 = DSConv1d(n_leads, 16, 7)
        self.stem_k15 = DSConv1d(n_leads, 32, 15)
        self.stem_proj = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), nn.PReLU())
        self.stem_pool = nn.AvgPool1d(2)
        self.tcn1a = DilatedTCNBlock(64, 64, 13, 1, 0.1)
        self.tcn1b = DilatedTCNBlock(64, 64, 13, 2, 0.1)
        self.pool1 = nn.AvgPool1d(2)
        self.tcn2a = DilatedTCNBlock(64, 128, 13, 4, 0.2)
        self.tcn2b = DilatedTCNBlock(128, 128, 13, 8, 0.2)
        self.pool2 = nn.AvgPool1d(2)
        self.tcn3a = DilatedTCNBlock(128, 256, 9, 1, 0.3)
        self.tcn3b = DilatedTCNBlock(256, 256, 9, 2, 0.3)
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward_features(self, x):
        """Return dict: feature_map (B,256,T'), pooled (B,512)=[max|mean], hidden (B,256), logits (B,C)."""
        x = self.lead_attn(x)
        x = torch.cat([self.stem_k3(x), self.stem_k7(x), self.stem_k15(x)], dim=1)
        x = self.stem_pool(self.stem_proj(x))
        x = self.pool1(self.tcn1b(self.tcn1a(x)))
        x = self.pool2(self.tcn2b(self.tcn2a(x)))
        fmap = self.tcn3b(self.tcn3a(x))
        pooled = torch.cat([fmap.max(dim=-1).values, fmap.mean(dim=-1)], dim=1)
        hidden = self.head[1](self.head[0](pooled))      # Linear(512,256) -> ReLU
        logits = self.head[3](self.head[2](hidden))      # Dropout -> Linear(256,C)
        return {"feature_map": fmap, "pooled": pooled, "hidden": hidden, "logits": logits}

    def forward(self, x):
        return self.forward_features(x)["logits"]
