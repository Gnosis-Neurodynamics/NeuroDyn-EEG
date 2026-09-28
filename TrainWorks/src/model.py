"""NeuroDyn v6: Multi-Branch Spatial-Spectral Inversion (MB-SSI)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.fft as _torch_fft
import torch.nn as nn

from .config_params import param_config as _default_pc


class BoundSigmoid(nn.Module):
    def __init__(self, low: float, high: float):
        super().__init__()
        self.low, self.high = float(low), float(high)

    def forward(self, x):
        return self.low + (self.high - self.low) * torch.sigmoid(x)


class SoftBoundSigmoid(nn.Module):
    """Sigmoid bound with linear overshoot allowance ``alpha`` * span.

    out = low + span * (sigmoid(x) * (1 + 2*alpha) - alpha)
    so out_min = low - alpha*span (when sigmoid -> 0)
       out_max = high + alpha*span (when sigmoid -> 1)
    """

    def __init__(self, low: float, high: float, alpha: float = 0.1):
        super().__init__()
        self.low, self.high, self.alpha = float(low), float(high), float(alpha)

    def forward(self, x):
        span = self.high - self.low
        return self.low + span * (torch.sigmoid(x) * (1 + 2 * self.alpha) - self.alpha)


class ConstrainedLayer(nn.Module):
    """Inhibitory time-constant constraint, paper Eq. (4):

        tau_if = low + (k * tau_ef - low) * sigmoid(raw)

    Reachable range: [low, k * tau_ef]. With k = 12/5 = 2.4 (David et al. 2003).
    """

    def __init__(self, low: float, high: float, k: float = 12 / 5):
        super().__init__()
        self.low, self.high = float(low), float(high)
        self.k = float(k)

    def forward(self, tau_e: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        return self.low + (self.k * tau_e - self.low) * torch.sigmoid(raw)


class Conv1dPad(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation=1, bias=True):
        super().__init__()
        total_p = kernel_size + (kernel_size - 1) * (dilation - 1) - 1
        left_p = total_p // 2
        right_p = total_p - left_p
        self.net = nn.Sequential(
            nn.ConstantPad1d((left_p, right_p), 0),
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadConv(nn.Module):
    def __init__(self, in_ch, hidden, stride=2):
        super().__init__()
        unit = hidden // 4

        def make(k):
            return nn.Sequential(
                Conv1dPad(in_ch, unit, kernel_size=k, stride=stride, bias=False),
                nn.BatchNorm1d(unit),
                nn.LeakyReLU(0.2),
            )

        self.h1 = make(3)
        self.h2 = make(5)
        self.h3 = make(7)
        self.h4 = make(9)

    def forward(self, x):
        return torch.cat([self.h1(x), self.h2(x), self.h3(x), self.h4(x)], dim=1)


# ---------------------------------------------------------------------------
# Pseudo-inverse helper
# ---------------------------------------------------------------------------
def compute_lead_field_pinv(lead_field: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """L: (19, 90) -> L⁺: (90, 19), Tikhonov-regularised."""
    L = np.asarray(lead_field, dtype=np.float64)
    LtL = L.T @ L  # (90, 90)
    inv = np.linalg.solve(LtL + ridge * np.eye(L.shape[1]), L.T)  # (90, 19)
    return inv.astype(np.float32)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def sinusoidal_pe(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class DilatedResBlock(nn.Module):
    """Pre-activation residual block with dilated 1D convs.

    Used to enlarge the temporal receptive field of the scalp branch without
    further down-sampling. Takes (B, C, T) and returns (B, C, T).
    """

    def __init__(
        self, channels: int, kernel: int = 5, dilation: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel, dilation=dilation, padding=pad
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel, dilation=dilation, padding=pad
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv1(self.act1(self.bn1(x)))
        y = self.drop(y)
        y = self.conv2(self.act2(self.bn2(y)))
        return x + y


class SpectralBranch(nn.Module):
    """Per-channel log-magnitude FFT followed by a small CNN over (channel, freq).

    Input  : (B, C=19, T)
    Output : (B, d_spec)
    """

    def __init__(
        self, in_ch: int = 19, n_freq_bins: int = 64, hidden: int = 32, d_out: int = 64
    ):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        # treat channels as separate input feature dim, freq as the temporal axis
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden * 2, d_out)

    def forward(self, x):
        # x: (B, C, T) — assumed already z-scored upstream
        spec = _torch_fft.rfft(x, dim=-1)  # (B, C, T//2+1) complex
        mag = spec.abs()
        log_mag = torch.log1p(mag)
        log_mag = log_mag[..., : self.n_freq_bins]  # keep low-freq bins
        z = self.net(log_mag).flatten(1)  # (B, hidden*2)
        return self.proj(z)  # (B, d_out)


class SelfAttnBlock(nn.Module):
    """Pre-LN self-attention + FFN. PyTorch-1.7-safe (manual (L,B,d) transpose)."""

    def __init__(
        self, d_model: int, n_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, L, d)
        h = self.norm1(x).transpose(0, 1)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a.transpose(0, 1)
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttnBlock(nn.Module):

    def __init__(
        self, d_model: int, n_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm_f = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        q_n = self.norm_q(q).transpose(0, 1)
        k_n = self.norm_k(kv).transpose(0, 1)
        a, _ = self.attn(q_n, k_n, k_n, need_weights=False)
        q = q + a.transpose(0, 1)
        q = q + self.ff(self.norm_f(q))
        return q


class PerNodeTimeBackboneV6(nn.Module):

    def __init__(
        self, n_nodes: int = 90, hidden_per_node: int = 12, out_per_node: int = 24
    ):
        super().__init__()
        N, h, o = n_nodes, hidden_per_node, out_per_node
        self.net = nn.Sequential(
            nn.Conv1d(N, N * h, kernel_size=7, stride=4, padding=3, groups=N),
            nn.BatchNorm1d(N * h),
            nn.LeakyReLU(0.2),
            nn.Conv1d(N * h, N * h, kernel_size=5, stride=4, padding=2, groups=N),
            nn.BatchNorm1d(N * h),
            nn.LeakyReLU(0.2),
            nn.Conv1d(N * h, N * o, kernel_size=3, stride=2, padding=1, groups=N),
            nn.BatchNorm1d(N * o),
            nn.LeakyReLU(0.2),
        )
        self.N = N
        self.out_per_node = o

    def forward(self, src):  # (B, N, T)
        return self.net(src)  # (B, N*o, T_short)


# ---------------------------------------------------------------------------
# v6 main model
# ---------------------------------------------------------------------------
class NeuroDynEEG(nn.Module):
    def __init__(
        self,
        pick_params,
        in_nodes: int = 19,
        T: int = 1280,
        hidden_scalp: int = 128,
        d_model: int = 192,
        n_heads: int = 6,
        n_self_attn: int = 2,
        n_cross_attn: int = 2,
        ff_dim: int = 384,
        dropout: float = 0.1,
        pinv_out_per_node: int = 24,
        pinv_hidden_per_node: int = 12,
        node_emb_dim: int = 32,
        spec_freq_bins: int = 201,
        spec_hidden: int = 32,
        d_spec: int = 64,
        time_pool: int = 8,
        head_hidden: int = 128,
        bound_alpha=0.1,  # float OR dict[param_name -> float]
        lead_field: Optional[np.ndarray] = None,
        lead_field_pinv: Optional[np.ndarray] = None,
        pinv_ridge: float = 1e-3,
        source_nodes: int = 90,
        param_config: Optional[dict] = None,
    ):
        super().__init__()
        if param_config is None:
            param_config = _default_pc
        self.param_config = param_config
        self.pick_params = list(pick_params)
        self.bound_alpha = bound_alpha  # may be float OR dict[str -> float]
        self.N = int(source_nodes)
        self.d_model = int(d_model)
        self.time_pool = int(time_pool)

        # ---------- 1. lead-field pseudo-inverse buffer ----------
        if lead_field_pinv is None:
            if lead_field is None:
                raise ValueError(
                    "model requires lead_field (19,90) or " "lead_field_pinv (90,19)."
                )
            lead_field_pinv = compute_lead_field_pinv(lead_field, ridge=pinv_ridge)
        L_pinv = torch.from_numpy(np.asarray(lead_field_pinv, dtype=np.float32))
        assert L_pinv.shape == (
            self.N,
            in_nodes,
        ), f"L_pinv shape {tuple(L_pinv.shape)} != ({self.N}, {in_nodes})"
        self.register_buffer("L_pinv", L_pinv)

        # ---------- 2. scalp time branch ----------
        self.scalp_backbone = MultiHeadConv(in_nodes, hidden=hidden_scalp, stride=2)
        self.scalp_res1 = DilatedResBlock(
            hidden_scalp, kernel=5, dilation=2, dropout=dropout
        )
        self.scalp_res2 = DilatedResBlock(
            hidden_scalp, kernel=5, dilation=4, dropout=dropout
        )
        # project scalp time tokens to d_model for cross-attention KV
        self.scalp_proj = nn.Conv1d(hidden_scalp, d_model, kernel_size=1)

        # ---------- 3. source pseudo-inverse branch ----------
        self.source_backbone = PerNodeTimeBackboneV6(
            n_nodes=self.N,
            hidden_per_node=pinv_hidden_per_node,
            out_per_node=pinv_out_per_node,
        )
        self.pool_source = nn.AdaptiveAvgPool1d(1)

        # ---------- 4. spectral branch ----------
        self.spec_branch = SpectralBranch(
            in_ch=in_nodes, n_freq_bins=spec_freq_bins, hidden=spec_hidden, d_out=d_spec
        )

        # ---------- 5. node token construction & SC-graph self-attention ----------
        self.node_emb = nn.Parameter(torch.randn(self.N, node_emb_dim) * 0.02)
        node_in_dim = pinv_out_per_node + node_emb_dim
        self.node_proj = nn.Linear(node_in_dim, d_model)
        self.sc_blocks = nn.ModuleList(
            [
                SelfAttnBlock(d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
                for _ in range(n_self_attn)
            ]
        )

        # ---------- 6. scalp <-> source cross-attention ----------
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttnBlock(d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
                for _ in range(n_cross_attn)
            ]
        )
        # extra learnable global query (delay_scale 等 sample-level scalar 用它)
        self.global_query = nn.Parameter(torch.randn(1, d_model) * 0.02)

        # ---------- 7. dynamic time-token sequence length (for sinusoidal PE) ----------
        with torch.no_grad():
            dummy = torch.zeros(1, in_nodes, T)
            t_feat = self.scalp_backbone(dummy)  # (1, hidden_scalp, T/2)
            t_feat = self.scalp_res1(t_feat)
            t_feat = self.scalp_res2(t_feat)
            # pool to fixed length so cross-attn doesn't blow up
            t_pool = nn.AdaptiveAvgPool1d(self.time_pool)(t_feat)
            self.T_kv = int(t_pool.shape[-1])
        self.register_buffer("time_pe", sinusoidal_pe(self.T_kv, d_model))
        self.kv_pool = nn.AdaptiveAvgPool1d(self.time_pool)

        # ---------- 8. per-parameter heads ----------
        self.heads = nn.ModuleDict()
        self.constrains = nn.ModuleDict()
        # node head sees (d_model from cross-attn) + (d_spec broadcast)
        node_feat_total = d_model + d_spec
        scalar_feat_total = d_model + d_spec
        for k in self.pick_params:
            cfg = param_config[k]
            shape = cfg["shape"]
            if shape == (1,):
                self.heads[k] = nn.Sequential(
                    nn.Linear(scalar_feat_total, head_hidden),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )
            else:
                self.heads[k] = nn.Sequential(
                    nn.Linear(node_feat_total, head_hidden),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, head_hidden),
                    nn.LeakyReLU(0.2),
                    nn.Linear(head_hidden, 1),
                )
            low, high = cfg["bounds"]
            # resolve per-param alpha: bound_alpha can be a scalar or a dict
            if isinstance(self.bound_alpha, dict):
                a_k = float(
                    self.bound_alpha.get(k, self.bound_alpha.get("__default__", 0.1))
                )
            else:
                a_k = float(self.bound_alpha)
            if k.startswith("tau_i"):
                self.constrains[k] = ConstrainedLayer(low, high, k=12 / 5)
            elif a_k > 0:
                self.constrains[k] = SoftBoundSigmoid(low, high, alpha=a_k)
            else:
                self.constrains[k] = BoundSigmoid(low, high)

    # ------------------------------------------------------------------
    def _scalp_features(self, x):
        """x: (B, 19, T) -> kv_tokens (B, T_kv, d_model)."""
        f = self.scalp_backbone(x)  # (B, hidden_scalp, T/2)
        f = self.scalp_res1(f)
        f = self.scalp_res2(f)  # large RF, still T/2
        f = self.kv_pool(f)  # (B, hidden_scalp, T_kv)
        f = self.scalp_proj(f)  # (B, d_model, T_kv)
        return f.transpose(1, 2)  # (B, T_kv, d_model)

    def _source_features(self, x):
        """x: (B, 19, T) -> per-node pinv features (B, N, pinv_out_per_node)."""
        src_pinv = torch.einsum("ij,bjt->bit", self.L_pinv, x)  # (B, N, T)
        f = self.source_backbone(src_pinv)  # (B, N*o, T_short)
        f = self.pool_source(f).squeeze(-1)  # (B, N*o)
        B = x.shape[0]
        return f.view(B, self.N, -1)  # (B, N, o)

    def _build_node_tokens(self, z_pinv):
        """concat per-node pinv feature with learnable node embedding -> project to d_model."""
        B = z_pinv.shape[0]
        emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, e)
        z = torch.cat([z_pinv, emb], dim=-1)  # (B, N, o+e)
        return self.node_proj(z)  # (B, N, d_model)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """x: (B, 19, T) -> dict[k -> (B, *shape)]"""
        B = x.shape[0]

        # --- 1) scalp time tokens (KV memory) ---
        kv = self._scalp_features(x)  # (B, T_kv, d_model)
        kv = kv + self.time_pe.unsqueeze(0)  # broadcast (1,T_kv,d)

        # --- 2) source per-node features (Tikhonov inversion) ---
        z_pinv = self._source_features(x)  # (B, N, o)

        # --- 3) spectral global feature ---
        z_spec = self.spec_branch(x)  # (B, d_spec)

        # --- 4) node tokens + SC-graph self-attention ---
        node_tok = self._build_node_tokens(z_pinv)  # (B, N, d_model)
        # append global query for sample-level scalars
        glob_tok = self.global_query.unsqueeze(0).expand(B, -1, -1)  # (B,1,d)
        all_tok = torch.cat([node_tok, glob_tok], dim=1)  # (B, N+1, d)
        for blk in self.sc_blocks:
            all_tok = blk(all_tok)

        # --- 5) cross-attention: node tokens look at scalp time memory ---
        for blk in self.cross_blocks:
            all_tok = blk(all_tok, kv)

        node_final = all_tok[:, : self.N, :]  # (B, N, d_model)
        glob_final = all_tok[:, self.N, :]  # (B, d_model)

        # --- 6) per-param heads ---
        z_spec_n = z_spec.unsqueeze(1).expand(-1, self.N, -1)  # (B, N, d_spec)
        node_feat = torch.cat([node_final, z_spec_n], dim=-1)  # (B, N, d+d_spec)
        scalar_feat = torch.cat([glob_final, z_spec], dim=-1)  # (B, d+d_spec)

        raw = {}
        for k in self.pick_params:
            shape = self.param_config[k]["shape"]
            if shape == (1,):
                raw[k] = self.heads[k](scalar_feat).view(B, 1)
            else:
                raw[k] = self.heads[k](node_feat).squeeze(-1)  # (B, N)

        out = {}
        for k in self.pick_params:
            if k.startswith("tau_i"):
                ek = "tau_e" + k[-1]
                out[k] = self.constrains[k](out[ek], raw[k])
            else:
                out[k] = self.constrains[k](raw[k])
        return out


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pp = list(_default_pc.keys())
    L = np.random.randn(19, 90).astype(np.float32) * 0.01
    model = NeuroDynEEG(pick_params=pp, T=1280, lead_field=L)
    n = sum(p.numel() for p in model.parameters())
    print(f"v6 param count: {n / 1e6:.2f}M")
    print(f"v6 T_kv (cross-attn KV length) = {model.T_kv}")

    x = torch.randn(2, 19, 1280)
    out = model(x)
    for k, v in out.items():
        print(f"  {k:14s}: {tuple(v.shape)}")

    # backward smoke
    loss = sum(v.abs().mean() for v in out.values())
    loss.backward()
    print("backward OK")
