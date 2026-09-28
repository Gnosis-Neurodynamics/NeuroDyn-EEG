from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _elementwise(
    pred: torch.Tensor, target: torch.Tensor, kind: str = "huber", beta: float = 0.1
) -> torch.Tensor:
    """Element-wise loss (no reduction)."""
    if kind == "huber":
        return F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    if kind == "mse":
        return (pred - target).pow(2)
    return (pred - target).abs()


def composite_loss(
    pred_dict: Dict[str, torch.Tensor],
    target_dict: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
    kind: str = "huber",
    beta: float = 0.1,
    normalise: str = "inv_std",
    train_stats: Optional[Dict[str, Dict[str, float]]] = None,
    param_config: Optional[Dict[str, dict]] = None,
    use_bounds_mask: bool = True,
):
    if param_config is None:
        from .config_params import param_config as _pc

        param_config = _pc

    rec: Dict[str, float] = {}
    total = 0.0
    for k, p in pred_dict.items():
        t = target_dict[k].to(p.device, dtype=p.dtype)
        cfg = param_config.get(k, {})
        lo, hi = cfg.get("bounds", (-float("inf"), float("inf")))

        if use_bounds_mask:
            with torch.no_grad():
                mask = ((t >= lo) & (t <= hi)).to(p.dtype)
        else:
            mask = torch.ones_like(t)

        n_valid = mask.sum().clamp_min(1.0)
        in_rate = float((mask.sum() / mask.numel()).item())

        if normalise == "span":
            scale = float(hi - lo) if hi > lo else 1.0
        elif normalise == "inv_std":
            if train_stats is None or k not in train_stats:
                raise ValueError(f"normalise='inv_std' requires train_stats for '{k}'")
            scale = max(float(train_stats[k]["std"]), 1e-6)
        else:
            scale = 1.0

        l_elem = _elementwise(p / scale, t / scale, kind=kind, beta=beta)
        l_norm = (l_elem * mask).sum() / n_valid

        with torch.no_grad():
            mae_phys = ((p - t).abs() * mask).sum() / n_valid

        w = float((weights or {}).get(k, cfg.get("loss_weight", 1.0)))
        rec[k] = float(mae_phys.item())
        rec[k + "_norm"] = float(l_norm.detach().item())
        rec[k + "_in"] = in_rate
        total = total + w * l_norm
    return total, rec


def normalised_mae(
    pred: torch.Tensor, target: torch.Tensor, lo: float, hi: float
) -> torch.Tensor:
    """Backward-compat helper used by older scripts."""
    span = hi - lo
    if span <= 0:
        return F.l1_loss(pred, target)
    return F.l1_loss(pred / span, target / span)
