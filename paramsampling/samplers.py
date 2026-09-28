from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import norm

from .param_specs import PARAM_SPECS, TAU_RATIO_K, TAU_RATIO_MIN

_PHI = norm.cdf
_PHI_INV = norm.ppf


# ---------------------------------------------------------------------------
# Low-level helper: truncated normal (fast inverse-CDF impl)
# ---------------------------------------------------------------------------
def truncated_normal(
    low: float, high: float, mean, std, size, rng: np.random.Generator
) -> np.ndarray:
    mean_b = np.broadcast_to(np.asarray(mean, dtype=np.float64), size)
    std_b = np.broadcast_to(np.asarray(std, dtype=np.float64), size)

    safe = std_b > 1e-12
    if not safe.any():
        return np.clip(mean_b, low, high).astype(np.float64).copy()

    std_safe = np.where(safe, std_b, 1.0)
    a = (low - mean_b) / std_safe
    b = (high - mean_b) / std_safe

    cdf_a = _PHI(a)
    cdf_b = _PHI(b)
    span_cdf = cdf_b - cdf_a

    u = rng.uniform(size=size)
    degenerate = span_cdf < 1e-12

    z = _PHI_INV(cdf_a + u * span_cdf)
    sampled = mean_b + std_b * z
    sampled = np.clip(sampled, low, high)
    out = np.where(safe & ~degenerate, sampled, np.clip(mean_b, low, high))
    return out


# ---------------------------------------------------------------------------
# Hierarchical sampler for one parameter
# ---------------------------------------------------------------------------
def sample_subject_then_nodes(
    spec: dict, n_samples: int, n_nodes: int, rng: np.random.Generator
) -> np.ndarray:

    low, high = float(spec["bounds"][0]), float(spec["bounds"][1])
    span = high - low
    if span <= 0:
        return np.full((n_samples, n_nodes), low, dtype=np.float64)

    dist = spec.get("distribution", "normal_truncated")

    # ----- step 1: subject mean -----
    if dist == "uniform":
        subj_mean = rng.uniform(low=low, high=high, size=n_samples)
    else:
        sigma_subj = span * float(spec.get("subject_std_frac", 1 / 4))
        subj_mean = truncated_normal(
            low=low,
            high=high,
            mean=(low + high) / 2.0,
            std=sigma_subj,
            size=n_samples,
            rng=rng,
        )

    # ----- step 2: per-node values around subject mean -----
    sigma_node = span * float(spec.get("node_std_frac", 1 / 12))
    if sigma_node <= 0:
        return np.broadcast_to(subj_mean[:, None], (n_samples, n_nodes)).copy()

    nodes = truncated_normal(
        low=low,
        high=high,
        mean=subj_mean[:, None],  # (n_samples, 1) -> broadcast
        std=sigma_node,
        size=(n_samples, n_nodes),
        rng=rng,
    )
    return nodes


# ---------------------------------------------------------------------------
# tau_e -> tau_i coupling  (David & Friston 2003, k = 12/5)
# ---------------------------------------------------------------------------
def apply_tau_coupling(
    tau_e: np.ndarray,
    low_i: float,
    high_i: float,
    rng: np.random.Generator,
    k: float = TAU_RATIO_K,
    ratio_min: float = TAU_RATIO_MIN,
) -> np.ndarray:

    tau_e = np.asarray(tau_e, dtype=np.float64)
    out = np.empty_like(tau_e)

    safe = tau_e > 1e-9
    ratio_lo_eff = np.where(
        safe, np.maximum(ratio_min, low_i / np.where(safe, tau_e, 1.0)), ratio_min
    )
    ratio_hi_eff = np.where(safe, np.minimum(k, high_i / np.where(safe, tau_e, 1.0)), k)

    valid = ratio_lo_eff <= ratio_hi_eff
    u = rng.uniform(size=tau_e.shape)
    ratio = ratio_lo_eff + u * (ratio_hi_eff - ratio_lo_eff)
    tau_i_valid = ratio * tau_e
    tau_i_fallback = np.clip(tau_e, low_i, high_i)
    out = np.where(valid, tau_i_valid, tau_i_fallback)
    return np.clip(out, low_i, high_i)


# ---------------------------------------------------------------------------
# Top-level: sample one chunk
# ---------------------------------------------------------------------------
def sample_chunk(
    n_samples: int,
    n_nodes: int = 90,
    rng: Optional[np.random.Generator] = None,
    specs: Optional[Dict[str, dict]] = None,
    enforce_tau_coupling: bool = True,
    dtype=np.float64,
) -> Dict[str, np.ndarray]:

    if rng is None:
        rng = np.random.default_rng()
    if specs is None:
        specs = PARAM_SPECS

    out: Dict[str, np.ndarray] = {}

    for k, spec in specs.items():
        if spec.get("coupled_with") is not None:
            continue
        shape = tuple(spec["shape"])
        if shape == (1,):
            if spec.get("distribution") == "uniform":
                # Paper Eq. (S1.1): one global draw per sample, shared by all
                # regions. Do not apply the hierarchical node perturbation.
                low, high = map(float, spec["bounds"])
                arr = rng.uniform(low=low, high=high, size=(n_samples, 1))
            else:
                arr = sample_subject_then_nodes(
                    spec, n_samples=n_samples, n_nodes=1, rng=rng
                )
            out[k] = arr.astype(dtype)  # (n_samples, 1)
        else:
            arr = sample_subject_then_nodes(
                spec, n_samples=n_samples, n_nodes=n_nodes, rng=rng
            )
            out[k] = arr.astype(dtype)  # (n_samples, n_nodes)

    for k, spec in specs.items():
        coupled = spec.get("coupled_with")
        if coupled is None:
            continue
        low_i, high_i = float(spec["bounds"][0]), float(spec["bounds"][1])
        if enforce_tau_coupling:
            if coupled not in out:
                raise KeyError(
                    f"{k} coupled_with={coupled} but {coupled} not yet sampled. "
                    "Make sure tau_e* is sampled before tau_i*."
                )
            tau_e = out[coupled]
            tau_i = apply_tau_coupling(tau_e, low_i, high_i, rng)
            out[k] = tau_i.astype(dtype)
        else:
            arr = sample_subject_then_nodes(
                spec, n_samples=n_samples, n_nodes=n_nodes, rng=rng
            )
            out[k] = arr.astype(dtype)

    return out
