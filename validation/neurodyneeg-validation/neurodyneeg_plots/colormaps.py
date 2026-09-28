"""Shared muted jet stepped colormaps for topomap / heatmap figures."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def _desaturate_colors(colors, saturation=0.55):
    hsv = rgb_to_hsv(colors[:, :3])
    hsv[:, 1] *= saturation
    colors = colors.copy()
    colors[:, :3] = hsv_to_rgb(hsv)
    return colors


def _muted_jet_stops(n_hues=7, saturation=0.68):
    colors = colormaps["jet"](np.linspace(0, 1, n_hues))
    return _desaturate_colors(colors, saturation=saturation)[:, :3]


def stepped_multicolor_cmap(
    vmin=0,
    vmax=1,
    step=0.1,
    n_hues=7,
    saturation=0.68,
    reverse=False,
    samples_per_step=12,
    name=None,
):
    """0.1-unit color bands; each band maps to one of n_hues jet-like hues."""
    n_steps = max(1, int(round((vmax - vmin) / step)))
    hue_stops = _muted_jet_stops(n_hues=n_hues, saturation=saturation)
    if reverse:
        hue_stops = hue_stops[::-1]

    segments = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        hue_idx = t * (n_hues - 1)
        lo = int(np.floor(hue_idx))
        hi = min(lo + 1, n_hues - 1)
        frac = hue_idx - lo
        c_mid = (1.0 - frac) * hue_stops[lo] + frac * hue_stops[hi]
        c_light = np.clip(c_mid * 1.06 + 0.02, 0, 1)
        c_dark = np.clip(c_mid * 0.94, 0, 1)
        segments.append(np.linspace(c_light, c_dark, samples_per_step))
    colors = np.vstack(segments)
    return mcolors.LinearSegmentedColormap.from_list(
        name or f"muted_jet_step{step}", colors
    )


def reconstruction_cmap(vmin=0, vmax=1, higher_is_better=True, step=0.1):
    """Warm colors for better reconstruction (high metric or low error)."""
    return stepped_multicolor_cmap(
        vmin=vmin,
        vmax=vmax,
        step=step,
        reverse=not higher_is_better,
        n_hues=7,
    )


def cbar_ticks_sparse(vmin, vmax, max_ticks=6):
    span = vmax - vmin
    if span <= 0:
        return np.array([vmin])
    raw_step = span / max(max_ticks - 1, 1)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * magnitude
        if step >= raw_step:
            break
    ticks = np.arange(vmin, vmax + step * 0.5, step)
    return ticks[ticks <= vmax + 1e-9]
