"""Migrated from plot_exp1_paper.py: all points, current configured parameters."""

from pathlib import Path
import numpy as np
import pandas as pd

from project_paths import resolve_project_path
from .common import (
    param_config,
    panel_grid,
    read_table,
    numeric,
    require_columns,
    validate_run_schema,
)
from .style import PALETTE, cm


def load_recovery(directory):
    directory = resolve_project_path(directory)
    validate_run_schema(directory)
    files = sorted(directory.glob("recovery_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No new recovery_*.npz files in {directory}; rerun unfiltered recovery"
        )
    truth = {name: [] for name in param_config}
    prediction = {name: [] for name in param_config}
    identifiers = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            ids = data["sample_id"].astype(str)
            if ids.ndim != 1 or not len(ids):
                raise ValueError(f"Invalid sample IDs: {path}")
            identifiers.extend(ids.tolist())
            for name, spec in param_config.items():
                a, b = data[f"true_{name}"], data[f"pred_{name}"]
                if a.shape != b.shape or a.shape != (len(ids), *spec["shape"]):
                    raise ValueError(f"Recovery shape mismatch: {path}/{name}")
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    raise ValueError(f"Nonfinite recovery values: {path}/{name}")
                # Original marginal-calibration figure averages all regions per sample.
                truth[name].append(a.mean(axis=1))
                prediction[name].append(b.mean(axis=1))
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate sample IDs across recovery chunks")
    return (
        {k: np.concatenate(v) for k, v in truth.items()},
        {k: np.concatenate(v) for k, v in prediction.items()},
        identifiers,
        files,
    )


def region_matrix(path):
    table = read_table(path)
    require_columns(table, ["parameter", "scope", "noise", "region_index", "pearson_r"])
    table = table[(table["noise"] == "clean") & (table["scope"] == "region")].copy()
    numeric(table, ["region_index"])
    numeric(table, ["pearson_r"], allow_nan=True)
    if table.duplicated(["parameter", "region_index"]).any():
        raise ValueError("Duplicate clean per-region metrics")
    names = [k for k, spec in param_config.items() if tuple(spec["shape"]) != (1,)]
    if not names:
        raise ValueError("No regional parameters configured")
    widths = {int(np.prod(param_config[name]["shape"])) for name in names}
    if len(widths) != 1:
        raise ValueError("Regional heatmap requires a common atlas shape")
    n = widths.pop()
    rows = []
    for name in names:
        selected = table[table["parameter"] == name].set_index("region_index")
        if set(selected.index) != set(range(n)):
            raise ValueError(f"Missing/extra brain regions for {name}")
        rows.append(selected.loc[list(range(n)), "pearson_r"].to_numpy())
    return names, np.stack(rows)


def plot_recovery(directory, out):
    import matplotlib.pyplot as plt
    from .colormaps import reconstruction_cmap, cbar_ticks_sparse

    directory = resolve_project_path(directory)
    truth, pred, ids, files = load_recovery(directory)
    for path in files:
        out.input(path)
    fig, axes = panel_grid(len(param_config))
    metrics = []
    for ax, name in zip(axes, param_config):
        a, b = truth[name], pred[name]
        ax.scatter(a, b, s=1.2, c=PALETTE[5], alpha=0.22, lw=0, rasterized=True)
        low, high = min(a.min(), b.min()), max(a.max(), b.max())
        pad = 0.02 * max(high - low, 1e-9)
        ax.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            color="0.35",
            lw=0.9,
            ls="--",
        )
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{name}  r={r:.3f}" if np.isfinite(r) else f"{name}  r=undefined",
            fontsize=8,
        )
        ax.set_xlabel(
            (
                "True"
                if tuple(param_config[name]["shape"]) == (1,)
                else "True (regional mean)"
            ),
            fontsize=7,
        )
        ax.set_ylabel("Predicted", fontsize=7)
        ax.tick_params(labelsize=6)
        metrics.append(
            {
                "parameter": name,
                "n_displayed": len(a),
                "n_total": len(a),
                "sample_mean_pearson_r": r,
            }
        )
    fig.tight_layout()
    out.save(fig, "recovery_sample_means")
    out.table(pd.DataFrame(metrics), "recovery_display_counts.csv")
    out.audit["recovery_samples"] = len(ids)
    out.audit["scatter_scope"] = (
        "sample mean across all configured regions; every sample displayed"
    )
    path = directory / "parameter_metrics.csv"
    out.input(path)
    names, matrix = region_matrix(path)
    fig, ax = plt.subplots(figsize=cm(21, max(5, 0.28 * len(names) + 2)))
    cmap = reconstruction_cmap(-1, 1)
    cmap.set_bad("#bdbdbd")
    im = ax.imshow(
        np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1, aspect="auto"
    )
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ticks = sorted(set([*range(0, matrix.shape[1], 5), matrix.shape[1] - 1]))
    ax.set_xticks(ticks, [i + 1 for i in ticks], fontsize=6)
    ax.set_xlabel("AAL region number (1-based)")
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which="minor", color="grey", linewidth=0.35)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_ticks(cbar_ticks_sparse(-1, 1))
    cb.set_label("Pearson r")
    if np.isnan(matrix).any():
        ax.set_title(
            "Grey cells: undefined correlation (constant target or prediction)",
            fontsize=8,
        )
    fig.tight_layout()
    out.save(fig, "recovery_region_heatmap")
    out.audit["undefined_region_correlations"] = int(np.isnan(matrix).sum())
