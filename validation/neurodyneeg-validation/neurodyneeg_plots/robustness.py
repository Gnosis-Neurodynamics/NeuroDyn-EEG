"""Migrated per-parameter noise plots; no parameter exclusion or pooled raw MAE."""

import numpy as np
from project_paths import resolve_project_path
from TrainWorks.src.config_params import param_config
from .common import (
    read_table,
    require_columns,
    numeric,
    require_parameters,
    panel_grid,
    validate_run_schema,
)
from .style import PALETTE


def load_noise(path, scope):
    frame = read_table(path)
    require_columns(
        frame, ["scope", "parameter", "noise", "snr_db", "pearson_r", "mae", "r2"]
    )
    frame = frame[frame["scope"] == scope].copy()
    require_parameters(frame)
    numeric(frame, ["pearson_r", "r2", "mae"], allow_nan=True)
    clean, noisy = (
        frame[frame["noise"] == "clean"].copy(),
        frame[frame["noise"] != "clean"].copy(),
    )
    numeric(noisy, ["snr_db"])
    if (
        clean.duplicated("parameter").any()
        or noisy.duplicated(["parameter", "noise", "snr_db"]).any()
    ):
        raise ValueError("Duplicate noise-condition metrics")
    require_parameters(clean)
    expected = {
        (k, family, snr)
        for k in param_config
        for family in ("white", "pink", "emg", "eog")
        for snr in (30, 20, 10, 5, 0, -5)
    }
    actual = set(
        noisy[["parameter", "noise", "snr_db"]].itertuples(index=False, name=None)
    )
    if expected != actual:
        raise ValueError(
            "Noise results must contain every configured parameter and all four families at 30/20/10/5/0/-5 dB"
        )
    return clean, noisy


def plot_noise(directory, out, scope="pooled"):
    directory = resolve_project_path(directory)
    validate_run_schema(directory)
    path = directory / "parameter_metrics.csv"
    clean, noisy = load_noise(path, scope)
    out.input(path)
    out.audit["metric_scope"] = scope
    out.audit["undefined_metrics"] = int(
        noisy[["pearson_r", "r2", "mae"]].isna().sum().sum()
    )
    for metric, label in (
        ("pearson_r", "Pearson r"),
        ("r2", r"$R^2$"),
        ("mae", "MAE (parameter units)"),
    ):
        fig, axes = panel_grid(len(param_config))
        for ax, parameter in zip(axes, param_config):
            rows = noisy[noisy["parameter"] == parameter]
            for i, family in enumerate(("white", "pink", "emg", "eog")):
                curve = rows[rows["noise"] == family].sort_values(
                    "snr_db", ascending=False
                )
                ax.plot(
                    curve["snr_db"],
                    curve[metric],
                    "o-",
                    label=family,
                    color=PALETTE[i + 1],
                    ms=3,
                    lw=1,
                )
            baseline = clean.loc[clean["parameter"] == parameter, metric].item()
            ax.axhline(baseline, color=".5", ls="--", lw=1, label="clean")
            if rows[metric].isna().any() or not np.isfinite(baseline):
                ax.text(
                    0.03,
                    0.03,
                    "Undefined values shown as gaps",
                    transform=ax.transAxes,
                    fontsize=5,
                )
            ax.set_xlim(30, -5)
            ax.set_xticks([30, 20, 10, 5, 0, -5])
            ax.set_title(parameter, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("SNR (dB)", fontsize=7)
            ax.set_ylabel(label, fontsize=7)
            # Let every parameter retain its full range (including negative R²).
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out.save(fig, f"noise_{scope}_{metric}")
