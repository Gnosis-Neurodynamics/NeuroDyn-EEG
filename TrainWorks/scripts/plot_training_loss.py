"""Per-parameter **train vs val physical MAE** in one figure.

Parameters and their order come from ``TrainWorks.src.config_params.param_config``,
the same configuration used by training. The subplot grid adapts to this list.

Train and validation use the **same metric**: masked physical MAE in original
units (jsonl ``rec[k]`` per step, same definition as ``validate()`` in
``train.py``). Training curve = **epoch-wise mean** of batch MAE over all steps
in that epoch; validation = ``[val]`` log per epoch.

Single **shared Y-axis** per subplot (no twin axis), so curves are directly
comparable.

Usage (repo root):

    python -m TrainWorks.scripts.plot_training_loss \\
        [--jsonl TrainWorks/logs/neurodynEEG.jsonl] \\
        [--log TrainWorks/logs/neurodynEEG.log] \\
        [--out TrainWorks/logs/neurodynEEG_train_val.png]

Default output: ``<stem>_train_val.png``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from TrainWorks.src.config_params import param_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_jsonl() -> Path:
    return _repo_root() / "TrainWorks" / "logs" / "neurodynEEG.jsonl"


def param_order() -> list[str]:
    return list(param_config.keys())


_VAL_HEAD = re.compile(
    r"\[val\]\s+epoch\s+(\d+)\s+mean_phys_mae=([0-9.eE+-]+)\s+(.*)",
)
_PARAM_VAL = re.compile(
    r"([a-zA-Z_]\w*)=([0-9.eE+-]+)\(in=([0-9.]+)%\)",
)


def load_jsonl(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Epoch per row, per-param masked physical MAE (``rec[k]`` in jsonl)."""
    steps: list[int] = []
    epochs_row: list[int] = []
    wants = param_order()
    phys: defaultdict[str, list[float]] = defaultdict(list)

    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if bad <= 5:
                    print(f"[warn] skip corrupt jsonl line {lineno}")
                continue
            if not steps:
                for name in wants:
                    if name not in rec:
                        raise SystemExit(f"first jsonl row missing {name!r} (phys MAE)")
            steps.append(int(rec["step"]))
            epochs_row.append(int(rec["epoch"]))
            for name in wants:
                phys[name].append(float(rec[name]))

    if bad:
        print(f"[warn] skipped {bad} corrupt jsonl line(s) total")

    pd_phys = {k: np.array(v, dtype=float) for k, v in phys.items()}
    return (
        np.array(steps, dtype=int),
        np.array(epochs_row, dtype=int),
        pd_phys,
    )


def mean_phys_mae_per_epoch(
    epoch_row: np.ndarray,
    phys: dict[str, np.ndarray],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = phys[name]
    ep_ids = np.unique(epoch_row)
    means = np.array([float(np.mean(y[epoch_row == e])) for e in ep_ids])
    return ep_ids, means


def parse_val_log(
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    epochs: list[int] = []
    means: list[float] = []
    branches_m: defaultdict[str, list[float]] = defaultdict(list)
    branches_in: defaultdict[str, list[float]] = defaultdict(list)

    if not log_path.is_file():
        return np.array([], dtype=int), np.array([], dtype=float), {}

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = _VAL_HEAD.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            mean_m = float(m.group(2))
            rest = m.group(3)
            epochs.append(ep)
            means.append(mean_m)
            for pm in _PARAM_VAL.finditer(rest):
                name, val, in_pct = pm.group(1), float(pm.group(2)), float(pm.group(3))
                branches_m[name].append(val)
                branches_in[name].append(in_pct)

    if not epochs:
        return np.array([], dtype=int), np.array([], dtype=float), {}

    n = len(epochs)
    per_m: dict[str, np.ndarray] = {}
    for k in param_order():
        if len(branches_m.get(k, [])) == n and len(branches_in.get(k, [])) == n:
            per_m[k] = np.array(branches_m[k], dtype=float)

    return np.array(epochs, dtype=int), np.array(means, dtype=float), per_m


def plot_train_val_mae(
    *,
    epoch_row: np.ndarray,
    phys: dict[str, np.ndarray],
    val_epochs: np.ndarray,
    val_per_mae: dict[str, np.ndarray],
    val_mean_mae: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    names = param_order()
    if not names:
        raise SystemExit("param_config is empty; no training parameters to plot")
    ncols = min(4, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        sharex=False,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    color_train, color_val = "#ff7f0e", "#1f77b4"
    legend_handles: list | None = None

    for i, name in enumerate(names):
        ax = axes_flat[i]
        ep_t, mt = mean_phys_mae_per_epoch(epoch_row, phys, name)
        (h_tr,) = ax.plot(
            ep_t,
            mt,
            color=color_train,
            linestyle="-",
            linewidth=1.5,
            alpha=0.9,
            label="train MAE (epoch mean)",
        )
        if name in val_per_mae:
            ym = val_per_mae[name]
            xe = val_epochs[: len(ym)]
            (h_va,) = ax.plot(
                xe,
                ym,
                color=color_val,
                linestyle="-",
                linewidth=1.2,
                marker="s",
                markersize=3.2,
                alpha=0.9,
                label="val MAE",
            )
        else:
            h_va = None

        if legend_handles is None:
            legend_handles = [h_tr] + ([h_va] if h_va is not None else [])

        ax.set_ylabel("MAE (phys)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(name, fontsize=10, fontweight="medium")
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", labelsize=7)

        lo_list: list[int] = []
        if len(epoch_row):
            lo_list.extend([int(epoch_row.min()), int(epoch_row.max())])
        if len(val_epochs):
            lo_list.extend([int(val_epochs.min()), int(val_epochs.max())])
        if lo_list:
            emin, emax = min(lo_list), max(lo_list)
            ax.set_xlim(emin - 0.35, emax + 0.35)

        # Single y-axis: include both series when present
        ys: list[float] = []
        ys.extend(mt.tolist())
        if name in val_per_mae:
            ys.extend(val_per_mae[name].tolist())
        if ys:
            y0, y1 = min(ys), max(ys)
            pad = (y1 - y0) * 0.08 + 1e-9
            ax.set_ylim(y0 - pad, y1 + pad)

        if i >= len(names) - ncols:
            ax.set_xlabel("epoch", fontsize=8)

    for j in range(len(names), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if legend_handles:
        labels = [h.get_label() for h in legend_handles]
        fig.legend(
            legend_handles,
            labels,
            loc="upper right",
            fontsize=9,
            framealpha=0.92,
        )

    subtitle = ""
    if len(val_epochs) and len(val_mean_mae) == len(val_epochs):
        subtitle = (
            f"Global val mean_phys_mae: {val_mean_mae[0]:.4f} → "
            f"{val_mean_mae[-1]:.4f}  (n={len(val_epochs)} epochs)"
        )
    warn_val = ""
    if len(val_epochs) == 0:
        warn_val = "\n[warn] No [val] lines — only train MAE shown."
    fig.suptitle(f"{title}\n{subtitle}{warn_val}", fontsize=11, y=1.02)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, default=None, help="Training .jsonl path")
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Training .log with [val] (default: stem of jsonl → .log)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default <stem>_train_val.png)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    jsonl_path = args.jsonl or _default_jsonl()
    if not jsonl_path.is_file():
        raise SystemExit(f"jsonl not found: {jsonl_path}")

    log_path = args.log
    if log_path is None:
        log_path = jsonl_path.with_suffix(".log")

    out_path = args.out or jsonl_path.with_name(f"{jsonl_path.stem}_train_val.png")

    steps, epoch_row, phys = load_jsonl(jsonl_path)
    if len(steps) == 0:
        raise SystemExit("jsonl is empty")

    ve, vm, v_mae = parse_val_log(log_path)

    plot_train_val_mae(
        epoch_row=epoch_row,
        phys=phys,
        val_epochs=ve,
        val_per_mae=v_mae,
        val_mean_mae=vm,
        title=f"Train vs val MAE (masked, phys) — {jsonl_path.name}",
        out_path=out_path,
    )

    print(
        f"Wrote {out_path}  (steps={len(steps)}, train epochs "
        f"{int(epoch_row.min())}–{int(epoch_row.max())}, val epochs={len(ve)})"
    )
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
