"""Real EEG (right panel) migrated from exp7/exp8 plot_topomap_pair and exp6.

Electrode coordinates, contour style and reconstruction colour map are preserved.
Every subject contributes once after averaging their retained segments per channel.
"""

import inspect
import numpy as np
import pandas as pd

from .common import CHANNELS, read_table, require_columns, numeric, finite_limits
from .style import cm, PALETTE

POS_DICT = {
    "FP1": [-0.024013627731023217, 0.08698885159208125],
    "FP2": [0.024386323846476342, 0.08794679166604386],
    "F3": [-0.03460004565821715, 0.05482686651769638],
    "F4": [0.03584847970265381, 0.05587744366121811],
    "C3": [-0.041938461319424795, 0.01074710762728574],
    "C4": [0.04315873978455615, 0.01122077263830799],
    "P3": [-0.034358363507553606, -0.032945542395651777],
    "P4": [0.036122943531615936, -0.03281364366136248],
    "O1": [-0.021100224032472345, -0.06308462553879438],
    "O2": [0.021407628169137917, -0.0628630906229748],
    "F7": [-0.05648759890871835, 0.052272893723496354],
    "F8": [0.05916068508703486, 0.0542319210392773],
    "T3": [-0.06458551243245102, 0.0052604677218147],
    "T4": [0.06545127432193645, 0.006048994724130777],
    "T5": [-0.05368939741176412, -0.03698805140934504],
    "T6": [0.05418721101109683, -0.03671756990993602],
    "FZ": [0.0001841097070134006, 0.05727051804372835],
    "CZ": [0.000224651498899583, 0.013418094415170482],
    "PZ": [0.0001892810684180513, -0.031909429078570986],
}

ALIASES = {
    "psd_log_r": "log_psd_r",
    "slope_err": "slope_abs_error",
    "alpha_peak_err": "alpha_peak_abs_error_hz",
    "time_plv_delta": "plv_delta",
    "time_plv_theta": "plv_theta",
    "time_plv_alpha": "plv_alpha",
    "subject": "subject_id",
}
METRICS = [
    "log_psd_r",
    "alpha_peak_abs_error_hz",
    "slope_abs_error",
    "plv_delta",
    "plv_theta",
    "plv_alpha",
]
SPEC = {
    "log_psd_r": ((0, 1), True, "log-PSD Pearson r", "log-PSD correlation (1–40 Hz)"),
    "alpha_peak_abs_error_hz": (
        (0, 1),
        False,
        "alpha-peak |error| (Hz)",
        "Alpha-peak frequency error",
    ),
    "slope_abs_error": ((0, 3), False, "1/f slope |error|", "Aperiodic slope error"),
}


def prepare_real_metrics(frame, dataset_label=None):
    frame = frame.copy()
    for old, new in ALIASES.items():
        if old in frame and new not in frame:
            frame = frame.rename(columns={old: new})
    if "ch_name" in frame:
        frame["channel"] = frame["ch_name"]
    if "dataset" not in frame:
        if not dataset_label:
            raise ValueError(
                "Legacy single-cohort input needs --dataset-label; subjects must have a dataset identity"
            )
        frame["dataset"] = dataset_label
    require_columns(frame, ["dataset", "subject_id", "channel", *METRICS])
    for col in ("dataset", "subject_id", "channel"):
        if frame[col].isna().any() or (frame[col].astype(str).str.len() == 0).any():
            raise ValueError(
                f"Missing {col}; cannot silently discard unidentified rows"
            )
    frame["channel"] = (
        frame["channel"]
        .astype(str)
        .str.upper()
        .replace({"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"})
    )
    if set(frame["channel"]) != set(CHANNELS):
        raise ValueError("Real EEG topomaps require exactly the configured 19 channels")
    numeric(frame, METRICS)
    if ((frame["log_psd_r"] < -1.000001) | (frame["log_psd_r"] > 1.000001)).any():
        raise ValueError("Correlation outside [-1,1]")
    for name in METRICS[3:]:
        if ((frame[name] < 0) | (frame[name] > 1.000001)).any():
            raise ValueError(f"Invalid PLV: {name}")
    if (frame[METRICS[1:3]] < 0).any().any():
        raise ValueError("Absolute errors must be nonnegative")
    trial = ["dataset", "subject_id"]
    if "segment_id" in frame:
        trial += ["segment_id"]
    elif "sample" in frame:
        trial += (["record_id"] if "record_id" in frame else []) + ["sample"]
    if frame[trial].isna().any().any() or frame.duplicated([*trial, "channel"]).any():
        raise ValueError(
            "Missing or duplicate trial/channel identity; supply complete, uniquely identified records"
        )
    if (frame.groupby(trial, dropna=False)["channel"].nunique() != 19).any():
        raise ValueError(
            "Every trial (or pre-averaged subject) must include all 19 channels; no channel-based subject removal"
        )
    subject_channels = frame.groupby(
        ["dataset", "subject_id", "channel"], as_index=False
    )[METRICS].mean()
    subject_values = subject_channels.groupby(
        ["dataset", "subject_id"], as_index=False
    )[METRICS].mean()
    channel_values = subject_channels.groupby(["dataset", "channel"], as_index=False)[
        METRICS
    ].mean()
    return subject_channels, subject_values, channel_values


def draw_right_topomap(channels, metric, out, name, title):
    import matplotlib.pyplot as plt
    import mne
    from .colormaps import reconstruction_cmap, cbar_ticks_sparse

    values = channels.set_index("channel").loc[CHANNELS, metric].to_numpy(dtype=float)
    defaults, higher_is_better, label, description = SPEC[metric]
    low, high = finite_limits(values, defaults)
    # Keep the old nominal scale when possible, expand it to include unfiltered values.
    low, high = np.floor(low * 10) / 10, np.ceil(high * 10) / 10
    cmap = reconstruction_cmap(low, high, higher_is_better=higher_is_better, step=0.1)
    fig, ax = plt.subplots(figsize=cm(10, 8))
    kwargs = dict(
        axes=ax,
        show=False,
        contours=5,
        cmap=cmap,
        mask=np.ones(19, dtype=bool),
        mask_params={"markersize": 4, "markerfacecolor": "0.25"},
        extrapolate="head",
    )
    if "vlim" in inspect.signature(mne.viz.plot_topomap).parameters:
        kwargs["vlim"] = (low, high)
    else:
        kwargs.update(vmin=low, vmax=high)
    im, _ = mne.viz.plot_topomap(
        values, np.array([POS_DICT[channel] for channel in CHANNELS]), **kwargs
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8, anchor=(0, 0.5))
    cb.set_ticks(cbar_ticks_sparse(low, high))
    cb.ax.tick_params(labelsize=8)
    cb.set_label(label, fontsize=9)
    ax.set_title(title + "\n" + description, fontsize=10, pad=8)
    fig.tight_layout()
    out.save(fig, name)
    out.audit.setdefault("topomaps", []).append(
        {
            "figure": name,
            "panel": "Real EEG (original right panel)",
            "n_channels": 19,
            "metric": metric,
            "vmin": low,
            "vmax": high,
        }
    )


def plot_real_eeg(path, out, dataset_label=None):
    import matplotlib.pyplot as plt

    frame = read_table(path)
    out.input(path)
    subject_channels, subjects, channels = prepare_real_metrics(frame, dataset_label)
    out.table(subject_channels, "real_eeg_subject_channels.csv")
    out.table(subjects, "real_eeg_subjects.csv")
    out.table(channels, "real_eeg_channel_means.csv")
    out.audit.update(
        n_input_rows=len(frame),
        n_subjects=len(subjects),
        averaging="all input segments per subject/channel, then equal weight per subject",
    )
    for i, dataset in enumerate(subjects["dataset"].unique()):
        subject = subjects[subjects["dataset"] == dataset]
        channel = channels[channels["dataset"] == dataset]
        prefix = f"real_eeg_{i:02d}"
        title = f"Real EEG · {dataset} · n={len(subject)} subjects"
        for metric in SPEC:
            draw_right_topomap(channel, metric, out, f"{prefix}_{metric}", title)
        fig, ax = plt.subplots(figsize=cm(10, 6))
        values = [
            subject[f"plv_{band}"].to_numpy() for band in ("delta", "theta", "alpha")
        ]
        bp = ax.boxplot(
            values,
            patch_artist=True,
            showfliers=True,
            widths=0.6,
            medianprops={"color": ".2"},
        )
        for box, color in zip(bp["boxes"], ["#1FC3CE", "#5DA9E9", "#B39DDB"]):
            box.set_facecolor(color)
            box.set_alpha(0.9)
            box.set_edgecolor(".3")
        ax.set_xticks([1, 2, 3], ["Delta", "Theta", "Alpha"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Time-domain PLV")
        ax.set_title(title, fontsize=9)
        fig.tight_layout()
        out.save(fig, f"{prefix}_plv")
        for metric, (_, _, label, _) in SPEC.items():
            fig, ax = plt.subplots(figsize=cm(10, 6))
            ax.hist(subject[metric], bins="auto", color=PALETTE[5], alpha=0.85)
            ax.set_xlabel(label)
            ax.set_ylabel("Subjects")
            ax.set_title(title, fontsize=9)
            fig.tight_layout()
            out.save(fig, f"{prefix}_{metric}_distribution")
