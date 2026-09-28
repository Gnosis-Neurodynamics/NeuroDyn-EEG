"""Migrated waveform/PSD figures; explicit sample illustration, all-subject summary."""

import json
import numpy as np
import pandas as pd
from project_paths import resolve_project_path
from .common import CHANNELS, panel_grid
from .style import PALETTE, cm


def aligned_record(observed_path, reconstructed_path):
    with np.load(observed_path, allow_pickle=False) as data:
        observed, original_ids = data["eeg"], data["segment_id"].astype(str)
    with np.load(reconstructed_path, allow_pickle=False) as data:
        reconstructed, ids = data["reconstructed"], data["segment_id"].astype(str)
    if (
        ids.ndim != 1
        or len(ids) != len(observed)
        or len(set(ids)) != len(ids)
        or not np.array_equal(original_ids, ids)
    ):
        raise ValueError(
            "Closed-loop records must have identical ordered segment IDs; no intersecting/cropping"
        )
    if (
        observed.shape != reconstructed.shape
        or observed.shape[1:] != (19, 1280)
        or not len(observed)
    ):
        raise ValueError(
            "Observed/reconstructed EEG must both have exact shape (N,19,1280)"
        )
    if not np.isfinite(observed).all() or not np.isfinite(reconstructed).all():
        raise ValueError("Nonfinite observed/reconstructed signals")
    return observed, reconstructed, ids


def plot_signals(results, prepared, out, segment_id=None):
    from scipy.signal import welch
    import matplotlib.pyplot as plt

    results, prepared = resolve_project_path(results), resolve_project_path(prepared)
    index = json.loads((prepared / "index.json").read_text(encoding="utf-8"))
    if not index.get("complete") or not index.get("records"):
        raise ValueError("Prepared cache is incomplete")
    if len({r["file"] for r in index["records"]}) != len(index["records"]):
        raise ValueError("Duplicate recording files in prepared index")
    provenance = results / "forward_provenance.json"
    if provenance.exists():
        source_index = json.loads(provenance.read_text(encoding="utf-8"))[
            "prepared_index"
        ]
        for key in ("checkpoint_sha256", "manifest_sha256"):
            if source_index[key] != index[key]:
                raise ValueError(
                    "Reconstruction and observed-cache provenance do not match"
                )
        out.input(provenance)
    out.input(prepared / "index.json")
    groups, example, total, seen = {}, None, 0, set()
    for record in index["records"]:
        a_path, b_path = prepared / record["file"], results / record["file"]
        observed, reconstructed, ids = aligned_record(a_path, b_path)
        if seen.intersection(ids.tolist()):
            raise ValueError("Duplicate segment IDs across recordings")
        seen.update(ids.tolist())
        out.input(a_path)
        out.input(b_path)
        f, a = welch(observed, fs=256, nperseg=512, noverlap=256, axis=-1)
        _, b = welch(reconstructed, fs=256, nperseg=512, noverlap=256, axis=-1)
        frequency = (f >= 1) & (f <= 40)
        a, b = np.log10(np.maximum(a[..., frequency], 1e-30)).mean(1), np.log10(
            np.maximum(b[..., frequency], 1e-30)
        ).mean(1)
        key = (record["dataset"], record["subject_id"])
        if key not in groups:
            groups[key] = [np.zeros(a.shape[1]), np.zeros(a.shape[1]), 0]
        groups[key][0] += a.sum(0)
        groups[key][1] += b.sum(0)
        groups[key][2] += len(a)
        total += len(ids)
        if segment_id is not None and segment_id in ids:
            if example is not None:
                raise ValueError("Illustration ID occurs in multiple records")
            i = ids.tolist().index(segment_id)
            example = observed[i], reconstructed[i]
    out.audit.update(
        n_signal_segments=total,
        n_signal_subjects=len(groups),
        illustrative_segment_id=segment_id,
    )
    if segment_id is not None and example is None:
        raise ValueError("Requested illustration segment ID not found")
    rows = []
    frequencies = f[frequency]
    for i, dataset in enumerate(sorted({key[0] for key in groups})):
        entries = [v for key, v in groups.items() if key[0] == dataset]
        for kind, j in (("observed", 0), ("reconstructed", 1)):
            curves = np.stack([v[j] / v[2] for v in entries])
            mean, std = curves.mean(0), (
                curves.std(0, ddof=1) if len(curves) > 1 else np.zeros_like(curves[0])
            )
            for hz, m, s in zip(frequencies, mean, std):
                rows.append(
                    {
                        "dataset": dataset,
                        "signal": kind,
                        "frequency_hz": hz,
                        "mean_log10_psd": m,
                        "subject_std": s,
                        "n_subjects": len(entries),
                    }
                )
        fig, ax = plt.subplots(figsize=cm(14, 6))
        table = pd.DataFrame([r for r in rows if r["dataset"] == dataset])
        for kind, color in (("observed", PALETTE[0]), ("reconstructed", PALETTE[6])):
            data = table[table["signal"] == kind]
            ax.plot(
                data["frequency_hz"], data["mean_log10_psd"], color=color, label=kind
            )
            ax.fill_between(
                data["frequency_hz"],
                data["mean_log10_psd"] - data["subject_std"],
                data["mean_log10_psd"] + data["subject_std"],
                color=color,
                alpha=0.15,
                lw=0,
            )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("log10(PSD)")
        ax.set_title(f"{dataset}: all subjects (equal weight), mean ± SD", fontsize=9)
        ax.legend()
        fig.tight_layout()
        out.save(fig, f"closed_loop_psd_{i:02d}")
    out.table(pd.DataFrame(rows), "closed_loop_psd_summary.csv")
    if example is not None:
        fig, axes = panel_grid(19, height=4)
        for i, (ax, name) in enumerate(zip(axes, CHANNELS)):
            ax.plot(
                np.arange(1280) / 256,
                example[0][i],
                color=PALETTE[0],
                lw=0.6,
                label="observed",
            )
            ax.plot(
                np.arange(1280) / 256,
                example[1][i],
                color=PALETTE[5],
                lw=0.6,
                alpha=0.8,
                label="reconstructed",
            )
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("µV", fontsize=7)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        out.save(fig, "closed_loop_explicit_segment")
