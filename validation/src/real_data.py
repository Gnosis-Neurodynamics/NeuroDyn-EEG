"""Explicit recording metadata, deterministic segmentation, and shared clinical QC."""

import csv
import json
from pathlib import Path
import numpy as np

ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def canonical_channel(name):
    name = str(name).strip().upper()
    return ALIASES.get(name, name)


def read_manifest(cfg):
    with open(cfg["paths"]["manifest"], newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty participants.csv; fill in one row per recording")
    seen, subjects = set(), {}
    for row in rows:
        for key in ("dataset", "subject_id", "record_id", "label", "path"):
            if not row.get(key, "").strip():
                raise ValueError(f"Manifest row lacks {key}: {row}")
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1):
            raise ValueError("Use label 0=control/normal, 1=case/abnormal")
        row["official_split"] = row.get("official_split", "").strip().lower()
        if row["dataset"].upper() == "TUAB" and row["official_split"] not in (
            "train",
            "eval",
        ):
            raise ValueError(
                "TUAB needs official_split=train/eval and actual subject_id (not record ID)"
            )
        key = (row["dataset"], row["record_id"])
        if key in seen:
            raise ValueError(f"Duplicate recording: {key}")
        seen.add(key)
        key = (row["dataset"], row["subject_id"])
        value = (row["label"], row["official_split"])
        if key in subjects and subjects[key] != value:
            raise ValueError(f"Conflicting label or official split for {key}")
        subjects[key] = value
        path = Path(row["path"]).expanduser()
        row["path"] = str(
            (
                path if path.is_absolute() else Path(cfg["paths"]["real_dir"]) / path
            ).resolve()
        )
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    return rows


def read_recording(row):
    path = Path(row["path"])
    if path.suffix.lower() in (".npz", ".npy"):
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as data:
                eeg = data["eeg"]
                fs = float(data["sfreq"].item())
                unit = str(data["unit"].item())
                channels = data["ch_names"].tolist()
        else:
            eeg = np.load(path, allow_pickle=False)
            if not all(row.get(k) for k in ("sfreq", "unit", "channels")):
                raise ValueError(
                    f"{path}: .npy needs sfreq, unit, channels in manifest; use | between channels"
                )
            fs, unit, channels = (
                float(row["sfreq"]),
                row["unit"],
                row["channels"].split("|"),
            )
        scale = {"V": 1e6, "mV": 1e3, "uV": 1, "µV": 1}.get(unit)
        if scale is None:
            raise ValueError(f"Unsupported explicit EEG unit {unit!r} in {path}")
        eeg = np.asarray(eeg, dtype=np.float64) * scale
    else:
        import mne

        readers = {
            ".edf": mne.io.read_raw_edf,
            ".bdf": mne.io.read_raw_bdf,
            ".set": mne.io.read_raw_eeglab,
            ".fif": mne.io.read_raw_fif,
        }
        if path.suffix.lower() not in readers:
            raise ValueError(f"Unsupported EEG extension: {path.suffix}")
        raw = readers[path.suffix.lower()](str(path), preload=True, verbose="ERROR")
        eeg, fs, channels = raw.get_data() * 1e6, float(raw.info["sfreq"]), raw.ch_names
    if eeg.ndim not in (2, 3) or eeg.shape[-2] != len(channels):
        raise ValueError(
            f"{path}: require (channels,time) or (segments,channels,time); no layout guessing"
        )
    if not np.isfinite(eeg).all() or fs <= 0:
        raise ValueError(f"Invalid EEG or sampling rate: {path}")
    return eeg, fs, channels


def prepare_recording(row, cfg):
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly
    from fractions import Fraction

    options = dict(cfg["real"])
    options.update(options.pop("cohort_overrides").get(row["dataset"], {}))
    if options["sfreq"] != 256 or options["segment_seconds"] != 5:
        raise ValueError("All cohorts must match the 256 Hz / 5 s inverse input")
    eeg, fs, channels = read_recording(row)
    names = [canonical_channel(name) for name in channels]
    wanted = [canonical_channel(name) for name in options["channels"]]
    if len(wanted) != 19 or len(set(wanted)) != 19:
        raise ValueError("Exactly 19 distinct configured channels are required")
    if any(names.count(name) != 1 for name in wanted):
        raise ValueError(
            f"{row['path']}: missing/duplicate required channels; got {names}"
        )
    eeg = eeg[..., [names.index(name) for name in wanted], :]
    low, high = options["bandpass_hz"]
    if not 0 < low < high < fs / 2:
        raise ValueError(
            "Bandpass must be below acquisition Nyquist; set an explicit cohort override"
        )
    for frequency in options["notch_hz"]:
        if not 0 < frequency < fs / 2:
            raise ValueError("Notch frequency must be below acquisition Nyquist")
        b, a = iirnotch(frequency, Q=30, fs=fs)
        eeg = filtfilt(b, a, eeg, axis=-1)
    eeg = sosfiltfilt(
        butter(4, [low, high], btype="bandpass", fs=fs, output="sos"), eeg, axis=-1
    )
    if fs != options["sfreq"]:
        ratio = Fraction(options["sfreq"] / fs).limit_denominator(100000)
        eeg = resample_poly(eeg, ratio.numerator, ratio.denominator, axis=-1)
    window = int(options["sfreq"] * options["segment_seconds"])
    if eeg.ndim == 3:
        if eeg.shape[-1] != window:
            raise ValueError(
                "Pre-segmented EEG must contain exactly 5 seconds per segment; no truncation"
            )
        tail = 0
    else:
        count, tail = divmod(eeg.shape[-1], window)
        if count == 0:
            raise ValueError(f"Recording shorter than one full window: {row['path']}")
        # Only an incomplete final window is omitted, with its length recorded.
        eeg = eeg[:, : count * window].reshape(19, count, window).transpose(1, 0, 2)
    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    eeg = eeg - eeg.mean(axis=-2, keepdims=True)
    amplitude = np.ptp(eeg, axis=-1).mean(axis=-1)
    lo, hi = options["qc_mean_channel_ptp_uv"]
    if not 0 <= lo < hi:
        raise ValueError("Invalid QC amplitude range")
    keep = (amplitude >= lo) & (amplitude <= hi)
    qc = [
        {
            "dataset": row["dataset"],
            "subject_id": row["subject_id"],
            "record_id": row["record_id"],
            "segment_index": i,
            "mean_channel_ptp_uv": float(value),
            "included": bool(keep[i]),
            "reason": "included" if keep[i] else "amplitude_qc",
        }
        for i, value in enumerate(amplitude)
    ]
    details = {
        "record": row,
        "preprocessing": options,
        "input_sfreq": fs,
        "n_segments_before_qc": len(keep),
        "n_segments_retained": int(keep.sum()),
        "incomplete_tail_samples_at_256hz": tail,
    }
    return eeg[keep].astype(np.float32), np.flatnonzero(keep), qc, details


def segment_id(row, index):
    # JSON encoding makes identifiers unambiguous even if user IDs contain separators.
    return json.dumps(
        [row["dataset"], row["subject_id"], row["record_id"], int(index)],
        ensure_ascii=False,
    )
