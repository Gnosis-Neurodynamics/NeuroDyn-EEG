"""Paths, audit output, and parameter contracts (no third-party model imports)."""

import argparse
import csv
import hashlib
import json
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from project_paths import DEFAULT_VALIDATION_CONFIG, resolve_project_path
from TrainWorks.src.config_params import param_config

CHANNELS = [
    "FP1",
    "FP2",
    "FZ",
    "F3",
    "F4",
    "F7",
    "F8",
    "CZ",
    "C3",
    "C4",
    "PZ",
    "P3",
    "P4",
    "T3",
    "T4",
    "T5",
    "T6",
    "O1",
    "O2",
]


def load_config(path=DEFAULT_VALIDATION_CONFIG):
    with open(resolve_project_path(path), encoding="utf-8") as stream:
        cfg = json.load(stream)
    for key, value in cfg["paths"].items():
        if key != "simulation_glob":
            cfg["paths"][key] = str(resolve_project_path(value))
    if cfg["real"]["sfreq"] != 256 or cfg["real"]["segment_seconds"] != 5:
        raise ValueError("Current trained model requires 256 Hz, 5-second segments")
    if cfg["real"]["channels"] != CHANNELS:
        raise ValueError(
            "Channel order must match the released inverse network's 19-channel order"
        )
    for override in cfg["real"]["cohort_overrides"].values():
        if "channels" in override and override["channels"] != CHANNELS:
            raise ValueError(
                "Cohort overrides cannot change inverse-model channel order"
            )
    return cfg


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", default=str(DEFAULT_VALIDATION_CONFIG))
    result.add_argument(
        "--output", help="New output directory; relative to project root"
    )
    return result


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def safe(value):
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value

    path.write_text(
        json.dumps(safe(data), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def new_output(cfg, stage, output=None):
    path = (
        resolve_project_path(output)
        if output
        else Path(cfg["paths"]["results_dir"])
        / (stage + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    )
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "config.json", cfg)
    write_json(path / "parameter_schema.json", param_config)
    versions = {}
    for name in ("numpy", "scipy", "torch", "scikit-learn", "mne"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(
        path / "runtime.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
        },
    )
    return path


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_for(seed, *keys):
    payload = json.dumps([seed, *keys], ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def feature_names():
    return [
        f"{key}:{i}"
        for key, spec in param_config.items()
        for i in range(int(np.prod(spec["shape"])))
    ]


def atlas_labels():
    path = Path(__file__).resolve().parents[1] / "config" / "aal_regions.csv"
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 90:
        raise ValueError("Expected AAL-90 mapping in simulator order")
    return rows


def flatten_parameters(params):
    from paramsampling.chunk_io import validate_params

    validate_params(params, param_config)
    return np.concatenate(
        [np.asarray(params[k]).reshape(len(params[k]), -1) for k in param_config],
        axis=1,
    )


def load_features(path, require_parameters=False):
    with np.load(path, allow_pickle=False) as file:
        data = {key: file[key] for key in file.files}
    required = (
        "features",
        "dataset",
        "subject_id",
        "label",
        "segment_id",
        "official_split",
    )
    for key in required:
        if key not in data:
            raise ValueError(f"{path}: missing {key}")
    n = len(data["features"])
    if not n or data["features"].ndim != 2 or not np.isfinite(data["features"]).all():
        raise ValueError(
            "Features must be a nonempty finite (segments, features) matrix"
        )
    for key in required[1:]:
        if data[key].shape != (n,):
            raise ValueError(f"{key} must have one value per segment")
    if not set(data["label"].tolist()) <= {0, 1}:
        raise ValueError("Labels must be 0=control/normal, 1=case/abnormal")
    if len(set(data["segment_id"].tolist())) != n:
        raise ValueError("Duplicate segment IDs")
    for dataset, subject in sorted(set(zip(data["dataset"], data["subject_id"]))):
        select = (data["dataset"] == dataset) & (data["subject_id"] == subject)
        for key in ("label", "official_split"):
            if len(np.unique(data[key][select])) != 1:
                raise ValueError(f"Conflicting {key}: {dataset}/{subject}")
    if (
        require_parameters
        and data.get("feature_names", np.array([])).tolist() != feature_names()
    ):
        raise ValueError(
            "Parameter features do not match current config_params.py order/shapes"
        )
    return data
