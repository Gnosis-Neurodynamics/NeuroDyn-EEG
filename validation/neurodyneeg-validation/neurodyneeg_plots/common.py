"""Result contracts and reproducible figure output; no outcome-based selection."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import resolve_project_path
from TrainWorks.src.config_params import param_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
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


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Result table is empty")


def numeric(frame, columns, allow_nan=False):
    require_columns(frame, columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=float)
        if np.isinf(values).any() or (not allow_nan and np.isnan(values).any()):
            raise ValueError(
                f"Nonfinite {column}; no rows are silently removed for plotting"
            )
    return frame


def require_parameters(frame, regional=False):
    wanted = [
        name
        for name, spec in param_config.items()
        if not regional or tuple(spec["shape"]) != (1,)
    ]
    actual = set(frame["parameter"])
    if set(wanted) != actual:
        raise ValueError(
            f"Result parameters differ from config_params.py: missing={set(wanted)-actual}, extra={actual-set(wanted)}"
        )
    return wanted


def finite_limits(values, defaults=None):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot choose display limits from nonfinite data")
    lo, hi = float(values.min()), float(values.max())
    if defaults:
        lo, hi = min(lo, defaults[0]), max(hi, defaults[1])
    if hi <= lo:
        hi = lo + 1
    return lo, hi


def read_table(path):
    path = resolve_project_path(path)
    # Refuse known result-selected legacy exports, rather than pretending to undo selection.
    words = re.split(r"[^a-z]+", path.stem.lower())
    if (
        any(word in path.name.lower() for word in ("picksegments", "pickchannels"))
        or "filtered" in words
    ):
        raise ValueError(
            "Use the complete result table; this filename identifies a filtered legacy export"
        )
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={
            "subject_id": str,
            "subject": str,
            "dataset": str,
            "segment_id": str,
            "record_id": str,
        },
    )


def write_json(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {str(k): safe(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(v) for v in item]
        if isinstance(item, np.ndarray):
            return safe(item.tolist())
        if isinstance(item, np.generic):
            return safe(item.item())
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    Path(path).write_text(
        json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


class FigureOutput:
    def __init__(self, directory, formats=("svg", "png")):
        self.directory = resolve_project_path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.formats = tuple(formats)
        self.audit = {
            "sample_selection": "none",
            "right_topomap_panel": "Real EEG",
            "figures": [],
            "inputs": [],
        }

    def input(self, path):
        path = resolve_project_path(path)
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.audit["inputs"].append({"path": str(path), "sha256": digest.hexdigest()})

    def save(self, figure, name):
        import matplotlib.pyplot as plt

        for fmt in self.formats:
            figure.savefig(
                self.directory / f"{name}.{fmt}",
                dpi=300,
                facecolor="white",
                bbox_inches="tight",
            )
        plt.close(figure)
        self.audit["figures"].append(name)

    def table(self, frame, name):
        frame.to_csv(self.directory / name, index=False, encoding="utf-8-sig")

    def finish(self):
        write_json(self.directory / "plot_audit.json", self.audit)
        print(f"Figures: {self.directory}")


def panel_grid(n, height=4.6):
    import matplotlib.pyplot as plt
    from .style import cm

    columns = min(4, n)
    rows = int(np.ceil(n / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=cm(18.5, rows * height), squeeze=False
    )
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def validate_run_schema(directory):
    path = Path(directory) / "parameter_schema.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        for name, spec in param_config.items():
            if name not in saved or list(saved[name]["shape"]) != list(spec["shape"]):
                raise ValueError(
                    "Result parameter schema differs from current training configuration"
                )
        if set(saved) != set(param_config):
            raise ValueError(
                "Result parameter list differs from current training configuration"
            )
