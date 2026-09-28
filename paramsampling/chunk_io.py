"""The NPZ contract shared by parameter sampling, simulation, and training."""

from pathlib import Path
import os
import tempfile
import zipfile

import numpy as np
from project_paths import find_chunk_files


def validate_params(params, specs, n_samples=None):
    if not isinstance(params, dict):
        raise ValueError("'params' must contain a parameter dictionary")
    missing = [key for key in specs if key not in params]
    if missing:
        raise ValueError(
            f"Missing configured training/simulation parameters: {missing}"
        )
    for key, spec in specs.items():
        values = np.asarray(params[key])
        expected_shape = tuple(spec["shape"])
        if values.ndim != len(expected_shape) + 1 or values.shape[1:] != expected_shape:
            raise ValueError(
                f"{key}: expected (samples, {expected_shape}), got {values.shape}"
            )
        if n_samples is None:
            n_samples = values.shape[0]
        if values.shape[0] != n_samples or n_samples <= 0:
            raise ValueError(
                f"{key}: inconsistent or empty sample dimension {values.shape[0]}"
            )
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{key}: parameters must be finite numeric values")
    return n_samples


def load_params(path, specs):
    with np.load(path, allow_pickle=True) as data:
        if "params" not in data:
            raise ValueError(
                f"{path}: missing 'params'; expected a sampler-generated chunk"
            )
        params = data["params"].item()
    validate_params(params, specs)
    return params


def inspect_simulated_chunk(path, n_nodes=90, n_times=1280):
    """Check keys and the signal header without loading a potentially GB-sized array."""
    try:
        with zipfile.ZipFile(path) as archive:
            if (
                "params.npy" not in archive.namelist()
                or "SimData.npy" not in archive.namelist()
            ):
                raise ValueError(
                    "missing 'params' or 'SimData'. Parameter sampling alone is not training "
                    "data; run python -m paramsampling.simulate_signal on this directory first"
                )
            with archive.open("SimData.npy") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unsupported SimData NPY version: {version}")
        if len(shape) != 3 or shape[0] <= 0 or shape[1:] != (n_nodes, n_times):
            raise ValueError(
                f"expected SimData (samples, {n_nodes}, {n_times}), got {shape}"
            )
        if not np.issubdtype(dtype, np.number):
            raise ValueError(f"SimData must have a numeric dtype, got {dtype}")
        return shape
    except (ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Invalid simulation chunk {path}: {exc}") from exc


def write_simulated_chunk(path, signals, params):
    """Replace a parameter chunk only after the completed NPZ has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream, SimData=signals, params=np.asarray(params, dtype=object)
            )
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def split_simulated_chunks(chunk_dir, file_glob, val_chunks):
    files = find_chunk_files(chunk_dir, file_glob)
    for path in files:
        inspect_simulated_chunk(path)
    if val_chunks < 0 or val_chunks >= len(files):
        raise ValueError(
            f"data.val_chunks={val_chunks}, but only {len(files)} chunks exist. "
            "Reserve fewer validation chunks so at least one training chunk remains; "
            "use 0 to explicitly disable validation."
        )
    if val_chunks == 0:
        return files, []
    return files[:-val_chunks], files[-val_chunks:]
