"""Shared paths for sampling, simulation, training, and plotting.

All relative paths are resolved from this project directory, not the shell's
working directory. Absolute paths can be used for data on another disk.
"""

from copy import deepcopy
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHUNK_DIR = PROJECT_ROOT / "TrainData" / "NeuroDynV1"
CHUNK_FILENAME_TEMPLATE = "chunk_{i}_Param14.npz"
CHUNK_FILE_GLOB = "chunk_*_Param14.npz"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "TrainWorks" / "config" / "neurodyneeg.yaml"
DEFAULT_LOG_DIR = PROJECT_ROOT / "TrainWorks" / "logs"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "TrainWorks" / "checkpoints"
DEFAULT_VALIDATION_DIR = PROJECT_ROOT / "validation_data"
DEFAULT_REAL_DIR = PROJECT_ROOT / "RealData"
DEFAULT_VALIDATION_CONFIG = PROJECT_ROOT / "validation" / "config" / "validation.json"


def resolve_project_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def resource_path(filename):
    directory = os.environ.get("NEURODYNEEG_RESOURCE_DIR", "utils")
    return resolve_project_path(directory) / filename


def require_resource(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Simulator resource not found: {path}. Place the original assets in "
            "<project>/utils or set NEURODYNEEG_RESOURCE_DIR to their directory."
        )
    return path


def resolve_training_paths(config):
    config = deepcopy(config)
    data = config.setdefault("data", {})
    data["chunk_dir"] = str(
        resolve_project_path(data.get("chunk_dir", DEFAULT_CHUNK_DIR))
    )
    data.setdefault("file_glob", CHUNK_FILE_GLOB)
    logging = config.setdefault("logging", {})
    for key, default in (("log_dir", DEFAULT_LOG_DIR), ("ckpt_dir", DEFAULT_CKPT_DIR)):
        logging[key] = str(resolve_project_path(logging.get(key, default)))
    training = config.setdefault("train", {})
    if training.get("train_stats_path"):
        training["train_stats_path"] = str(
            resolve_project_path(training["train_stats_path"])
        )
    return config


def find_chunk_files(chunk_dir=DEFAULT_CHUNK_DIR, file_glob=CHUNK_FILE_GLOB):
    directory = resolve_project_path(chunk_dir)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Chunk directory not found: {directory}. Run "
            "python -m paramsampling.sample_main first, or select the same data directory."
        )
    files = sorted(path for path in directory.glob(file_glob) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No parameter chunks matching {file_glob!r} in {directory}"
        )
    return [str(path) for path in files]
