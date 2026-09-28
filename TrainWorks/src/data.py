from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config_params import param_config
from paramsampling.chunk_io import inspect_simulated_chunk, validate_params
from project_paths import CHUNK_FILE_GLOB, find_chunk_files

PP_ALL = list(param_config.keys())


def get_lead_field(scale=1e-3):
    """Returns numpy array (19, 90) ready to dot-product source signals."""
    from paramsampling.Forward import read_matched_source

    sol = read_matched_source()
    return sol * scale


def project_source_to_scalp(src, lead_field):
    """src: (S, 90, T) -> (S, 19, T)"""
    return np.einsum("ij,sjt->sit", lead_field, src)


def centre_eeg(eeg):
    """Two-step de-mean as in paper §2.6."""
    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    eeg = eeg - eeg.mean(axis=-2, keepdims=True)
    return eeg


def amp_mask(eeg, fs=256, lo=3.0, hi=100.0):
    """Returns boolean mask ``(S,)``: True if avg abs-amp in [lo, hi] uV."""
    seg = eeg[..., int(fs) :]
    amp = np.abs(seg - seg.mean(axis=-1, keepdims=True)).mean(axis=(-1, -2))
    return (amp >= lo) & (amp <= hi)


def zscore_eeg(eeg, eps: float = 1e-6):
    """Per-sample (per (channel,time)-window) zero-mean / unit-std."""
    mu = eeg.mean(axis=(-1, -2), keepdims=True)
    sd = eeg.std(axis=(-1, -2), keepdims=True)
    return (eeg - mu) / np.maximum(sd, eps)


class ChunkDataset(Dataset):
    """Backed by a single chunk file held in RAM."""

    def __init__(self, eeg_scalp, params, dtype=np.float32):
        self.eeg = eeg_scalp.astype(dtype)
        self.params = {k: v.astype(dtype) for k, v in params.items()}
        self.n = self.eeg.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.from_numpy(self.eeg[idx])
        y = {k: torch.from_numpy(v[idx]) for k, v in self.params.items()}
        return x, y


def load_chunk_processed(
    path, lead_field, amp_lo=3.0, amp_hi=100.0, normalise: Optional[str] = None
):
    """Load one chunk -> scalp EEG + params, after preprocessing.

    Args:
        normalise: None | 'zscore' (per-sample z-score after de-mean & filter).
    """
    inspect_simulated_chunk(path, n_nodes=lead_field.shape[1])
    with np.load(path, allow_pickle=True) as data:
        src = data["SimData"]
        params = data["params"].item()
    validate_params(params, param_config, n_samples=src.shape[0])
    params = {key: np.asarray(params[key]) for key in param_config}

    eeg = project_source_to_scalp(src, lead_field)
    eeg = centre_eeg(eeg)

    mask = amp_mask(eeg, lo=amp_lo, hi=amp_hi)
    if mask.sum() == 0:
        return None
    eeg = eeg[mask]
    params = {k: v[mask] for k, v in params.items()}

    if normalise == "zscore":
        eeg = zscore_eeg(eeg)
    return eeg, params


class TrainChunkIterator:
    """Yields ChunkDataset one file at a time. Cheap on RAM."""

    def __init__(
        self,
        chunk_dir,
        lead_field,
        file_glob=CHUNK_FILE_GLOB,
        amp_lo=3.0,
        amp_hi=100.0,
        shuffle=True,
        seed=0,
        normalise: Optional[str] = None,
    ):
        self.files = find_chunk_files(chunk_dir, file_glob)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.files)
        self.lead_field = lead_field
        self.amp_lo, self.amp_hi = amp_lo, amp_hi
        self.normalise = normalise
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        for path in self.files:
            res = load_chunk_processed(
                path,
                self.lead_field,
                self.amp_lo,
                self.amp_hi,
                normalise=self.normalise,
            )
            if res is None:
                continue
            eeg, params = res
            yield path, ChunkDataset(eeg, params)
