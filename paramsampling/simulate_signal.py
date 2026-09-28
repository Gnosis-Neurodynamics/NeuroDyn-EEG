"""Add SimData to sampled chunks, in the same directory consumed by training."""

import argparse
from pathlib import Path

import numpy as np

from project_paths import (
    DEFAULT_CHUNK_DIR,
    CHUNK_FILE_GLOB,
    find_chunk_files,
    resolve_project_path,
)
from .param_specs import PARAM_SPECS
from .chunk_io import load_params, inspect_simulated_chunk, write_simulated_chunk

PP = list(PARAM_SPECS)


class ParamtoJRSim(object):
    def __init__(
        self, T, pick_channels=None, source_sfreq=256, eeg_filter=[1, 40], eeg_freq=256
    ):
        from .Forward import GenerateSimuSignal

        if pick_channels is None:
            from eegaux.montage import STANDARD_1020

            pick_channels = STANDARD_1020
        self.JR_Simulator = GenerateSimuSignal(
            source_sfreq=source_sfreq,
            T=T,
            eeg_sfreq=eeg_freq,
            eeg_filter=eeg_filter,
            pick_channels=pick_channels,
        )

    def fit_Single(self, params_dict):
        jr_out_ = self.JR_Simulator.fit_singleband_JRnetwork(**params_dict)
        return jr_out_[:, 1:]


def transform_sample_params(path):
    params = load_params(path, PARAM_SPECS)
    res = [{p: params[p][i] for p in PP} for i in range(params[PP[0]].shape[0])]
    return res, params


def get_sampled_data(path, n_jobs=10):
    params_dict, sp = transform_sample_params(path)
    SimModel = ParamtoJRSim(T=6, source_sfreq=256, eeg_filter=[1, 40], eeg_freq=256)
    if n_jobs == 1:
        out = [SimModel.fit_Single(params) for params in params_dict]
    else:
        from joblib import Parallel, delayed

        out = Parallel(n_jobs=n_jobs, verbose=False)(
            delayed(SimModel.fit_Single)(p) for p in params_dict
        )
    out = np.array(out)[..., 256:]
    expected = (len(params_dict), 90, 1280)
    if out.shape != expected or not np.isfinite(out).all():
        raise ValueError(
            f"{path}: expected finite SimData {expected}, got {out.shape}; chunk not overwritten"
        )
    write_simulated_chunk(path, out, sp)
    return out


def check_exist_chunks(path):
    if not Path(path).is_file():
        return False
    try:
        inspect_simulated_chunk(path)
        return True
    except ValueError:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk_dir",
        type=resolve_project_path,
        default=DEFAULT_CHUNK_DIR,
        help="Same directory as sample_main --out_dir; relative to the project",
    )
    parser.add_argument("--file_glob", default=CHUNK_FILE_GLOB)
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate chunks that already contain SimData",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_jobs == 0:
        raise SystemExit("n_jobs must not be zero")
    files = find_chunk_files(args.chunk_dir, args.file_glob)
    print(f"[simulate_signal] chunk_dir = {args.chunk_dir}; {len(files)} chunks")
    for f in files:
        print(f"Checking {f} ...")
        if not args.overwrite and check_exist_chunks(f):
            print(f"Already exists SimData in {f}, skip.")
            continue

        print(f"Processing {f} ...")
        out = get_sampled_data(f, n_jobs=args.n_jobs)
        print(out.shape)


if __name__ == "__main__":
    main()
