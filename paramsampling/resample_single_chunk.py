from __future__ import annotations

import argparse
import os

import numpy as np

from project_paths import (
    DEFAULT_CHUNK_DIR,
    CHUNK_FILENAME_TEMPLATE,
    resolve_project_path,
)


def parse_args():
    p = argparse.ArgumentParser(description="Generate a single chunk_*_Param14.npz")
    p.add_argument(
        "--out_dir",
        type=resolve_project_path,
        default=DEFAULT_CHUNK_DIR,
        help="Same project-relative or absolute output directory as sample_main",
    )
    p.add_argument(
        "--chunk_id",
        type=int,
        default=234,
        help="Chunk index i, filename chunk_<i>_Param14.npz",
    )
    p.add_argument(
        "--chunk_size", type=int, default=1000, help="Number of samples in this chunk"
    )
    p.add_argument("--n_nodes", type=int, default=90)
    p.add_argument(
        "--seed",
        type=int,
        default=20260503,
        help="Must match sample_main's --seed for reproducibility",
    )
    p.add_argument("--no_tau_coupling", action="store_true")
    p.add_argument("--filename_template", type=str, default=CHUNK_FILENAME_TEMPLATE)
    p.add_argument(
        "--dtype", type=str, choices=["float32", "float64"], default="float64"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite if the target already exists",
    )
    return p.parse_args()


def make_seed(master_seed: int, chunk_id: int) -> int:
    ss = np.random.SeedSequence(master_seed).spawn(1)[0].spawn(chunk_id + 1)[-1]
    return int(ss.entropy ^ ss.pool[0])


def write_chunk(path: str, params: dict) -> None:
    np.savez_compressed(path, params=np.asarray(params, dtype=object))


def main():
    args = parse_args()
    from .samplers import sample_chunk

    os.makedirs(args.out_dir, exist_ok=True)
    fname = args.filename_template.format(i=args.chunk_id)
    fpath = os.path.join(args.out_dir, fname)

    if os.path.exists(fpath) and not args.overwrite:
        raise SystemExit(f"existsed: {fpath}")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    sub_seed = make_seed(args.seed, args.chunk_id)
    rng = np.random.default_rng(sub_seed)

    params = sample_chunk(
        n_samples=args.chunk_size,
        n_nodes=args.n_nodes,
        rng=rng,
        enforce_tau_coupling=not args.no_tau_coupling,
        dtype=dtype,
    )
    write_chunk(fpath, params)

    print(f"[resample_single_chunk] wrote {fpath}")
    print(
        f"[resample_single_chunk] chunk_id={args.chunk_id} size={args.chunk_size} sub_seed={sub_seed}"
    )


if __name__ == "__main__":
    main()
