from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np

from .param_specs import PARAM_SPECS
from project_paths import (
    DEFAULT_CHUNK_DIR,
    CHUNK_FILENAME_TEMPLATE,
    resolve_project_path,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=resolve_project_path,
        default=DEFAULT_CHUNK_DIR,
        help="Output directory (default: <project>/TrainData/NeuroDynV1). "
        "Relative paths are project-relative; created automatically.",
    )
    p.add_argument(
        "--n_total",
        type=int,
        default=1_000_000,
        help="Total number of samples; default 1M.",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of samples per chunk; default 1000 (consistent with existing data).",
    )
    p.add_argument(
        "--n_nodes",
        type=int,
        default=90,
        help="Number of nodes per spatial parameter; default 90.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260922,
        help="Master seed; each chunk derives an independent sub-seed as seed + chunk_id, "
        "ensuring reproducibility and support for resumable sampling.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite existing chunk files (default: skip).",
    )
    p.add_argument(
        "--no_tau_coupling",
        action="store_true",
        help="Disable the tau_e -> tau_i physiological coupling constraint (default: enabled).",
    )
    p.add_argument(
        "--filename_template",
        type=str,
        default=CHUNK_FILENAME_TEMPLATE,
        help="Filename template; {i} is the placeholder for the chunk id.",
    )
    p.add_argument("--manifest_name", type=str, default="_sampling_manifest.json")
    p.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Output dtype. Existing chunks use float64, so the default follows that.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print progress / summary, not each individual chunk.",
    )
    return p.parse_args()


def make_seed(master_seed: int, chunk_id: int) -> int:
    ss = np.random.SeedSequence(master_seed).spawn(1)[0].spawn(chunk_id + 1)[-1]
    return int(ss.entropy ^ ss.pool[0])


def write_chunk(path: str, params: Dict[str, np.ndarray]):
    np.savez_compressed(path, params=np.asarray(params, dtype=object))


def write_manifest(path: str, args, chunks_done):
    spec_snapshot = {
        k: {
            "bounds": list(map(float, s["bounds"])),
            "shape": list(s["shape"]),
            "distribution": s.get("distribution"),
            "subject_std_frac": s.get("subject_std_frac"),
            "node_std_frac": s.get("node_std_frac"),
            "coupled_with": s.get("coupled_with"),
        }
        for k, s in PARAM_SPECS.items()
    }
    manifest = {
        "out_dir": str(args.out_dir),
        "n_total": args.n_total,
        "chunk_size": args.chunk_size,
        "n_chunks": (args.n_total + args.chunk_size - 1) // args.chunk_size,
        "n_nodes": args.n_nodes,
        "master_seed": args.seed,
        "tau_coupling": not args.no_tau_coupling,
        "filename_template": args.filename_template,
        "dtype": args.dtype,
        "spec": spec_snapshot,
        "chunks_done": sorted(chunks_done),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    if args.n_total <= 0 or args.chunk_size <= 0 or args.n_nodes <= 0:
        raise SystemExit("n_total, chunk_size and n_nodes must be positive")
    from .samplers import sample_chunk

    os.makedirs(args.out_dir, exist_ok=True)

    n_chunks = (args.n_total + args.chunk_size - 1) // args.chunk_size
    last_chunk_size = args.n_total - (n_chunks - 1) * args.chunk_size

    print(f"[sample_main] target = {args.n_total} samples in {n_chunks} chunks")
    print(f"[sample_main] out_dir = {args.out_dir}")
    print(
        f"[sample_main] master_seed = {args.seed}, "
        f"tau_coupling = {not args.no_tau_coupling}, dtype = {args.dtype}"
    )

    dtype = np.float32 if args.dtype == "float32" else np.float64

    chunks_done = []
    skipped = 0
    t_start = time.time()

    for i in range(n_chunks):
        size = args.chunk_size if i < n_chunks - 1 else last_chunk_size
        fname = args.filename_template.format(i=i)
        fpath = os.path.join(args.out_dir, fname)

        if os.path.exists(fpath) and not args.overwrite:
            skipped += 1
            chunks_done.append(i)
            if not args.quiet:
                print(f"  [skip] chunk {i:5d}: {fpath} exists")
            continue

        sub_seed = make_seed(args.seed, i)
        rng = np.random.default_rng(sub_seed)

        params = sample_chunk(
            n_samples=size,
            n_nodes=args.n_nodes,
            rng=rng,
            enforce_tau_coupling=not args.no_tau_coupling,
            dtype=dtype,
        )
        write_chunk(fpath, params)
        chunks_done.append(i)

        if not args.quiet:
            elapsed = time.time() - t_start
            done = i + 1 - skipped
            rate = done / max(elapsed, 1e-6)
            eta = (n_chunks - 1 - i) / max(rate, 1e-6)
            print(
                f"  [done] chunk {i:5d}/{n_chunks-1}  "
                f"size={size}  seed={sub_seed}  "
                f"({rate:.1f} chunk/s, ETA {eta/60:.1f} min)"
            )
        elif (i + 1) % 50 == 0 or i + 1 == n_chunks:
            elapsed = time.time() - t_start
            print(f"  ... {i+1}/{n_chunks} chunks done in {elapsed/60:.1f} min")

    write_manifest(os.path.join(args.out_dir, args.manifest_name), args, chunks_done)

    elapsed = time.time() - t_start
    print(
        f"\n[sample_main] DONE in {elapsed/60:.2f} min  "
        f"(generated {len(chunks_done) - skipped}, skipped {skipped})"
    )
    print(
        f"[sample_main] manifest: " f"{os.path.join(args.out_dir, args.manifest_name)}"
    )


if __name__ == "__main__":
    main()
