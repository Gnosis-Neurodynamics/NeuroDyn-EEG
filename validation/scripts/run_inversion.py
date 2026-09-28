"""Layer 1: synthetic recovery, noise robustness, and real-EEG closed loop."""

import json
from pathlib import Path
import numpy as np
from validation.src.common import (
    parser,
    load_config,
    new_output,
    write_csv,
    write_json,
    param_config,
    seed_for,
    digest_file,
)
from validation.src.metrics import ParameterMetrics, closed_loop_metrics


def run_synthetic(cfg, out, noise=False):
    from validation.src.inference import Predictor
    from validation.src.noise import inject_noise
    from TrainWorks.src.data import (
        get_lead_field,
        project_source_to_scalp,
        centre_eeg,
        amp_mask,
    )
    from paramsampling.chunk_io import inspect_simulated_chunk, validate_params
    from project_paths import find_chunk_files

    files = find_chunk_files(
        cfg["paths"]["simulation_dir"], cfg["paths"]["simulation_glob"]
    )
    predictor, lead = Predictor(cfg), get_lead_field()
    conditions = [("clean", None)]
    if noise:
        conditions += [
            (family, snr)
            for family in cfg["noise"]["families"]
            for snr in cfg["noise"]["snr_db"]
        ]
    stats = {
        (family, snr): ParameterMetrics(param_config) for family, snr in conditions
    }
    audits = []
    for i, path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {path}", flush=True)
        inspect_simulated_chunk(path)
        with np.load(path, allow_pickle=True) as data:
            source, params = data["SimData"], data["params"].item()
        validate_params(params, param_config, n_samples=len(source))
        if not np.isfinite(source).all():
            raise ValueError(f"Nonfinite simulation: {path}")
        eeg = centre_eeg(project_source_to_scalp(source, lead))
        selection = cfg["simulation"]
        if selection["qc"] == "training":
            keep = amp_mask(eeg, lo=selection["amp_lo"], hi=selection["amp_hi"])
        elif selection["qc"] == "none":
            keep = np.ones(len(eeg), dtype=bool)
        else:
            raise ValueError("simulation.qc must be training or none")
        relative = Path(path).relative_to(cfg["paths"]["simulation_dir"]).as_posix()
        ids = np.asarray([f"{relative}:{index}" for index in np.flatnonzero(keep)])
        audits.append(
            {
                "file": path,
                "n_input": len(eeg),
                "n_retained": int(keep.sum()),
                "excluded_sample_indices": np.flatnonzero(~keep).tolist(),
                "qc": selection,
            }
        )
        write_json(out / "synthetic_inclusion.json", audits)
        if not keep.any():
            continue
        eeg = eeg[keep]
        params = {key: np.asarray(params[key])[keep] for key in param_config}
        for family, snr in conditions:
            batch = (
                eeg
                if family == "clean"
                else inject_noise(eeg, ids, family, snr, cfg["noise"])
            )
            prediction = predictor(batch)
            stats[family, snr].update(params, prediction)
            if family == "clean":
                np.savez_compressed(
                    out / f"recovery_{i:06d}.npz",
                    sample_id=ids,
                    **{f"true_{key}": params[key] for key in param_config},
                    **{f"pred_{key}": prediction[key] for key in param_config},
                )
    write_csv(
        out / "parameter_metrics.csv",
        (
            row
            for (family, snr), stat in stats.items()
            for row in stat.rows(noise=family, snr_db=snr)
        ),
    )
    write_json(
        out / "model.json",
        {
            "checkpoint_sha256": predictor.checkpoint_hash,
            "normalise": predictor.normalise,
        },
    )


def run_closed_loop(cfg, out):
    from paramsampling.simulate_signal import ParamtoJRSim
    from TrainWorks.src.data import get_lead_field, project_source_to_scalp, centre_eeg
    from project_paths import PROJECT_ROOT

    prepared = Path(cfg["paths"]["prepared_dir"])
    index = json.loads((prepared / "index.json").read_text(encoding="utf-8"))
    if not index.get("complete") or not index.get("records"):
        raise ValueError("Incomplete prepared data; run prepare_real_data first")
    burn = cfg["closed_loop"]["burn_in_seconds"]
    if burn < 0 or int(burn * 256) != burn * 256:
        raise ValueError("Burn-in must be nonnegative and align with sampling grid")
    simulator = ParamtoJRSim(
        T=5 + burn, pick_channels=cfg["real"]["channels"], source_sfreq=256
    )
    lead, all_rows = get_lead_field(), []
    for record in index["records"]:
        print(
            f"Closed loop: {record['dataset']}/{record['subject_id']}/{record['record_id']}",
            flush=True,
        )
        with np.load(prepared / record["file"], allow_pickle=False) as data:
            observed, ids = data["eeg"], data["segment_id"]
            predictions = {key: data[key] for key in param_config}
        source = []
        for i, trial_id in enumerate(ids):
            np.random.seed(seed_for(cfg["closed_loop"]["seed"], str(trial_id)))
            params = {key: value[i] for key, value in predictions.items()}
            simulated = simulator.fit_Single(params)
            source.append(simulated[:, int(burn * 256) :])
        reconstructed = centre_eeg(project_source_to_scalp(np.stack(source), lead))
        metrics = closed_loop_metrics(observed, reconstructed)
        np.savez_compressed(
            out / record["file"], segment_id=ids, reconstructed=reconstructed, **metrics
        )
        for i, trial_id in enumerate(ids):
            for ch, name in enumerate(cfg["real"]["channels"]):
                all_rows.append(
                    {
                        "dataset": record["dataset"],
                        "subject_id": record["subject_id"],
                        "segment_id": str(trial_id),
                        "channel": name,
                        **{key: float(value[i, ch]) for key, value in metrics.items()},
                    }
                )
    write_csv(out / "closed_loop_segments.csv", all_rows)
    # Average segments within each subject first, then weight every subject equally.
    names = list(metrics)
    groups = {}
    for row in all_rows:
        key = (row["dataset"], row["subject_id"], row["channel"])
        groups.setdefault(key, []).append(row)
    subject_rows = [
        {
            "dataset": key[0],
            "subject_id": key[1],
            "channel": key[2],
            "n_segments": len(rows),
            **{name: float(np.mean([row[name] for row in rows])) for name in names},
        }
        for key, rows in groups.items()
    ]
    write_csv(out / "closed_loop_subjects.csv", subject_rows)
    summary = []
    for dataset in [*sorted({row["dataset"] for row in subject_rows}), "ALL"]:
        for channel in cfg["real"]["channels"]:
            rows = [
                r
                for r in subject_rows
                if r["channel"] == channel
                and (dataset == "ALL" or r["dataset"] == dataset)
            ]
            for name in names:
                values = np.asarray([row[name] for row in rows])
                summary.append(
                    {
                        "dataset": dataset,
                        "channel": channel,
                        "metric": name,
                        "n_subjects": len(values),
                        "n_finite": int(np.isfinite(values).sum()),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                    }
                )
    write_csv(out / "closed_loop_summary.csv", summary)
    write_json(
        out / "forward_provenance.json",
        {
            "prepared_index": index,
            "burn_in_seconds": burn,
            "Forward_sha256": digest_file(PROJECT_ROOT / "paramsampling/Forward.py"),
            "JR_model_sha256": digest_file(PROJECT_ROOT / "paramsampling/JR_model.py"),
            "note": "Uses the same forward implementation as sampling; see README for unresolved paper/simulator differences.",
        },
    )


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument(
        "--experiment",
        choices=["recovery", "noise", "closed_loop", "all"],
        default="recovery",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    out = new_output(cfg, args.experiment, args.output)
    if args.experiment in ("recovery", "noise", "all"):
        run_synthetic(cfg, out, noise=args.experiment != "recovery")
    if args.experiment in ("closed_loop", "all"):
        run_closed_loop(cfg, out)
    print(f"Results: {out}")
