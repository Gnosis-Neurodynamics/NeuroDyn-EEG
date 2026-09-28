"""Layer 2: train this project's classification head; no external backbones required."""

from pathlib import Path
import numpy as np
from project_paths import resolve_project_path
from validation.src.common import (
    parser,
    load_config,
    new_output,
    load_features,
    write_json,
    write_csv,
    digest_file,
)
from validation.src.downstream import subject_table, make_splits, train_fold


def run(cfg, output=None, features=None, datasets=None, plan_only=False):
    own_path = Path(cfg["paths"]["prepared_dir"]) / "features.npz"
    path = resolve_project_path(features) if features else own_path
    data = load_features(path, require_parameters=features is None)
    reference = load_features(own_path, require_parameters=True) if features else data
    chosen = datasets or sorted(np.unique(data["dataset"]).tolist())
    if not chosen or not set(chosen) <= set(data["dataset"]):
        raise ValueError("Requested dataset is absent from feature file")
    options = cfg["downstream"]
    if len(options["seeds"]) != 2 or len(set(options["seeds"])) != 2:
        raise ValueError("Paper evaluation requires two distinct seeds")
    out = new_output(cfg, "downstream", output)
    write_json(
        out / "features.json",
        {
            "path": str(path),
            "sha256": digest_file(path),
            "kind": (
                "external_cached_features" if features else "NeuroDynEEG_parameters"
            ),
            "head": "BatchNorm -> Linear -> ReLU -> Dropout -> Linear",
            "note": "External feature interface does not implement third-party backbones or their official heads.",
        },
    )
    results = []
    for dataset_number, dataset in enumerate(chosen):
        subjects = subject_table(data, dataset)
        if subjects != subject_table(reference, dataset):
            raise ValueError(
                f"{dataset}: external features must use the same subject IDs, labels and official splits as prepared data"
            )
        for seed in options["seeds"]:
            for split in make_splits(subjects, dataset, seed, options["folds"]):
                fold_out = (
                    out
                    / f"dataset_{dataset_number:02d}"
                    / f"seed_{seed}_fold_{split['fold']}"
                )
                fold_out.mkdir(parents=True)
                write_json(fold_out / "split.json", {"dataset": dataset, **split})
                if not plan_only:
                    results.append(train_fold(data, dataset, split, options, fold_out))
                    write_csv(out / "runs.csv", results)
    if results:
        summary = []
        for dataset in chosen:
            rows = [row for row in results if row["dataset"] == dataset]
            for metric in ("bacc", "auroc", "aucpr"):
                values = [row[metric] for row in rows]
                summary.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "n_runs": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)),
                    }
                )
        write_csv(out / "summary.csv", summary)
    print(f"{'Split plan' if plan_only else 'Results'}: {out}")
    return out


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument(
        "--features",
        help="Optional user-prepared NPZ features; no third-party model code is imported",
    )
    p.add_argument(
        "--dataset",
        action="append",
        help="Repeat for multiple datasets; defaults to all",
    )
    p.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate data and export subject splits without torch training",
    )
    args = p.parse_args()
    run(
        load_config(args.config),
        args.output,
        args.features,
        args.dataset,
        args.plan_only,
    )
