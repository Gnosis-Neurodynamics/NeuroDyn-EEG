"""Layer 3: unselected subject-level regional parameter group comparisons."""

from pathlib import Path
import numpy as np
from validation.src.common import (
    parser,
    load_config,
    new_output,
    load_features,
    write_csv,
    write_json,
    feature_names,
    param_config,
    digest_file,
    atlas_labels,
)
from validation.src.group_analysis import subject_parameters, welch_statistics


def run(cfg, output=None, datasets=None):
    path = Path(cfg["paths"]["prepared_dir"]) / "features.npz"
    data = load_features(path, require_parameters=True)
    chosen = datasets or sorted(np.unique(data["dataset"]).tolist())
    if not chosen or not set(chosen) <= set(data["dataset"]):
        raise ValueError("Requested dataset is absent")
    out = new_output(cfg, "group_analysis", output)
    write_json(
        out / "input.json",
        {
            "features_sha256": digest_file(path),
            "path": str(path),
            "family_size": len(feature_names()),
            "positive_t": "case > control",
        },
    )
    q_cutoff = cfg["group_analysis"]["fdr_q"]
    if not 0 < q_cutoff < 1:
        raise ValueError("fdr_q must be between zero and one")
    rows, counts, subjects = [], [], []
    atlas = atlas_labels()
    for i, dataset in enumerate(chosen):
        features, labels, subject_rows = subject_parameters(data, dataset)
        subjects.extend(subject_rows)
        stats = welch_statistics(features, labels)
        np.savez_compressed(
            out / f"subject_parameters_{i:02d}.npz",
            features=features,
            label=labels,
            subject_id=np.asarray([r["subject_id"] for r in subject_rows]),
            feature_names=np.asarray(feature_names()),
        )
        for j, name in enumerate(feature_names()):
            parameter, node = name.rsplit(":", 1)
            global_parameter = tuple(param_config[parameter]["shape"]) == (1,)
            region = None if global_parameter else atlas[int(node)]
            rows.append(
                {
                    "dataset": dataset,
                    "parameter": parameter,
                    "region_index": -1 if global_parameter else int(node),
                    "aal_roi_number": "" if global_parameter else region["ROI number"],
                    "region_name": (
                        "Whole brain" if global_parameter else region["ROI name"]
                    ),
                    "atlas_name": "" if global_parameter else region["atlas_name"],
                    "scope": "global" if global_parameter else "regional",
                    "family_size": features.shape[1],
                    **{key: value[j].item() for key, value in stats.items()},
                    "significant": bool(stats["q"][j] < q_cutoff),
                }
            )
        for parameter, spec in param_config.items():
            if tuple(spec["shape"]) == (1,):
                continue
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["parameter"] == parameter
            ]
            counts.append(
                {
                    "dataset": dataset,
                    "parameter": parameter,
                    "n_regions": len(selected),
                    "n_significant": sum(r["significant"] for r in selected),
                    "n_case_higher": sum(
                        r["significant"] and r["t"] > 0 for r in selected
                    ),
                    "n_case_lower": sum(
                        r["significant"] and r["t"] < 0 for r in selected
                    ),
                    "q_cutoff": q_cutoff,
                }
            )
    write_csv(out / "all_tests.csv", rows)
    write_csv(
        out / "regional_counts.csv",
        sorted(
            counts, key=lambda r: (r["dataset"], -r["n_significant"], r["parameter"])
        ),
    )
    write_csv(out / "subjects.csv", subjects)
    print(f"Results: {out}")
    return out


if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--dataset", action="append")
    args = p.parse_args()
    run(load_config(args.config), args.output, args.dataset)
