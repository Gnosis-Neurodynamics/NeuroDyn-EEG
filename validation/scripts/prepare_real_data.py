"""Preprocess all manifest recordings and infer shared features for all three layers."""

from pathlib import Path
import numpy as np
from validation.src.common import (
    parser,
    load_config,
    new_output,
    write_json,
    write_csv,
    digest_file,
    flatten_parameters,
    feature_names,
)
from validation.src.real_data import read_manifest, prepare_recording, segment_id


def run(cfg, output=None):
    from validation.src.inference import Predictor

    rows = read_manifest(cfg)
    destination = output or cfg["paths"]["prepared_dir"]
    out = new_output(cfg, "prepared", destination)
    predictor = Predictor(cfg)
    tables = {
        key: []
        for key in (
            "features",
            "dataset",
            "subject_id",
            "label",
            "segment_id",
            "official_split",
        )
    }
    audits, records, qc_rows = [], [], []
    for i, row in enumerate(rows):
        print(
            f"[{i+1}/{len(rows)}] {row['dataset']}/{row['subject_id']}/{row['record_id']}",
            flush=True,
        )
        eeg, indices, qc, details = prepare_recording(row, cfg)
        details["source_sha256"] = digest_file(row["path"])
        audits.append(details)
        qc_rows.extend(qc)
        # Persist exclusions even if a later recording or empty cohort stops the run.
        write_json(out / "recordings.json", audits)
        write_csv(out / "segment_qc.csv", qc_rows)
        if not len(eeg):
            continue
        params = predictor(eeg)
        ids = np.asarray([segment_id(row, index) for index in indices])
        record_file = f"record_{i:06d}.npz"
        np.savez_compressed(out / record_file, eeg=eeg, segment_id=ids, **params)
        records.append(
            {
                "file": record_file,
                "dataset": row["dataset"],
                "subject_id": row["subject_id"],
                "record_id": row["record_id"],
                "n_segments": len(eeg),
            }
        )
        tables["features"].append(flatten_parameters(params))
        tables["segment_id"].extend(ids.tolist())
        for key in ("dataset", "subject_id", "label", "official_split"):
            tables[key].extend([row[key]] * len(eeg))
    expected = {(r["dataset"], r["subject_id"]) for r in rows}
    actual = set(zip(tables["dataset"], tables["subject_id"]))
    if expected != actual:
        raise ValueError(
            f"Subjects with no usable EEG: {sorted(expected-actual)}. Inspect segment_qc.csv; no silent subject removal."
        )
    tables["features"] = np.concatenate(tables["features"]).astype(np.float32)
    np.savez_compressed(
        out / "features.npz", feature_names=np.asarray(feature_names()), **tables
    )
    write_json(
        out / "index.json",
        {
            "complete": True,
            "checkpoint_sha256": predictor.checkpoint_hash,
            "manifest_sha256": digest_file(cfg["paths"]["manifest"]),
            "normalise": predictor.normalise,
            "records": records,
            "n_subjects": len(actual),
            "n_segments": len(tables["label"]),
        },
    )
    print(f"Prepared shared data: {out}")
    return out


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    run(load_config(args.config), args.output)
