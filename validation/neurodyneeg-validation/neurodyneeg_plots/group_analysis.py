"""AAL plots migrated from altas_worker/plot_altas.py, driven by all_tests.csv."""

import numpy as np
import pandas as pd
from project_paths import resolve_project_path
from .common import (
    param_config,
    PACKAGE_ROOT,
    read_table,
    require_columns,
    numeric,
    require_parameters,
    validate_run_schema,
)
from .style import PALETTE, cm


def load_group_results(path, cutoff):
    if not 0 < cutoff < 1:
        raise ValueError("q threshold must be between zero and one")
    table = read_table(path)
    require_columns(
        table,
        ["dataset", "parameter", "scope", "region_index", "t", "p", "q", "family_size"],
    )
    if table["dataset"].isna().any() or table["dataset"].str.strip().eq("").any():
        raise ValueError("Missing dataset identity; no group tests may be discarded")
    numeric(table, ["t", "p", "q"], allow_nan=True)
    numeric(table, ["region_index", "family_size"])
    expected_size = sum(int(np.prod(spec["shape"])) for spec in param_config.values())
    for dataset, rows in table.groupby("dataset"):
        require_parameters(rows)
        if rows.duplicated(["parameter", "region_index"]).any():
            raise ValueError(f"Duplicate group tests: {dataset}")
        if (
            len(rows) != expected_size
            or not (rows["family_size"] == expected_size).all()
        ):
            raise ValueError(
                "Incomplete hypothesis family; use all_tests.csv, including global delay"
            )
        for parameter, spec in param_config.items():
            sub = rows[rows["parameter"] == parameter]
            global_parameter = tuple(spec["shape"]) == (1,)
            expected = (
                {-1} if global_parameter else set(range(int(np.prod(spec["shape"]))))
            )
            if set(sub["region_index"]) != expected:
                raise ValueError(f"Missing/extra group test regions: {parameter}")
            if set(sub["scope"]) != ({"global"} if global_parameter else {"regional"}):
                raise ValueError(f"Incorrect global/regional scope: {parameter}")
    if ((table["q"] < 0) | (table["q"] > 1)).any():
        raise ValueError("Invalid BH q values")
    table["significant_for_plot"] = table["q"] < cutoff
    return table


def map_atlas_values(atlas_data, atlas_labels, values):
    if len(values) != len(atlas_labels) or len(set(atlas_labels)) != len(atlas_labels):
        raise ValueError("Atlas label/value mismatch")
    if not np.isfinite(values).all():
        raise ValueError(
            "Undefined region statistics cannot be rendered as nonsignificant brain regions"
        )
    present = set(np.unique(atlas_data).tolist())
    if not set(atlas_labels) <= present:
        raise ValueError(
            "Atlas volume does not contain all AAL labels from mapping CSV"
        )
    result = np.zeros(atlas_data.shape, dtype=np.float32)
    for code, value in zip(atlas_labels, values):
        result[atlas_data == code] = value
    return result


def draw_brain_map(rows, atlas, labels, out, name, title):
    import nibabel as nib
    from nilearn import plotting
    import matplotlib.pyplot as plt

    ordered = rows.set_index("region_index").loc[list(range(len(labels)))]
    if ordered["q"].isna().any():
        raise ValueError(
            "Selected brain-map parameter has undefined q values; inspect the full table instead of suppressing them"
        )
    image = nib.load(atlas)
    values = map_atlas_values(
        image.get_fdata(),
        labels["atlas_mni_id"].to_numpy(),
        ordered["significant_for_plot"].to_numpy(dtype=float),
    )
    image = nib.Nifti1Image(values, image.affine)
    fig = plt.figure(figsize=cm(20, 5))
    # Same AAL-to-NIfTI mapping, z-slices and RdBu_r palette as old plot_stat_map.
    display = plotting.plot_stat_map(
        image,
        display_mode="z",
        cut_coords=list(range(-40, 70, 25)),
        threshold=0.5,
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        symmetric_cbar=False,
        colorbar=True,
        title=None,
        figure=fig,
        axes=(0.02, 0.03, 0.92, 0.80),
    )
    # A separate title band keeps cohort labels clear of the L/R orientation marks.
    fig.suptitle(title, y=0.98, fontsize=10)
    out.save(fig, name)
    display.close()


def plot_groups(directory, out, cutoff=0.001, brain_parameters=None, atlas=None):
    import matplotlib.pyplot as plt

    directory = resolve_project_path(directory)
    validate_run_schema(directory)
    path = directory / "all_tests.csv"
    table = load_group_results(path, cutoff)
    out.input(path)
    regional = [k for k, spec in param_config.items() if tuple(spec["shape"]) != (1,)]
    if not regional:
        raise ValueError("No regional parameters configured")
    if brain_parameters and not set(brain_parameters) <= set(regional):
        raise ValueError(
            "Brain-map parameters must be explicitly selected regional targets"
        )
    labels_path = PACKAGE_ROOT / "assets" / "aal_regions.csv"
    labels = pd.read_csv(labels_path)
    out.input(labels_path)
    if brain_parameters:
        atlas = resolve_project_path(atlas)
        out.input(atlas)
    out.audit.update(
        q_threshold=cutoff,
        undefined_tests=int(table["q"].isna().sum()),
        hypothesis_family="all configured regional and global tests, as supplied by group analysis",
        brain_parameters=brain_parameters or [],
    )
    count_rows = []
    for i, dataset in enumerate(table["dataset"].unique()):
        rows = table[table["dataset"] == dataset]
        counts, means = [], []
        for parameter in regional:
            sub = rows[rows["parameter"] == parameter].sort_values("region_index")
            counts.append(int(sub["significant_for_plot"].sum()))
            means.append(sub["t"].to_numpy())
            count_rows.append(
                {
                    "dataset": dataset,
                    "parameter": parameter,
                    "n_regions": len(sub),
                    "n_significant": counts[-1],
                    "n_undefined": int(sub["q"].isna().sum()),
                    "q_cutoff": cutoff,
                }
            )
        order = np.argsort(-np.asarray(counts), kind="stable")
        fig, ax = plt.subplots(figsize=cm(14, 8))
        ax.barh(np.arange(len(regional)), np.asarray(counts)[order], color=PALETTE[5])
        ax.set_yticks(np.arange(len(regional)), np.asarray(regional)[order], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(int(np.prod(param_config[k]["shape"])) for k in regional))
        ax.set_xlabel(f"Significant regions (BH q < {cutoff:g})")
        ax.set_title(str(dataset))
        for y, value in enumerate(np.asarray(counts)[order]):
            ax.text(value + 0.6, y, str(value), va="center", fontsize=7)
        fig.tight_layout()
        out.save(fig, f"group_{i:02d}_region_counts")
        matrix = np.stack(means)
        finite = matrix[np.isfinite(matrix)]
        maximum = max(float(np.abs(finite).max()), 1e-8) if len(finite) else 1.0
        fig, ax = plt.subplots(figsize=cm(21, 7))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#bdbdbd")
        im = ax.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            vmin=-maximum,
            vmax=maximum,
            aspect="auto",
        )
        ax.set_yticks(range(len(regional)), regional, fontsize=8)
        ax.set_xlabel("AAL region index (0-based)")
        ax.set_title(
            f"{dataset}: all regional Welch t values (case − control)", fontsize=9
        )
        fig.colorbar(
            im, ax=ax, label="Welch t (grey = undefined)", fraction=0.025, pad=0.02
        )
        fig.tight_layout()
        out.save(fig, f"group_{i:02d}_all_t")
        for parameter in brain_parameters or []:
            sub = rows[rows["parameter"] == parameter]
            draw_brain_map(
                sub,
                atlas,
                labels,
                out,
                f"group_{i:02d}_{parameter}_aal",
                f"{dataset} · {parameter} · BH q < {cutoff:g}",
            )
    out.table(pd.DataFrame(count_rows), "regional_counts_plotted.csv")
    out.table(table, "all_tests_plotted.csv")
