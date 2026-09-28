"""Draw the four-task comparison from complete run metrics (no backbone imports)."""

import numpy as np
import pandas as pd
from .common import read_table, require_columns, numeric
from .style import PALETTE, cm


def load_runs(run_specs):
    if len({name for name, _ in run_specs}) != len(run_specs):
        raise ValueError("Model labels must be distinct")
    tables, reference = [], None
    for label, path in run_specs:
        table = read_table(path)
        require_columns(table, ["dataset", "seed", "fold", "bacc", "auroc", "aucpr"])
        if table["dataset"].isna().any() or table["dataset"].str.strip().eq("").any():
            raise ValueError(
                "Missing dataset identity; no classification rows may be discarded"
            )
        numeric(table, ["seed", "fold", "bacc", "auroc", "aucpr"])
        if (
            (
                (table[["bacc", "auroc", "aucpr"]] < 0)
                | (table[["bacc", "auroc", "aucpr"]] > 1)
            )
            .any()
            .any()
        ):
            raise ValueError("Classification metrics must be proportions in [0,1]")
        if table.duplicated(["dataset", "seed", "fold"]).any():
            raise ValueError(
                "Duplicate classification runs; plotting never averages duplicates away"
            )
        for dataset, rows in table.groupby("dataset"):
            if len(rows) != 10 or rows["seed"].nunique() != 2:
                raise ValueError(f"{label}/{dataset}: expected all 2 seeds × 5 folds")
            for _, seed_rows in rows.groupby("seed"):
                if set(seed_rows["fold"]) != set(range(5)):
                    raise ValueError("Missing classification folds")
        keys = set(
            table[["dataset", "seed", "fold"]].itertuples(index=False, name=None)
        )
        if reference is not None and keys != reference:
            raise ValueError(
                "Comparison models must supply identical dataset/seed/fold coverage"
            )
        reference = keys
        table["model"] = label
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def plot_downstream(run_specs, out):
    import matplotlib.pyplot as plt

    table = load_runs(run_specs)
    for _, path in run_specs:
        out.input(path)
    out.audit["classification_rows"] = len(table)
    out.audit["comparison_note"] = (
        "Supplied run metrics only; no baseline implementation or weighting changes. Matching split IDs do not alone prove identical subject lists."
    )
    datasets = table["dataset"].drop_duplicates().tolist()
    models = [name for name, _ in run_specs]
    summary = []
    for metric in ("bacc", "auroc", "aucpr"):
        fig, ax = plt.subplots(figsize=cm(max(14, len(datasets) * 4), 7))
        width = 0.8 / len(models)
        for j, model in enumerate(models):
            means, stds, runs = [], [], []
            for dataset in datasets:
                values = table.loc[
                    (table["model"] == model) & (table["dataset"] == dataset), metric
                ].to_numpy()
                means.append(values.mean())
                stds.append(values.std(ddof=1))
                runs.append(values)
                summary.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "metric": metric,
                        "n_runs": len(values),
                        "mean": values.mean(),
                        "std": values.std(ddof=1),
                    }
                )
            x = np.arange(len(datasets)) - 0.4 + width / 2 + j * width
            ax.bar(
                x,
                means,
                width,
                yerr=stds,
                label=model,
                color=PALETTE[(j + 1) % len(PALETTE)],
                capsize=2,
                alpha=0.8,
            )
            for pos, values in zip(x, runs):
                # Every run is displayed; offsets are deterministic visual spacing only.
                ax.scatter(
                    pos + np.linspace(-width * 0.22, width * 0.22, len(values)),
                    values,
                    s=8,
                    color=".25",
                    zorder=3,
                )
        ax.set_xticks(range(len(datasets)), datasets)
        ax.set_ylabel(metric.upper())
        ax.set_title("All runs: mean ± sample SD", fontsize=10)
        # Error bars may extend beyond [0,1]; keep them visible rather than clipping.
        ax.set_ylim(
            min(0, min(r["mean"] - r["std"] for r in summary if r["metric"] == metric)),
            max(1, max(r["mean"] + r["std"] for r in summary if r["metric"] == metric)),
        )
        ax.legend(
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(3, len(models)),
        )
        fig.tight_layout()
        out.save(fig, f"downstream_{metric}")
    out.table(pd.DataFrame(summary), "downstream_summary_plotted.csv")
    out.table(table, "downstream_all_runs_plotted.csv")
