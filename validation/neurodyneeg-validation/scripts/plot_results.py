"""Draw NeuroDynEEG validation figures from complete result files, without model weights."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        help="New output directory; default: validation/neurodyneeg-validation/figures/<timestamp>",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=["svg", "png", "pdf"], default=["svg", "png"]
    )
    tasks = parser.add_subparsers(dest="task", required=True)
    p = tasks.add_parser(
        "recovery",
        help="All-sample parameter recovery grid and regional correlation heatmap",
    )
    p.add_argument("--results", required=True)
    p = tasks.add_parser("noise", help="Per-parameter SNR curves with clean baseline")
    p.add_argument("--results", required=True)
    p.add_argument("--scope", choices=["pooled", "sample_mean"], default="pooled")
    p = tasks.add_parser(
        "real-eeg",
        help="Real EEG only: retain the right panel of legacy sim/real topomaps",
    )
    p.add_argument(
        "--metrics",
        required=True,
        help="Complete closed_loop_segments.csv or unfiltered legacy closed_loop_metrics.csv",
    )
    p.add_argument(
        "--dataset-label",
        help="Dataset identity for a single-cohort legacy CSV without dataset column",
    )
    p = tasks.add_parser(
        "closed-loop-signals",
        help="Complete-record waveforms and subject-weighted Welch PSD",
    )
    p.add_argument(
        "--results",
        required=True,
        help="New closed-loop result directory containing record_*.npz",
    )
    p.add_argument(
        "--prepared", required=True, help="Matching prepared real EEG cache directory"
    )
    p.add_argument(
        "--segment-id",
        help="Optional explicitly chosen illustration segment ID; never selected by metric",
    )
    p = tasks.add_parser(
        "downstream", help="Classification results for all datasets/models supplied"
    )
    p.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("MODEL_LABEL", "RUNS_CSV"),
        required=True,
    )
    p = tasks.add_parser(
        "groups", help="All-parameter significance counts and regional t maps"
    )
    p.add_argument("--results", required=True)
    p.add_argument(
        "--q",
        type=float,
        default=0.001,
        help="Prespecified inferential significance threshold; not a sample filter",
    )
    p.add_argument(
        "--brain-parameter",
        action="append",
        help="Optional parameter(s) for AAL maps; chosen explicitly, not by the largest effect",
    )
    p.add_argument("--atlas", default=str(PACKAGE_ROOT / "assets" / "ROI_MNI_V4.nii"))
    # Accept output options on either side of the subcommand without overriding
    # explicitly supplied parent-parser values.
    for subparser in tasks.choices.values():
        subparser.add_argument(
            "--output", default=argparse.SUPPRESS, help="New output directory"
        )
        subparser.add_argument(
            "--formats",
            nargs="+",
            choices=["svg", "png", "pdf"],
            default=argparse.SUPPRESS,
        )
    return parser.parse_args()


def main():
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    from neurodyneeg_plots.common import FigureOutput
    from neurodyneeg_plots.style import set_paper_style

    set_paper_style()
    output = args.output or str(
        PACKAGE_ROOT
        / "figures"
        / (args.task + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    out = FigureOutput(output, args.formats)
    out.audit["command"] = vars(args)
    if args.task == "recovery":
        from neurodyneeg_plots.recovery import plot_recovery

        plot_recovery(args.results, out)
    elif args.task == "noise":
        from neurodyneeg_plots.robustness import plot_noise

        plot_noise(args.results, out, args.scope)
    elif args.task == "real-eeg":
        from neurodyneeg_plots.real_eeg import plot_real_eeg

        plot_real_eeg(args.metrics, out, args.dataset_label)
    elif args.task == "closed-loop-signals":
        from neurodyneeg_plots.signals import plot_signals

        plot_signals(args.results, args.prepared, out, args.segment_id)
    elif args.task == "downstream":
        from neurodyneeg_plots.downstream import plot_downstream

        plot_downstream(args.run, out)
    elif args.task == "groups":
        from neurodyneeg_plots.group_analysis import plot_groups

        plot_groups(args.results, out, args.q, args.brain_parameter, args.atlas)
    out.finish()


if __name__ == "__main__":
    main()
