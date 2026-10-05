from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
)

DRIFTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "drifted"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)


def ensure_report_directory():
    """
    Create the reports directory if required.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_summary():
    """
    Load the project-level drift summary.
    """

    path = (
        RESULTS_DIR
        / "drift_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Drift summary not found: {path}\n"
            "Run experiments/run_experiment.py first."
        )

    return pd.read_csv(
        path
    )


def load_scenario_metrics(
    scenario,
):
    """
    Load feature-level detection metrics
    for one drift scenario.
    """

    path = (
        DRIFTED_DIR
        / scenario
        / "drift_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics not found: {path}\n"
            "Run experiments/run_experiment.py first."
        )

    return pd.read_csv(
        path
    )


def plot_dsi_summary(
    summary,
):
    """
    Plot DSI for all drift scenarios.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        summary["scenario"],
        summary["dsi"],
    )

    ax.set_title(
        "DataDecay-X Drift Severity Index"
    )

    ax.set_xlabel(
        "Drift Scenario"
    )

    ax.set_ylabel(
        "Drift Severity Index"
    )

    ax.set_ylim(
        0,
        max(
            1.0,
            summary["dsi"].max()
            * 1.20,
        ),
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / "dsi_summary.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def plot_metric_comparison(
    summary,
):
    """
    Plot average PSI, KS and normalized
    Wasserstein scores by scenario.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = range(
        len(summary)
    )

    width = 0.25

    ax.bar(
        [
            value - width
            for value in x
        ],
        summary["mean_psi"],
        width=width,
        label="Mean PSI",
    )

    ax.bar(
        x,
        summary["mean_ks"],
        width=width,
        label="Mean KS",
    )

    ax.bar(
        [
            value + width
            for value in x
        ],
        summary[
            "mean_wasserstein"
        ],
        width=width,
        label="Mean Wasserstein",
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        summary["scenario"]
    )

    ax.set_title(
        "Drift Detection Metrics by Scenario"
    )

    ax.set_xlabel(
        "Drift Scenario"
    )

    ax.set_ylabel(
        "Metric Value"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / "metric_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def plot_feature_psi(
    scenario,
):
    """
    Plot PSI for each affected feature.
    """

    metrics = load_scenario_metrics(
        scenario
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        metrics["feature"],
        metrics["psi"],
    )

    ax.set_title(
        f"Feature PSI — {scenario.title()} Drift"
    )

    ax.set_xlabel(
        "Feature"
    )

    ax.set_ylabel(
        "PSI"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / f"{scenario}_psi.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def plot_feature_ks(
    scenario,
):
    """
    Plot KS statistic for each affected feature.
    """

    metrics = load_scenario_metrics(
        scenario
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        metrics["feature"],
        metrics["ks_statistic"],
    )

    ax.set_title(
        f"Feature KS Statistic — "
        f"{scenario.title()} Drift"
    )

    ax.set_xlabel(
        "Feature"
    )

    ax.set_ylabel(
        "KS Statistic"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / f"{scenario}_ks.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def plot_feature_wasserstein(
    scenario,
):
    """
    Plot normalized Wasserstein distance
    for each affected feature.
    """

    metrics = load_scenario_metrics(
        scenario
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        metrics["feature"],
        metrics[
            "normalized_wasserstein"
        ],
    )

    ax.set_title(
        f"Normalized Wasserstein Distance — "
        f"{scenario.title()} Drift"
    )

    ax.set_xlabel(
        "Feature"
    )

    ax.set_ylabel(
        "Normalized Wasserstein"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / f"{scenario}_wasserstein.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return output_path


def plot_all_scenarios(
    summary,
):
    """
    Generate feature-level plots for every scenario.
    """

    scenarios = summary[
        "scenario"
    ].tolist()

    output_paths = []

    for scenario in scenarios:

        output_paths.append(
            plot_feature_psi(
                scenario
            )
        )

        output_paths.append(
            plot_feature_ks(
                scenario
            )
        )

        output_paths.append(
            plot_feature_wasserstein(
                scenario
            )
        )

    return output_paths


def main():

    print("=" * 80)

    print(
        "DataDecay-X | Person 1 Drift Plots"
    )

    print("=" * 80)

    print()

    ensure_report_directory()

    summary = load_summary()

    print(
        "Loaded drift summary."
    )

    print(
        summary[
            [
                "scenario",
                "dsi",
                "severity",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    # ---------------------------------------------------------
    # DSI PLOT
    # ---------------------------------------------------------

    dsi_path = plot_dsi_summary(
        summary
    )

    print(
        "Created:",
        dsi_path
    )

    # ---------------------------------------------------------
    # METRIC COMPARISON
    # ---------------------------------------------------------

    comparison_path = (
        plot_metric_comparison(
            summary
        )
    )

    print(
        "Created:",
        comparison_path
    )

    # ---------------------------------------------------------
    # FEATURE-LEVEL PLOTS
    # ---------------------------------------------------------

    feature_paths = (
        plot_all_scenarios(
            summary
        )
    )

    for path in feature_paths:

        print(
            "Created:",
            path
        )

    print()

    print("=" * 80)

    print(
        "PERSON 1 PLOTS COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()