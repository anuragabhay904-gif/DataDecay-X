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
    / "person2"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "person2"
)


def ensure_report_directory():
    """
    Create the Person 2 reports directory.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_model_metrics():
    """
    Load baseline and drifted model metrics.
    """

    path = (
        RESULTS_DIR
        / "model_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model metrics not found: {path}\n"
            "Run run_person2_experiment.py first."
        )

    return pd.read_csv(
        path
    )


def load_reliability_metrics():
    """
    Load model reliability metrics.
    """

    path = (
        RESULTS_DIR
        / "reliability_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Reliability metrics not found: {path}\n"
            "Run run_person2_experiment.py first."
        )

    return pd.read_csv(
        path
    )


def load_monitoring_metrics():
    """
    Load combined model monitoring metrics.
    """

    path = (
        RESULTS_DIR
        / "monitoring_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Monitoring metrics not found: {path}\n"
            "Run run_person2_experiment.py first."
        )

    return pd.read_csv(
        path
    )


def plot_model_performance(
    metrics,
):
    """
    Plot model performance before and after drift.
    """

    if "r2" not in metrics.columns:
        raise ValueError(
            "Expected 'r2' column in model metrics."
        )

    periods = metrics[
        "period"
    ].tolist()

    scores = metrics[
        "r2"
    ].tolist()

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        periods,
        scores,
    )

    ax.set_title(
        "Model Performance Across Time"
    )

    ax.set_xlabel(
        "Evaluation Period"
    )

    ax.set_ylabel(
        "R²"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / "model_performance.png"
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


def plot_model_errors(
    metrics,
):
    """
    Plot MAE and RMSE across evaluation periods.
    """

    if (
        "mae" not in metrics.columns
        or "rmse" not in metrics.columns
    ):
        raise ValueError(
            "Model metrics must contain "
            "'mae' and 'rmse'."
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    x = range(
        len(metrics)
    )

    width = 0.35

    ax.bar(
        [
            value - width / 2
            for value in x
        ],
        metrics["mae"],
        width=width,
        label="MAE",
    )

    ax.bar(
        [
            value + width / 2
            for value in x
        ],
        metrics["rmse"],
        width=width,
        label="RMSE",
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        metrics["period"]
    )

    ax.set_title(
        "Model Error Across Time"
    )

    ax.set_xlabel(
        "Evaluation Period"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / "model_errors.png"
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


def plot_reliability(
    reliability,
):
    """
    Plot model reliability index.
    """

    if (
        "baseline_score"
        not in reliability.columns
        or "drifted_score"
        not in reliability.columns
    ):
        raise ValueError(
            "Reliability metrics must contain "
            "'baseline_score' and 'drifted_score'."
        )

    baseline = reliability[
        "baseline_score"
    ].tolist()

    drifted = reliability[
        "drifted_score"
    ].tolist()

    models = reliability[
        "model_name"
    ].tolist()

    x = range(
        len(models)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        [
            value - width / 2
            for value in x
        ],
        baseline,
        width=width,
        label="Baseline",
    )

    ax.bar(
        [
            value + width / 2
            for value in x
        ],
        drifted,
        width=width,
        label="Drifted",
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        models
    )

    ax.set_title(
        "Model Reliability: Baseline vs Drifted"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Performance Score"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / "model_reliability.png"
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


def plot_degradation(
    reliability,
):
    """
    Plot relative model-performance degradation.
    """

    required = {
        "model_name",
        "degradation_ratio",
    }

    missing = (
        required
        - set(
            reliability.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing reliability columns: "
            f"{sorted(missing)}"
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        reliability["model_name"],
        reliability[
            "degradation_ratio"
        ],
    )

    ax.set_title(
        "Relative Model Performance Degradation"
    )

    ax.set_xlabel(
        "Model"
    )

    ax.set_ylabel(
        "Degradation Ratio"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / "model_degradation.png"
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


def plot_drift_vs_reliability(
    monitoring,
):
    """
    Visualize detected drift against model
    reliability degradation.
    """

    if (
        "dsi" not in monitoring.columns
        or "reliability_degradation_ratio"
        not in monitoring.columns
    ):
        raise ValueError(
            "Monitoring metrics must contain "
            "'dsi' and "
            "'reliability_degradation_ratio'."
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        monitoring["dsi"],
        monitoring[
            "reliability_degradation_ratio"
        ],
    )

    for _, row in monitoring.iterrows():

        ax.annotate(
            row["model_name"],
            (
                row["dsi"],
                row[
                    "reliability_degradation_ratio"
                ],
            ),
        )

    ax.set_title(
        "Data Drift vs Model Reliability Degradation"
    )

    ax.set_xlabel(
        "Drift Severity Index"
    )

    ax.set_ylabel(
        "Reliability Degradation Ratio"
    )

    ax.grid(
        alpha=0.3,
    )

    fig.tight_layout()

    output_path = (
        REPORTS_DIR
        / "drift_vs_reliability.png"
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


def main():

    print("=" * 80)

    print(
        "DataDecay-X | Person 2 Reports"
    )

    print("=" * 80)

    print()

    ensure_report_directory()

    model_metrics = (
        load_model_metrics()
    )

    reliability_metrics = (
        load_reliability_metrics()
    )

    monitoring_metrics = (
        load_monitoring_metrics()
    )

    generated = []

    generated.append(
        plot_model_performance(
            model_metrics
        )
    )

    generated.append(
        plot_model_errors(
            model_metrics
        )
    )

    generated.append(
        plot_reliability(
            reliability_metrics
        )
    )

    generated.append(
        plot_degradation(
            reliability_metrics
        )
    )

    generated.append(
        plot_drift_vs_reliability(
            monitoring_metrics
        )
    )

    print(
        "Generated reports:"
    )

    for path in generated:
        print(
            " -",
            path,
        )

    print()

    print("=" * 80)

    print(
        "PERSON 2 REPORTS COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()