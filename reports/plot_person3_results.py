from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "person3"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "person3"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_csv(filename):
    path = RESULTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}"
        )

    return pd.read_csv(path)


def plot_synthetic_quality():
    path = (
        RESULTS_DIR
        / "distribution_comparison.csv"
    )

    if not path.exists():
        return

    df = pd.read_csv(path)

    if df.empty:
        return

    features = df["feature"]

    x = range(len(features))

    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [
            value - width / 2
            for value in x
        ],
        df["reference_mean"],
        width=width,
        label="Reference mean",
    )

    plt.bar(
        [
            value + width / 2
            for value in x
        ],
        df["synthetic_mean"],
        width=width,
        label="Synthetic mean",
    )

    plt.xticks(
        list(x),
        features,
        rotation=45,
        ha="right",
    )

    plt.ylabel("Mean")
    plt.title(
        "Reference vs Synthetic Feature Means"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "synthetic_mean_comparison.png",
        dpi=150,
    )

    plt.close()


def plot_recovery_comparison():
    df = load_csv(
        "recovery_comparison.csv"
    )

    if df.empty:
        return

    plt.figure(figsize=(10, 6))

    plt.bar(
        df["method"],
        df["rmse"],
    )

    plt.ylabel("RMSE")
    plt.xlabel("Recovery method")
    plt.title(
        "Recovery Method Comparison - RMSE"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "recovery_rmse_comparison.png",
        dpi=150,
    )

    plt.close()


def plot_recovery_mae():
    df = load_csv(
        "recovery_comparison.csv"
    )

    if df.empty:
        return

    plt.figure(figsize=(10, 6))

    plt.bar(
        df["method"],
        df["mae"],
    )

    plt.ylabel("MAE")
    plt.xlabel("Recovery method")
    plt.title(
        "Recovery Method Comparison - MAE"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "recovery_mae_comparison.png",
        dpi=150,
    )

    plt.close()


def plot_distribution_errors():
    df = load_csv(
        "distribution_comparison.csv"
    )

    if df.empty:
        return

    x = range(len(df))

    plt.figure(figsize=(10, 6))

    plt.bar(
        [
            value - 0.2
            for value in x
        ],
        df["mean_relative_error"],
        width=0.4,
        label="Mean relative error",
    )

    plt.bar(
        [
            value + 0.2
            for value in x
        ],
        df["std_relative_error"],
        width=0.4,
        label="Std relative error",
    )

    plt.xticks(
        list(x),
        df["feature"],
        rotation=45,
        ha="right",
    )

    plt.ylabel("Relative error")
    plt.xlabel("Feature")
    plt.title(
        "Synthetic Distribution Errors"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "synthetic_distribution_errors.png",
        dpi=150,
    )

    plt.close()


def plot_quality_summary():
    path = (
        RESULTS_DIR
        / "quality_gate.json"
    )

    if not path.exists():
        return

    import json

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    labels = [
        "Schema",
        "Missingness",
        "Distribution",
        "Correlation",
    ]

    values = [
        int(
            data.get(
                "schema_valid",
                False,
            )
        ),
        int(
            data.get(
                "missingness_valid",
                False,
            )
        ),
        int(
            data.get(
                "distribution_valid",
                False,
            )
        ),
        int(
            data.get(
                "correlation_valid",
                False,
            )
        ),
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(
        labels,
        values,
    )

    plt.ylim(
        0,
        1.1,
    )

    plt.ylabel(
        "Pass status"
    )

    plt.title(
        "Synthetic Data Quality Gate"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "quality_gate_summary.png",
        dpi=150,
    )

    plt.close()


def main():
    print(
        "=" * 70
    )
    print(
        "PERSON 3 - RESULT PLOTS"
    )
    print(
        "=" * 70
    )

    plot_synthetic_quality()

    plot_recovery_comparison()

    plot_recovery_mae()

    plot_distribution_errors()

    plot_quality_summary()

    print(
        f"\nPlots saved to: {REPORT_DIR}"
    )


if __name__ == "__main__":
    main()