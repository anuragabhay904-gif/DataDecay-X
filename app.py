from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.datadecay.quality_gate import run_quality_gate
from src.datadecay.recovery import (
    create_missingness,
    recover_missing_values,
    recovery_quality,
)
from src.datadecay.synthetic import (
    generate_synthetic_data,
    synthetic_quality_score,
)
from src.datadecay.utils import (
    load_electricity_dataset,
    temporal_split,
)


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
)

PERSON1_RESULTS = (
    RESULTS_DIR
    / "drift_summary.csv"
)

PERSON2_RESULTS = (
    RESULTS_DIR
    / "person2"
)

PERSON3_RESULTS = (
    RESULTS_DIR
    / "person3"
)


def load_dataset():
    """
    Load the real electricity dataset.
    """
    return load_electricity_dataset(
        DATA_PATH
    )


def select_features(
    df,
    n_features=5,
):
    """
    Select the highest-variance numeric electricity series.
    """
    numeric_columns = [
        column
        for column in df.columns
        if column != "timestamp"
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    variances = (
        df[numeric_columns]
        .var()
        .sort_values(
            ascending=False
        )
    )

    return list(
        variances.head(
            n_features
        ).index
    )


def load_json(path):
    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def show_dataset_overview(df):
    print("\nDATASET OVERVIEW")
    print("-" * 60)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        f"Start: {df['timestamp'].min()}"
    )

    print(
        f"End: {df['timestamp'].max()}"
    )

    print(
        f"Electricity series: "
        f"{len(df.columns) - 1}"
    )


def show_split_overview(df):
    reference, monitoring, evaluation = (
        temporal_split(
            df,
            train_fraction=0.60,
            monitor_fraction=0.20,
        )
    )

    print("\nTEMPORAL SPLIT")
    print("-" * 60)

    print(
        f"Reference: "
        f"{len(reference):,} rows"
    )

    print(
        f"Monitoring: "
        f"{len(monitoring):,} rows"
    )

    print(
        f"Evaluation: "
        f"{len(evaluation):,} rows"
    )

    print(
        "\nReference:"
    )

    print(
        f"  {reference['timestamp'].min()}"
        f" -> "
        f"{reference['timestamp'].max()}"
    )

    print(
        "Monitoring:"
    )

    print(
        f"  {monitoring['timestamp'].min()}"
        f" -> "
        f"{monitoring['timestamp'].max()}"
    )

    print(
        "Evaluation:"
    )

    print(
        f"  {evaluation['timestamp'].min()}"
        f" -> "
        f"{evaluation['timestamp'].max()}"
    )


def run_synthetic_demo(
    df,
    n_features=5,
):
    """
    Run a lightweight synthetic-data and recovery demonstration.
    """
    reference, monitoring, _ = (
        temporal_split(
            df,
            train_fraction=0.60,
            monitor_fraction=0.20,
        )
    )

    features = select_features(
        reference,
        n_features=n_features,
    )

    reference_features = (
        reference[
            features
        ]
        .copy()
    )

    monitoring_sample = (
        monitoring[
            features
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("\nSYNTHETIC DATA DEMO")
    print("-" * 60)

    print(
        "Features:"
    )

    for feature in features:
        print(
            f"  - {feature}"
        )

    synthetic_result = (
        generate_synthetic_data(
            reference=reference_features,
            features=features,
            n_samples=len(
                monitoring_sample
            ),
            method="bootstrap",
            random_state=42,
            noise_scale=0.01,
        )
    )

    synthetic = (
        synthetic_result.samples
    )

    quality_score = (
        synthetic_quality_score(
            reference=reference_features,
            synthetic=synthetic,
            features=features,
        )
    )

    gate = run_quality_gate(
        reference=reference_features,
        candidate=synthetic,
        features=features,
        threshold=0.50,
        psi_threshold=0.50,
        wasserstein_threshold=2.0,
        correlation_difference=0.50,
    )

    print(
        f"Synthetic rows: "
        f"{len(synthetic):,}"
    )

    print(
        f"Synthetic quality score: "
        f"{quality_score:.4f}"
    )

    print(
        f"Quality gate accepted: "
        f"{gate.accepted}"
    )

    # Controlled corruption.
    clean_monitoring = (
        monitoring[
            ["timestamp"]
            + features
        ]
        .copy()
        .reset_index(drop=True)
    )

    corrupted = create_missingness(
        clean_monitoring,
        features=features,
        missing_fraction=0.05,
        random_state=42,
    )

    missing_cells = int(
        corrupted[
            features
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Artificially missing cells: "
        f"{missing_cells:,}"
    )

    # Interpolation recovery.
    recovered = recover_missing_values(
        corrupted,
        method="interpolate",
        features=features,
    )

    quality = recovery_quality(
        original=clean_monitoring,
        recovered=recovered.recovered,
        corrupted=corrupted,
        features=features,
    )

    print(
        f"Recovered cells: "
        f"{recovered.filled_cells:,}"
    )

    print(
        f"Remaining missing cells: "
        f"{recovered.remaining_missing_cells:,}"
    )

    if quality["rmse"] is not None:
        print(
            f"Recovery RMSE: "
            f"{quality['rmse']:.6f}"
        )

    return {
        "features": features,
        "synthetic": synthetic,
        "quality_gate": gate,
        "corrupted": corrupted,
        "recovered": recovered.recovered,
        "recovery_quality": quality,
    }


def plot_feature_series(
    df,
    feature,
    start=0,
    end=None,
):
    """
    Plot one electricity series.
    """
    if feature not in df.columns:
        raise ValueError(
            f"Feature not found: {feature}"
        )

    if end is None:
        end = len(df)

    subset = df.iloc[
        start:end
    ]

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        subset["timestamp"],
        subset[feature],
    )

    plt.xlabel(
        "Timestamp"
    )

    plt.ylabel(
        "Load"
    )

    plt.title(
        f"Electricity Load - {feature}"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.show()


def show_existing_results():
    """
    Display already-generated experiment summaries.
    """
    print("\nEXISTING EXPERIMENT RESULTS")
    print("-" * 60)

    if PERSON1_RESULTS.exists():
        person1 = pd.read_csv(
            PERSON1_RESULTS
        )

        print(
            "\nPerson 1 drift summary:"
        )

        print(
            person1.to_string(
                index=False
            )
        )
    else:
        print(
            "Person 1 results not found."
        )

    person2_summary = (
        PERSON2_RESULTS
        / "person2_summary.json"
    )

    if person2_summary.exists():
        data = load_json(
            person2_summary
        )

        print(
            "\nPerson 2 summary:"
        )

        print(
            json.dumps(
                data,
                indent=4,
                default=str,
            )
        )
    else:
        print(
            "Person 2 results not found."
        )

    person3_summary = (
        PERSON3_RESULTS
        / "person3_summary.json"
    )

    if person3_summary.exists():
        data = load_json(
            person3_summary
        )

        print(
            "\nPerson 3 summary:"
        )

        print(
            json.dumps(
                data,
                indent=4,
                default=str,
            )
        )
    else:
        print(
            "Person 3 results not found."
        )


def main():
    print("=" * 70)
    print("DataDecay-X")
    print("Data Drift, Reliability, Synthetic Recovery")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(
            "\nDataset not found:"
        )

        print(
            DATA_PATH
        )

        return

    print(
        "\nLoading dataset..."
    )

    df = load_dataset()

    show_dataset_overview(
        df
    )

    show_split_overview(
        df
    )

    # Run the Person 3 demonstration.
    run_synthetic_demo(
        df,
        n_features=5,
    )

    # Display results produced by the three project sections.
    show_existing_results()

    print(
        "\nDataDecay-X application finished."
    )


if __name__ == "__main__":
    main()