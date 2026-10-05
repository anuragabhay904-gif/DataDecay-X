from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.datadecay import (
    load_electricity_dataset,
    numeric_features,
    temporal_split,
    DriftSimulator,
    calculate_feature_metrics,
    calculate_dsi,
    classify_dsi,
    save_json,
    ensure_dir,
)


DATASET_PATH = (
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


DRIFTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "drifted"
)


SEED = 42

TRAIN_FRACTION = 0.60

MONITOR_FRACTION = 0.20

DRIFT_MAGNITUDE = 0.30

SCENARIOS = [
    "stable",
    "gradual",
    "abrupt",
    "recurring",
]


def select_features(
    df,
    count=5,
):
    """
    Select numeric features with
    non-zero variance.

    Highest-variance features are selected.
    """

    features = numeric_features(
        df
    )

    variance = (
        df[features]
        .std()
        .dropna()
    )

    variance = variance[
        variance > 0
    ]

    return (
        variance
        .sort_values(
            ascending=False
        )
        .head(count)
        .index
        .tolist()
    )


def run_scenario(
    scenario,
    reference,
    monitoring,
    features,
):
    """
    Run one drift scenario.
    """

    simulator = DriftSimulator(
        seed=SEED
    )

    if scenario == "stable":

        drifted_monitoring, metadata = (
            simulator.stable(
                df=monitoring,
                features=features,
            )
        )

    elif scenario == "gradual":

        start = (
            len(monitoring)
            // 4
        )

        end = (
            len(monitoring)
            * 3
            // 4
        )

        drifted_monitoring, metadata = (
            simulator.gradual(
                df=monitoring,
                features=features,
                magnitude=DRIFT_MAGNITUDE,
                start_window=start,
                end_window=end,
                baseline_df=reference,
            )
        )

    elif scenario == "abrupt":

        start = (
            len(monitoring)
            // 2
        )

        drifted_monitoring, metadata = (
            simulator.abrupt(
                df=monitoring,
                features=features,
                magnitude=DRIFT_MAGNITUDE,
                start_window=start,
                baseline_df=reference,
            )
        )

    elif scenario == "recurring":

        drifted_monitoring, metadata = (
            simulator.recurring(
                df=monitoring,
                features=features,
                magnitude=DRIFT_MAGNITUDE,
                cycle_length=4,
                baseline_df=reference,
            )
        )

    else:
        raise ValueError(
            f"Unknown scenario: {scenario}"
        )

    metrics = calculate_feature_metrics(
        reference_df=reference,
        current_df=drifted_monitoring,
        features=features,
    )

    dsi = calculate_dsi(
        metrics
    )

    severity = classify_dsi(
        dsi
    )

    return (
        drifted_monitoring,
        metadata,
        metrics,
        dsi,
        severity,
    )


def save_metrics(
    metrics,
    scenario,
):
    """
    Save feature-level drift metrics.
    """

    path = (
        RESULTS_DIR
        / f"{scenario}_metrics.csv"
    )

    metrics.to_csv(
        path,
        index=False,
    )


def save_drifted_data(
    drifted_data,
    scenario,
):
    """
    Save drifted monitoring data.
    """

    scenario_dir = (
        DRIFTED_DIR
        / scenario
    )

    ensure_dir(
        scenario_dir
    )

    path = (
        scenario_dir
        / "monitoring.csv"
    )

    drifted_data.to_csv(
        path,
        index=False,
    )


def main():

    print("=" * 70)

    print(
        "DataDecay-X | Person 1 Experiment"
    )

    print("=" * 70)

    ensure_dir(
        RESULTS_DIR
    )

    ensure_dir(
        DRIFTED_DIR
    )

    print()

    print(
        "Loading dataset..."
    )

    df = load_electricity_dataset(
        DATASET_PATH
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    print(
        "First timestamp:",
        df["timestamp"].min(),
    )

    print(
        "Last timestamp:",
        df["timestamp"].max(),
    )

    print()

    print(
        "Creating chronological windows..."
    )

    (
        reference,
        monitoring,
        evaluation,
    ) = temporal_split(
        df,
        train_fraction=TRAIN_FRACTION,
        monitor_fraction=MONITOR_FRACTION,
    )

    print(
        "Reference:",
        reference.shape,
    )

    print(
        "Monitoring:",
        monitoring.shape,
    )

    print(
        "Evaluation:",
        evaluation.shape,
    )

    print()

    print(
        "Selecting experiment features..."
    )

    features = select_features(
        reference,
        count=5,
    )

    print(
        "Selected features:",
        features,
    )

    print()

    print(
        "Running scenarios..."
    )

    summary_rows = []

    all_metadata = {}

    for scenario in SCENARIOS:

        print()

        print("-" * 70)

        print(
            "SCENARIO:",
            scenario.upper(),
        )

        print("-" * 70)

        (
            drifted_monitoring,
            metadata,
            metrics,
            dsi,
            severity,
        ) = run_scenario(
            scenario=scenario,
            reference=reference,
            monitoring=monitoring,
            features=features,
        )

        print(
            "DSI:",
            round(dsi, 4),
        )

        print(
            "Severity:",
            severity,
        )

        print(
            "Affected features:",
            metadata.affected_features,
        )

        print(
            "Magnitude:",
            metadata.magnitude,
        )

        print(
            "Schedule:",
            metadata.schedule,
        )

        save_metrics(
            metrics,
            scenario,
        )

        save_drifted_data(
            drifted_monitoring,
            scenario,
        )

        all_metadata[
            scenario
        ] = metadata.to_dict()

        summary_rows.append(
            {
                "scenario": scenario,
                "dsi": dsi,
                "severity": severity,
                "magnitude":
                    metadata.magnitude,
                "affected_feature_count":
                    len(
                        metadata.affected_features
                    ),
                "affected_features":
                    ", ".join(
                        metadata.affected_features
                    ),
                "schedule":
                    metadata.schedule,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULTS_DIR
        / "drift_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    metadata_path = (
        RESULTS_DIR
        / "drift_metadata.json"
    )

    save_json(
        all_metadata,
        metadata_path,
    )

    window_metadata = {
        "reference": {
            "rows": len(reference),
            "start":
                reference["timestamp"].min(),
            "end":
                reference["timestamp"].max(),
        },
        "monitoring": {
            "rows": len(monitoring),
            "start":
                monitoring["timestamp"].min(),
            "end":
                monitoring["timestamp"].max(),
        },
        "evaluation": {
            "rows": len(evaluation),
            "start":
                evaluation["timestamp"].min(),
            "end":
                evaluation["timestamp"].max(),
        },
        "features": features,
        "seed": SEED,
    }

    save_json(
        window_metadata,
        RESULTS_DIR
        / "window_metadata.json",
    )

    print()

    print("=" * 70)

    print(
        "FINAL SUMMARY"
    )

    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Results written to:",
        RESULTS_DIR,
    )

    print(
        "Drifted datasets written to:",
        DRIFTED_DIR,
    )

    print()

    print(
        "Person 1 experiment complete."
    )


if __name__ == "__main__":
    main()