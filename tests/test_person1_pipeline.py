from pathlib import Path
import sys

import numpy as np
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
    make_windows,
    DriftSimulator,
    calculate_feature_metrics,
    calculate_dsi,
    classify_dsi,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)


def select_features(
    reference,
    count=5,
):
    features = numeric_features(
        reference
    )

    variance = (
        reference[features]
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


def check_dataset(
    df,
):
    assert not df.empty

    assert (
        "timestamp"
        in df.columns
    )

    assert (
        df["timestamp"]
        .is_monotonic_increasing
    )

    assert (
        len(
            numeric_features(df)
        )
        > 0
    )


def check_split(
    df,
):
    (
        reference,
        monitoring,
        evaluation,
    ) = temporal_split(
        df,
        train_fraction=0.60,
        monitor_fraction=0.20,
    )

    assert len(reference) > 0
    assert len(monitoring) > 0
    assert len(evaluation) > 0

    assert (
        len(reference)
        + len(monitoring)
        + len(evaluation)
        == len(df)
    )

    assert (
        reference["timestamp"].iloc[-1]
        < monitoring["timestamp"].iloc[0]
    )

    assert (
        monitoring["timestamp"].iloc[-1]
        < evaluation["timestamp"].iloc[0]
    )

    return (
        reference,
        monitoring,
        evaluation,
    )


def check_windows(
    monitoring,
):
    windows = make_windows(
        monitoring,
        window_size=1000,
        step_size=1000,
    )

    assert len(windows) > 0

    for window in windows:

        assert (
            len(window)
            == 1000
        )

        assert (
            window["timestamp"]
            .is_monotonic_increasing
        )


def main():

    print("=" * 80)

    print(
        "DataDecay-X | Person 1 Full Pipeline Test"
    )

    print("=" * 80)

    df = load_electricity_dataset(
        DATASET_PATH
    )

    check_dataset(
        df
    )

    print(
        "Dataset: PASS"
    )

    (
        reference,
        monitoring,
        evaluation,
    ) = check_split(
        df
    )

    print(
        "Temporal split: PASS"
    )

    check_windows(
        monitoring
    )

    print(
        "Window generation: PASS"
    )

    features = select_features(
        reference
    )

    print(
        "Features:",
        features,
    )

    simulator = DriftSimulator(
        seed=42
    )

    # ---------------------------------------------------------
    # STABLE
    # ---------------------------------------------------------

    stable, metadata = (
        simulator.stable(
            monitoring,
            features,
        )
    )

    pd.testing.assert_frame_equal(
        stable,
        monitoring,
    )

    print(
        "Stable: PASS"
    )

    # ---------------------------------------------------------
    # GRADUAL
    # ---------------------------------------------------------

    gradual, metadata = (
        simulator.gradual(
            monitoring,
            features,
            magnitude=0.30,
            start_window=5000,
            end_window=15000,
            baseline_df=reference,
        )
    )

    assert (
        gradual.shape
        == monitoring.shape
    )

    print(
        "Gradual: PASS"
    )

    # ---------------------------------------------------------
    # ABRUPT
    # ---------------------------------------------------------

    abrupt, metadata = (
        simulator.abrupt(
            monitoring,
            features,
            magnitude=0.30,
            start_window=10000,
            baseline_df=reference,
        )
    )

    assert (
        abrupt.shape
        == monitoring.shape
    )

    print(
        "Abrupt: PASS"
    )

    # ---------------------------------------------------------
    # RECURRING
    # ---------------------------------------------------------

    recurring, metadata = (
        simulator.recurring(
            monitoring,
            features,
            magnitude=0.30,
            cycle_length=1000,
            baseline_df=reference,
        )
    )

    assert (
        recurring.shape
        == monitoring.shape
    )

    print(
        "Recurring: PASS"
    )

    # ---------------------------------------------------------
    # DETECTION
    # ---------------------------------------------------------

    clean = reference.copy()

    clean_metrics = (
        calculate_feature_metrics(
            reference_df=reference,
            current_df=clean,
            features=features,
        )
    )

    clean_dsi = calculate_dsi(
        clean_metrics
    )

    drift_metrics = (
        calculate_feature_metrics(
            reference_df=reference,
            current_df=abrupt,
            features=features,
        )
    )

    drift_dsi = calculate_dsi(
        drift_metrics
    )

    clean_severity = (
        classify_dsi(
            clean_dsi
        )
    )

    drift_severity = (
        classify_dsi(
            drift_dsi
        )
    )

    assert np.isclose(
        clean_dsi,
        0.0,
        atol=1e-10,
    )

    assert (
        drift_dsi
        > clean_dsi
    )

    print(
        "Detection: PASS"
    )

    print(
        "Clean DSI:",
        clean_dsi,
    )

    print(
        "Drift DSI:",
        drift_dsi,
    )

    print(
        "Clean severity:",
        clean_severity,
    )

    print(
        "Drift severity:",
        drift_severity,
    )

    print()

    print("=" * 80)

    print(
        "PERSON 1 FULL PIPELINE TEST PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()