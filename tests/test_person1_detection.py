import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.datadecay import (
    load_electricity_dataset,
    numeric_features,
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


def select_active_features(
    df,
    count=5,
):
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


def print_metrics(
    metrics,
):
    columns = [
        "feature",
        "psi",
        "ks_statistic",
        "ks_pvalue",
        "wasserstein",
        "normalized_wasserstein",
    ]

    print(
        metrics[
            columns
        ].to_string(
            index=False
        )
    )


def main():

    print("=" * 70)

    print(
        "DataDecay-X | Person 1 Detection Test"
    )

    print("=" * 70)

    print()

    df = load_electricity_dataset(
        DATASET_PATH
    )

    reference = (
        df.iloc[:30000]
        .copy()
        .reset_index(drop=True)
    )

    features = select_active_features(
        reference,
        count=5,
    )

    print(
        "Reference shape:",
        reference.shape,
    )

    print(
        "Selected features:",
        features,
    )

    simulator = DriftSimulator(
        seed=42
    )

    # ---------------------------------------------------------
    # CLEAN CONTROL
    # ---------------------------------------------------------

    current_clean = (
        reference.copy()
    )

    clean_current, clean_metadata = (
        simulator.stable(
            df=current_clean,
            features=features,
        )
    )

    clean_metrics = (
        calculate_feature_metrics(
            reference_df=reference,
            current_df=clean_current,
            features=features,
        )
    )

    clean_dsi = calculate_dsi(
        clean_metrics
    )

    clean_severity = (
        classify_dsi(
            clean_dsi
        )
    )

    print()

    print(
        "CONTROL / STABLE"
    )

    print(
        clean_metadata.to_dict()
    )

    print_metrics(
        clean_metrics
    )

    print(
        "DSI:",
        round(
            clean_dsi,
            4,
        ),
    )

    print(
        "SEVERITY:",
        clean_severity,
    )

    # ---------------------------------------------------------
    # ABRUPT DRIFT
    # ---------------------------------------------------------

    drifted_current, drift_metadata = (
        simulator.abrupt(
            df=current_clean,
            features=features,
            magnitude=0.30,
            start_window=(
                len(current_clean) // 2
            ),
            baseline_df=reference,
        )
    )

    drift_metrics = (
        calculate_feature_metrics(
            reference_df=reference,
            current_df=drifted_current,
            features=features,
        )
    )

    drift_dsi = calculate_dsi(
        drift_metrics
    )

    drift_severity = (
        classify_dsi(
            drift_dsi
        )
    )

    print()

    print(
        "CONTROLLED ABRUPT DRIFT"
    )

    print(
        drift_metadata.to_dict()
    )

    print_metrics(
        drift_metrics
    )

    print(
        "DSI:",
        round(
            drift_dsi,
            4,
        ),
    )

    print(
        "SEVERITY:",
        drift_severity,
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    assert np.isclose(
        clean_dsi,
        0.0,
        atol=1e-10,
    )

    assert (
        drift_dsi
        > clean_dsi
    )

    print()

    print(
        "Stable DSI:",
        round(
            clean_dsi,
            4,
        ),
    )

    print(
        "Drift DSI:",
        round(
            drift_dsi,
            4,
        ),
    )

    print(
        "DSI change:",
        round(
            drift_dsi - clean_dsi,
            4,
        ),
    )

    print()

    print("=" * 70)

    print(
        "PERSON 1 DETECTION TEST PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()