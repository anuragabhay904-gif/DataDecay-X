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
    DriftSimulator,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)


FEATURES = [
    "MT_001",
    "MT_002",
    "MT_003",
    "MT_004",
    "MT_005",
]


def check_common_integrity(
    original,
    modified,
):
    assert (
        original.shape
        == modified.shape
    )

    assert (
        original["timestamp"].equals(
            modified["timestamp"]
        )
    )


def check_untouched_features(
    original,
    modified,
    affected_features,
):
    for column in original.columns:

        if column == "timestamp":
            continue

        if column in affected_features:
            continue

        assert np.allclose(
            original[column].to_numpy(),
            modified[column].to_numpy(),
            equal_nan=True,
        )


def get_changed_features(
    original,
    modified,
):
    changed = []

    for column in original.columns:

        if column == "timestamp":
            continue

        if not np.allclose(
            original[column].to_numpy(),
            modified[column].to_numpy(),
            equal_nan=True,
        ):
            changed.append(column)

    return changed


def main():

    print("=" * 70)

    print(
        "DataDecay-X | Person 1 Drift Simulator Test"
    )

    print("=" * 70)

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

    simulator = DriftSimulator(
        seed=42
    )

    # ---------------------------------------------------------
    # STABLE
    # ---------------------------------------------------------

    stable, metadata = (
        simulator.stable(
            df=df,
            features=FEATURES,
        )
    )

    check_common_integrity(
        df,
        stable,
    )

    check_untouched_features(
        df,
        stable,
        [],
    )

    changed = get_changed_features(
        df,
        stable,
    )

    assert changed == []

    print(
        "Stable scenario: PASS"
    )

    # ---------------------------------------------------------
    # GRADUAL
    # ---------------------------------------------------------

    gradual, metadata = (
        simulator.gradual(
            df=df,
            features=FEATURES,
            magnitude=0.30,
            start_window=50000,
            end_window=100000,
        )
    )

    check_common_integrity(
        df,
        gradual,
    )

    check_untouched_features(
        df,
        gradual,
        FEATURES,
    )

    changed = get_changed_features(
        df,
        gradual,
    )

    assert set(changed) == set(
        FEATURES
    )

    assert (
        metadata.scenario
        == "gradual"
    )

    print(
        "Gradual scenario: PASS"
    )

    # ---------------------------------------------------------
    # ABRUPT
    # ---------------------------------------------------------

    abrupt, metadata = (
        simulator.abrupt(
            df=df,
            features=FEATURES,
            magnitude=0.30,
            start_window=70000,
        )
    )

    check_common_integrity(
        df,
        abrupt,
    )

    check_untouched_features(
        df,
        abrupt,
        FEATURES,
    )

    changed = get_changed_features(
        df,
        abrupt,
    )

    assert set(changed) == set(
        FEATURES
    )

    assert (
        metadata.scenario
        == "abrupt"
    )

    print(
        "Abrupt scenario: PASS"
    )

    # ---------------------------------------------------------
    # RECURRING
    # ---------------------------------------------------------

    recurring, metadata = (
        simulator.recurring(
            df=df,
            features=FEATURES,
            magnitude=0.30,
            cycle_length=1000,
        )
    )

    check_common_integrity(
        df,
        recurring,
    )

    check_untouched_features(
        df,
        recurring,
        FEATURES,
    )

    changed = get_changed_features(
        df,
        recurring,
    )

    assert set(changed) == set(
        FEATURES
    )

    assert (
        metadata.scenario
        == "recurring"
    )

    print(
        "Recurring scenario: PASS"
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "ALL DRIFT SIMULATOR TESTS PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()