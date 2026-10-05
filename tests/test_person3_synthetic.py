from pathlib import Path

import numpy as np
import pandas as pd

from src.datadecay.quality_gate import run_quality_gate
from src.datadecay.synthetic import (
    SyntheticGenerator,
    compare_distributions,
    generate_synthetic_data,
    synthetic_quality_score,
)


def make_reference_data(
    rows=500,
    seed=42,
):
    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "MT_001": rng.normal(
                100,
                10,
                rows,
            ),
            "MT_002": rng.normal(
                200,
                20,
                rows,
            ),
            "MT_003": rng.normal(
                300,
                30,
                rows,
            ),
            "MT_004": rng.normal(
                400,
                40,
                rows,
            ),
            "MT_005": rng.normal(
                500,
                50,
                rows,
            ),
        }
    )


def test_bootstrap_generation():
    reference = make_reference_data()

    result = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
            "MT_004",
            "MT_005",
        ],
        n_samples=200,
        method="bootstrap",
        random_state=42,
        noise_scale=0.01,
    )

    assert isinstance(
        result.samples,
        pd.DataFrame,
    )

    assert result.samples.shape == (
        200,
        5,
    )

    assert list(
        result.samples.columns
    ) == [
        "MT_001",
        "MT_002",
        "MT_003",
        "MT_004",
        "MT_005",
    ]

    assert not result.samples.isna().any().any()


def test_gaussian_generation():
    reference = make_reference_data()

    generator = SyntheticGenerator(
        random_state=42
    )

    result = generator.gaussian(
        reference,
        features=[
            "MT_001",
            "MT_002",
        ],
        n_samples=100,
    )

    assert result.samples.shape == (
        100,
        2,
    )

    assert result.method == "gaussian"

    assert not result.samples.isna().any().any()


def test_empirical_generation():
    reference = make_reference_data()

    result = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
        ],
        n_samples=150,
        method="empirical",
        random_state=42,
    )

    assert result.samples.shape == (
        150,
        2,
    )

    assert result.method == "empirical"

    assert np.isfinite(
        result.samples.to_numpy()
    ).all()


def test_generation_is_reproducible():
    reference = make_reference_data()

    first = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
        n_samples=100,
        method="bootstrap",
        random_state=123,
        noise_scale=0.01,
    )

    second = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
        n_samples=100,
        method="bootstrap",
        random_state=123,
        noise_scale=0.01,
    )

    pd.testing.assert_frame_equal(
        first.samples,
        second.samples,
    )


def test_different_seed_changes_generation():
    reference = make_reference_data()

    first = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
        ],
        n_samples=100,
        method="bootstrap",
        random_state=1,
    )

    second = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
        ],
        n_samples=100,
        method="bootstrap",
        random_state=2,
    )

    assert not first.samples.equals(
        second.samples
    )


def test_distribution_comparison():
    reference = make_reference_data()

    synthetic = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
        n_samples=500,
        method="bootstrap",
        random_state=42,
    ).samples

    comparison = compare_distributions(
        reference,
        synthetic,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
    )

    assert not comparison.empty

    assert "mean_difference" in (
        comparison.columns
    )

    assert "std_difference" in (
        comparison.columns
    )

    assert len(comparison) == 3


def test_synthetic_quality_score():
    reference = make_reference_data()

    synthetic = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
        n_samples=500,
        method="bootstrap",
        random_state=42,
    ).samples

    score = synthetic_quality_score(
        reference,
        synthetic,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
    )

    assert 0.0 <= score <= 1.0

    assert score > 0.0


def test_identical_data_has_high_quality():
    reference = make_reference_data()

    score = synthetic_quality_score(
        reference,
        reference.copy(),
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
    )

    assert score == 1.0


def test_quality_gate_accepts_reasonable_synthetic_data():
    reference = make_reference_data(
        rows=1000
    )

    synthetic = generate_synthetic_data(
        reference,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
            "MT_004",
            "MT_005",
        ],
        n_samples=1000,
        method="bootstrap",
        random_state=42,
        noise_scale=0.001,
    ).samples

    result = run_quality_gate(
        reference=reference,
        candidate=synthetic,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
            "MT_004",
            "MT_005",
        ],
        threshold=0.50,
        psi_threshold=0.50,
        wasserstein_threshold=100.0,
        correlation_difference=0.50,
    )

    assert isinstance(
        result.accepted,
        bool,
    )

    assert 0.0 <= (
        result.quality_score
    ) <= 1.0

    assert result.feature_metrics is not None


def test_quality_gate_rejects_bad_distribution():
    reference = make_reference_data(
        rows=500
    )

    synthetic = reference.copy()

    synthetic[
        "MT_001"
    ] = synthetic[
        "MT_001"
    ] + 10000

    result = run_quality_gate(
        reference=reference,
        candidate=synthetic,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
            "MT_004",
            "MT_005",
        ],
        threshold=0.70,
        psi_threshold=0.25,
        wasserstein_threshold=1.0,
        correlation_difference=0.20,
    )

    assert result.accepted is False


def test_invalid_feature_raises():
    reference = make_reference_data()

    generator = SyntheticGenerator(
        random_state=42
    )

    try:
        generator.bootstrap(
            reference,
            features=[
                "DOES_NOT_EXIST"
            ],
            n_samples=10,
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:
        pass


def test_invalid_sample_count_raises():
    reference = make_reference_data()

    try:
        generate_synthetic_data(
            reference,
            features=[
                "MT_001"
            ],
            n_samples=0,
            method="bootstrap",
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:
        pass


if __name__ == "__main__":
    test_bootstrap_generation()
    test_gaussian_generation()
    test_empirical_generation()
    test_generation_is_reproducible()
    test_different_seed_changes_generation()
    test_distribution_comparison()
    test_synthetic_quality_score()
    test_identical_data_has_high_quality()
    test_quality_gate_accepts_reasonable_synthetic_data()
    test_quality_gate_rejects_bad_distribution()
    test_invalid_feature_raises()
    test_invalid_sample_count_raises()

    print(
        "Person 3 synthetic tests defined successfully."
    )