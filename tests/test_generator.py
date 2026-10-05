"""
Tests for the legacy synthetic-data generator.
"""

import numpy as np
import pandas as pd

from src.synthetic.generator import (
    SyntheticGenerationResult,
    generate_block_bootstrap,
    generate_bootstrap,
    generate_perturbed,
    generate_synthetic_data,
)


def make_reference_dataframe(
    rows=500,
):
    """Create deterministic reference data for testing."""

    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01",
                periods=rows,
                freq="15min",
            ),
            "feature_1": rng.normal(
                100,
                10,
                rows,
            ),
            "feature_2": rng.normal(
                50,
                5,
                rows,
            ),
            "feature_3": rng.normal(
                200,
                20,
                rows,
            ),
        }
    )


def test_bootstrap_generation():
    """Bootstrap generation should produce the requested shape."""

    reference = make_reference_dataframe()

    result = generate_bootstrap(
        reference_df=reference,
        n_rows=250,
        features=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        seed=42,
    )

    assert isinstance(
        result,
        SyntheticGenerationResult,
    )

    assert len(result.data) == 250

    assert list(
        result.data.columns
    ) == [
        "timestamp",
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    assert result.method == "bootstrap"
    assert result.seed == 42
    assert result.source_rows == 500
    assert result.generated_rows == 250


def test_bootstrap_preserves_numeric_schema():
    """Generated feature columns should remain numeric."""

    reference = make_reference_dataframe()

    result = generate_bootstrap(
        reference,
        n_rows=100,
        features=[
            "feature_1",
            "feature_2",
        ],
        seed=42,
    )

    assert pd.api.types.is_numeric_dtype(
        result.data["feature_1"]
    )

    assert pd.api.types.is_numeric_dtype(
        result.data["feature_2"]
    )


def test_bootstrap_reproducibility():
    """The same seed should produce identical synthetic values."""

    reference = make_reference_dataframe()

    result_a = generate_bootstrap(
        reference,
        n_rows=200,
        features=[
            "feature_1",
            "feature_2",
        ],
        seed=123,
    )

    result_b = generate_bootstrap(
        reference,
        n_rows=200,
        features=[
            "feature_1",
            "feature_2",
        ],
        seed=123,
    )

    pd.testing.assert_frame_equal(
        result_a.data,
        result_b.data,
    )


def test_different_seeds_can_change_output():
    """Different seeds should produce different bootstrap samples."""

    reference = make_reference_dataframe()

    result_a = generate_bootstrap(
        reference,
        n_rows=200,
        features=["feature_1"],
        seed=1,
    )

    result_b = generate_bootstrap(
        reference,
        n_rows=200,
        features=["feature_1"],
        seed=2,
    )

    assert not np.array_equal(
        result_a.data[
            "feature_1"
        ].to_numpy(),
        result_b.data[
            "feature_1"
        ].to_numpy(),
    )


def test_perturbation_generation():
    """Perturbation generation should produce finite numeric values."""

    reference = make_reference_dataframe()

    result = generate_perturbed(
        reference,
        n_rows=300,
        features=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        seed=42,
        noise_scale=0.01,
    )

    assert len(result.data) == 300

    values = result.data[
        [
            "feature_1",
            "feature_2",
            "feature_3",
        ]
    ].to_numpy()

    assert np.isfinite(
        values
    ).all()


def test_block_bootstrap_generation():
    """Block bootstrap should preserve chronological blocks."""

    reference = make_reference_dataframe()

    result = generate_block_bootstrap(
        reference,
        n_rows=250,
        features=[
            "feature_1",
            "feature_2",
        ],
        seed=42,
        block_size=25,
    )

    assert len(result.data) == 250

    assert list(
        result.data.columns
    ) == [
        "timestamp",
        "feature_1",
        "feature_2",
    ]

    values = result.data[
        [
            "feature_1",
            "feature_2",
        ]
    ].to_numpy()

    assert np.isfinite(
        values
    ).all()


def test_generated_timestamps_are_future_ordered():
    """Synthetic timestamps should continue after the reference period."""

    reference = make_reference_dataframe()

    result = generate_bootstrap(
        reference,
        n_rows=100,
        features=["feature_1"],
        seed=42,
    )

    timestamps = result.data[
        "timestamp"
    ]

    reference_last = reference[
        "timestamp"
    ].iloc[-1]

    assert timestamps.iloc[0] > reference_last

    assert timestamps.is_monotonic_increasing

    differences = timestamps.diff().dropna()

    assert (
        differences == pd.Timedelta(
            minutes=15
        )
    ).all()


def test_default_row_count():
    """When n_rows is omitted, reference row count is used."""

    reference = make_reference_dataframe(
        rows=300
    )

    result = generate_synthetic_data(
        reference_df=reference,
        features=["feature_1"],
        method="bootstrap",
        seed=42,
    )

    assert len(result.data) == 300


def test_multiple_methods():
    """All supported generation methods should work."""

    reference = make_reference_dataframe()

    methods = [
        "bootstrap",
        "perturbation",
        "block_bootstrap",
    ]

    for method in methods:

        result = generate_synthetic_data(
            reference_df=reference,
            n_rows=100,
            features=[
                "feature_1",
                "feature_2",
            ],
            method=method,
            seed=42,
            block_size=20,
        )

        assert len(result.data) == 100

        assert result.method == method


def test_invalid_method():
    """Unsupported methods should raise ValueError."""

    reference = make_reference_dataframe()

    try:
        generate_synthetic_data(
            reference_df=reference,
            n_rows=100,
            features=["feature_1"],
            method="unknown_method",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for unsupported method."
        )


def test_invalid_feature():
    """Missing features should raise ValueError."""

    reference = make_reference_dataframe()

    try:
        generate_bootstrap(
            reference,
            n_rows=100,
            features=["missing_feature"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for missing feature."
        )


def test_invalid_row_count():
    """Non-positive row counts should raise ValueError."""

    reference = make_reference_dataframe()

    try:
        generate_bootstrap(
            reference,
            n_rows=0,
            features=["feature_1"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for n_rows=0."
        )


def test_metadata():
    """Generation metadata should describe the generated dataset."""

    reference = make_reference_dataframe()

    result = generate_bootstrap(
        reference,
        n_rows=75,
        features=[
            "feature_1",
            "feature_2",
        ],
        seed=42,
    )

    metadata = result.to_dict()

    assert metadata["method"] == "bootstrap"
    assert metadata["seed"] == 42
    assert metadata["source_rows"] == 500
    assert metadata["generated_rows"] == 75

    assert metadata["features"] == [
        "feature_1",
        "feature_2",
    ]