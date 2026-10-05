import numpy as np
import pandas as pd

from src.datadecay.recovery import (
    backward_fill_recovery,
    compare_recovery_methods,
    create_missingness,
    forward_fill_recovery,
    interpolate_recovery,
    mean_recovery,
    median_recovery,
    recovery_quality,
    recover_missing_values,
    synthetic_recovery,
)


def make_time_series_data(
    rows=200,
):
    timestamps = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="15min",
    )

    x = np.arange(
        rows,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "MT_001": 100 + x,
            "MT_002": 200 + (x * 2),
            "MT_003": 300 + (x * 3),
        }
    )


def make_synthetic_data(
    rows=200,
):
    return pd.DataFrame(
        {
            "MT_001": np.linspace(
                100,
                300,
                rows,
            ),
            "MT_002": np.linspace(
                200,
                500,
                rows,
            ),
            "MT_003": np.linspace(
                300,
                700,
                rows,
            ),
        }
    )


def test_create_missingness():
    original = make_time_series_data()

    corrupted = create_missingness(
        original,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
        missing_fraction=0.10,
        random_state=42,
    )

    assert len(corrupted) == len(
        original
    )

    assert (
        corrupted["timestamp"]
        .equals(
            original["timestamp"]
        )
    )

    missing_count = int(
        corrupted[
            [
                "MT_001",
                "MT_002",
                "MT_003",
            ]
        ]
        .isna()
        .sum()
        .sum()
    )

    assert missing_count > 0


def test_create_missingness_is_reproducible():
    original = make_time_series_data()

    first = create_missingness(
        original,
        features=[
            "MT_001",
            "MT_002",
        ],
        missing_fraction=0.10,
        random_state=42,
    )

    second = create_missingness(
        original,
        features=[
            "MT_001",
            "MT_002",
        ],
        missing_fraction=0.10,
        random_state=42,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_forward_fill_recovery():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        10:15,
        "MT_001",
    ] = np.nan

    result = forward_fill_recovery(
        corrupted,
        features=[
            "MT_001"
        ],
    )

    assert result.filled_cells > 0

    assert (
        result.remaining_missing_cells
        == 0
    )

    assert not result.recovered[
        "MT_001"
    ].isna().any()

    assert (
        result.recovered[
            "timestamp"
        ].equals(
            original["timestamp"]
        )
    )


def test_backward_fill_recovery():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        10:15,
        "MT_001",
    ] = np.nan

    result = backward_fill_recovery(
        corrupted,
        features=[
            "MT_001"
        ],
    )

    assert result.filled_cells > 0

    assert (
        result.remaining_missing_cells
        == 0
    )

    assert not result.recovered[
        "MT_001"
    ].isna().any()


def test_interpolation_recovery():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        50:60,
        "MT_001",
    ] = np.nan

    result = interpolate_recovery(
        corrupted,
        features=[
            "MT_001"
        ],
    )

    assert result.filled_cells > 0

    assert (
        result.remaining_missing_cells
        == 0
    )

    assert not result.recovered[
        "MT_001"
    ].isna().any()


def test_median_recovery():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        20:30,
        "MT_001",
    ] = np.nan

    result = median_recovery(
        corrupted,
        features=[
            "MT_001"
        ],
    )

    assert result.filled_cells > 0

    assert (
        result.remaining_missing_cells
        == 0
    )

    assert not result.recovered[
        "MT_001"
    ].isna().any()


def test_mean_recovery():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        20:30,
        "MT_002",
    ] = np.nan

    result = mean_recovery(
        corrupted,
        features=[
            "MT_002"
        ],
    )

    assert result.filled_cells > 0

    assert (
        result.remaining_missing_cells
        == 0
    )

    assert not result.recovered[
        "MT_002"
    ].isna().any()


def test_synthetic_recovery():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        10:20,
        "MT_001",
    ] = np.nan

    synthetic = make_synthetic_data()

    result = synthetic_recovery(
        corrupted,
        synthetic,
        features=[
            "MT_001"
        ],
    )

    assert result.filled_cells > 0

    assert (
        result.remaining_missing_cells
        == 0
    )

    assert not result.recovered[
        "MT_001"
    ].isna().any()


def test_generic_recovery_dispatch():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        20:30,
        "MT_001",
    ] = np.nan

    result = recover_missing_values(
        corrupted,
        method="interpolate",
        features=[
            "MT_001"
        ],
    )

    assert result.method == (
        "interpolation"
    )

    assert (
        result.remaining_missing_cells
        == 0
    )


def test_synthetic_dispatch_requires_data():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        20:30,
        "MT_001",
    ] = np.nan

    try:
        recover_missing_values(
            corrupted,
            method="synthetic",
            features=[
                "MT_001"
            ],
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:
        pass


def test_recovery_preserves_existing_values():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        30:40,
        "MT_001",
    ] = np.nan

    result = interpolate_recovery(
        corrupted,
        features=[
            "MT_001"
        ],
    )

    non_missing_mask = (
        corrupted["MT_001"]
        .notna()
    )

    assert np.allclose(
        result.recovered.loc[
            non_missing_mask,
            "MT_001",
        ].to_numpy(),
        corrupted.loc[
            non_missing_mask,
            "MT_001",
        ].to_numpy(),
    )


def test_recovery_preserves_timestamp():
    original = make_time_series_data()

    corrupted = create_missingness(
        original,
        features=[
            "MT_001",
            "MT_002",
        ],
        missing_fraction=0.10,
        random_state=42,
    )

    result = interpolate_recovery(
        corrupted,
        features=[
            "MT_001",
            "MT_002",
        ],
    )

    assert (
        result.recovered[
            "timestamp"
        ].equals(
            original["timestamp"]
        )
    )


def test_recovery_quality():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        50:60,
        "MT_001",
    ] = np.nan

    recovered = interpolate_recovery(
        corrupted,
        features=[
            "MT_001"
        ],
    )

    quality = recovery_quality(
        original=original,
        recovered=recovered.recovered,
        corrupted=corrupted,
        features=[
            "MT_001"
        ],
    )

    assert (
        quality["corrupted_cells"]
        > 0
    )

    assert (
        quality["evaluated_cells"]
        > 0
    )

    assert quality["mae"] is not None

    assert quality["rmse"] is not None

    assert quality["mae"] >= 0

    assert quality["rmse"] >= 0


def test_linear_series_interpolation_is_accurate():
    original = make_time_series_data()

    corrupted = original.copy()

    corrupted.loc[
        50:60,
        "MT_001",
    ] = np.nan

    result = interpolate_recovery(
        corrupted,
        features=[
            "MT_001"
        ],
    )

    quality = recovery_quality(
        original=original,
        recovered=result.recovered,
        corrupted=corrupted,
        features=[
            "MT_001"
        ],
    )

    assert quality["mae"] < 1e-10

    assert quality["rmse"] < 1e-10


def test_compare_recovery_methods():
    original = make_time_series_data()

    corrupted = create_missingness(
        original,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
        missing_fraction=0.10,
        random_state=42,
    )

    synthetic = make_synthetic_data()

    comparison = compare_recovery_methods(
        original=original,
        corrupted=corrupted,
        methods=[
            "forward_fill",
            "backward_fill",
            "interpolate",
            "median",
            "mean",
            "synthetic",
        ],
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
        synthetic_df=synthetic,
    )

    assert not comparison.empty

    assert len(comparison) == 6

    assert "method" in (
        comparison.columns
    )

    assert "mae" in (
        comparison.columns
    )

    assert "rmse" in (
        comparison.columns
    )

    assert comparison[
        "filled_cells"
    ].ge(0).all()


def test_invalid_missing_fraction():
    original = make_time_series_data()

    try:
        create_missingness(
            original,
            missing_fraction=1.5,
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:
        pass


def test_invalid_recovery_method():
    original = make_time_series_data()

    try:
        recover_missing_values(
            original,
            method="unknown_method",
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:
        pass


def test_missing_feature_raises():
    original = make_time_series_data()

    try:
        interpolate_recovery(
            original,
            features=[
                "DOES_NOT_EXIST"
            ],
        )

        raise AssertionError(
            "Expected ValueError was not raised."
        )

    except ValueError:
        pass


if __name__ == "__main__":
    test_create_missingness()
    test_create_missingness_is_reproducible()
    test_forward_fill_recovery()
    test_backward_fill_recovery()
    test_interpolation_recovery()
    test_median_recovery()
    test_mean_recovery()
    test_synthetic_recovery()
    test_generic_recovery_dispatch()
    test_synthetic_dispatch_requires_data()
    test_recovery_preserves_existing_values()
    test_recovery_preserves_timestamp()
    test_recovery_quality()
    test_linear_series_interpolation_is_accurate()
    test_compare_recovery_methods()
    test_invalid_missing_fraction()
    test_invalid_recovery_method()
    test_missing_feature_raises()

    print(
        "Person 3 recovery tests defined successfully."
    )