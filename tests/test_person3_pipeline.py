from pathlib import Path

import numpy as np
import pandas as pd

from src.datadecay.quality_gate import run_quality_gate
from src.datadecay.recovery import (
    create_missingness,
    recovery_quality,
    recover_missing_values,
)
from src.datadecay.synthetic import (
    generate_synthetic_data,
    synthetic_quality_score,
)
from src.datadecay.utils import (
    load_electricity_dataset,
    temporal_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)


def select_features(
    reference: pd.DataFrame,
    n_features: int = 5,
):
    numeric_columns = [
        column
        for column in reference.columns
        if column != "timestamp"
        and pd.api.types.is_numeric_dtype(
            reference[column]
        )
    ]

    variances = (
        reference[numeric_columns]
        .var()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values(
            ascending=False
        )
    )

    features = list(
        variances.head(
            n_features
        ).index
    )

    assert len(features) == n_features

    return features


def test_person3_dataset_pipeline():
    """
    End-to-end Person 3 validation using the real electricity
    dataset.

    The test covers:
    1. chronological split
    2. feature selection
    3. synthetic generation
    4. quality evaluation
    5. controlled missingness
    6. recovery
    """

    assert DATA_PATH.exists(), (
        f"Dataset not found: {DATA_PATH}"
    )

    df = load_electricity_dataset(
        DATA_PATH
    )

    assert not df.empty

    assert "timestamp" in df.columns

    assert df[
        "timestamp"
    ].is_monotonic_increasing

    reference, monitoring, evaluation = (
        temporal_split(
            df,
            train_fraction=0.60,
            monitor_fraction=0.20,
        )
    )

    assert len(reference) > 0

    assert len(monitoring) > 0

    assert len(evaluation) > 0

    assert (
        reference["timestamp"].max()
        < monitoring["timestamp"].min()
    )

    assert (
        monitoring["timestamp"].max()
        < evaluation["timestamp"].min()
    )

    features = select_features(
        reference,
        n_features=5,
    )

    reference_features = (
        reference[
            features
        ]
        .copy()
    )

    monitoring_features = (
        monitoring[
            features
        ]
        .copy()
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # Synthetic generation
    # ----------------------------------------------------------

    synthetic_result = (
        generate_synthetic_data(
            reference=reference_features,
            features=features,
            n_samples=len(
                monitoring_features
            ),
            method="bootstrap",
            random_state=42,
            noise_scale=0.01,
        )
    )

    synthetic = (
        synthetic_result.samples
    )

    assert isinstance(
        synthetic,
        pd.DataFrame,
    )

    assert synthetic.shape == (
        len(monitoring_features),
        len(features),
    )

    assert list(
        synthetic.columns
    ) == features

    assert not synthetic.isna().any().any()

    assert np.isfinite(
        synthetic.to_numpy(
            dtype=float
        )
    ).all()

    # ----------------------------------------------------------
    # Synthetic quality
    # ----------------------------------------------------------

    quality_score = (
        synthetic_quality_score(
            reference=reference_features,
            synthetic=synthetic,
            features=features,
        )
    )

    assert 0.0 <= quality_score <= 1.0

    quality_gate = run_quality_gate(
        reference=reference_features,
        candidate=synthetic,
        features=features,
        threshold=0.50,
        psi_threshold=0.50,
        wasserstein_threshold=2.0,
        correlation_difference=0.50,
    )

    assert isinstance(
        quality_gate.accepted,
        bool,
    )

    assert 0.0 <= (
        quality_gate.quality_score
    ) <= 1.0

    # ----------------------------------------------------------
    # Controlled corruption
    # ----------------------------------------------------------

    clean_monitoring = (
        monitoring[
            ["timestamp"]
            + features
        ]
        .copy()
        .reset_index(drop=True)
    )

    corrupted_monitoring = (
        create_missingness(
            clean_monitoring,
            features=features,
            missing_fraction=0.05,
            random_state=42,
        )
    )

    corrupted_count = int(
        corrupted_monitoring[
            features
        ]
        .isna()
        .sum()
        .sum()
    )

    assert corrupted_count > 0

    # ----------------------------------------------------------
    # Interpolation recovery
    # ----------------------------------------------------------

    interpolation_result = (
        recover_missing_values(
            corrupted_monitoring,
            method="interpolate",
            features=features,
        )
    )

    assert (
        interpolation_result.filled_cells
        > 0
    )

    assert (
        interpolation_result
        .remaining_missing_cells
        == 0
    )

    assert not (
        interpolation_result.recovered[
            features
        ]
        .isna()
        .any()
        .any()
    )

    # ----------------------------------------------------------
    # Synthetic recovery
    # ----------------------------------------------------------

    synthetic_result_recovery = (
        recover_missing_values(
            corrupted_monitoring,
            method="synthetic",
            features=features,
            synthetic_df=synthetic,
        )
    )

    assert (
        synthetic_result_recovery
        .filled_cells
        > 0
    )

    assert (
        synthetic_result_recovery
        .remaining_missing_cells
        == 0
    )

    assert not (
        synthetic_result_recovery
        .recovered[features]
        .isna()
        .any()
        .any()
    )

    # ----------------------------------------------------------
    # Recovery quality
    # ----------------------------------------------------------

    interpolation_quality = (
        recovery_quality(
            original=clean_monitoring,
            recovered=(
                interpolation_result
                .recovered
            ),
            corrupted=corrupted_monitoring,
            features=features,
        )
    )

    synthetic_quality = (
        recovery_quality(
            original=clean_monitoring,
            recovered=(
                synthetic_result_recovery
                .recovered
            ),
            corrupted=corrupted_monitoring,
            features=features,
        )
    )

    assert (
        interpolation_quality[
            "evaluated_cells"
        ]
        > 0
    )

    assert (
        synthetic_quality[
            "evaluated_cells"
        ]
        > 0
    )

    assert (
        interpolation_quality["mae"]
        is not None
    )

    assert (
        interpolation_quality["rmse"]
        is not None
    )

    assert (
        synthetic_quality["mae"]
        is not None
    )

    assert (
        synthetic_quality["rmse"]
        is not None
    )

    assert (
        interpolation_quality["mae"]
        >= 0
    )

    assert (
        interpolation_quality["rmse"]
        >= 0
    )

    assert (
        synthetic_quality["mae"]
        >= 0
    )

    assert (
        synthetic_quality["rmse"]
        >= 0
    )

    # ----------------------------------------------------------
    # Timestamp integrity
    # ----------------------------------------------------------

    pd.testing.assert_series_equal(
        clean_monitoring[
            "timestamp"
        ],
        interpolation_result.recovered[
            "timestamp"
        ],
        check_names=True,
    )

    pd.testing.assert_series_equal(
        clean_monitoring[
            "timestamp"
        ],
        synthetic_result_recovery.recovered[
            "timestamp"
        ],
        check_names=True,
    )


def test_person3_small_synthetic_pipeline():
    """
    Fast deterministic pipeline test using generated data.
    This avoids repeatedly loading the large real dataset.
    """

    rng = np.random.default_rng(
        42
    )

    reference = pd.DataFrame(
        {
            "MT_001": rng.normal(
                100,
                10,
                500,
            ),
            "MT_002": rng.normal(
                200,
                20,
                500,
            ),
            "MT_003": rng.normal(
                300,
                30,
                500,
            ),
        }
    )

    synthetic_result = (
        generate_synthetic_data(
            reference=reference,
            features=[
                "MT_001",
                "MT_002",
                "MT_003",
            ],
            n_samples=300,
            method="bootstrap",
            random_state=42,
        )
    )

    synthetic = (
        synthetic_result.samples
    )

    assert synthetic.shape == (
        300,
        3,
    )

    gate = run_quality_gate(
        reference=reference,
        candidate=synthetic,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
        threshold=0.50,
        psi_threshold=0.50,
        wasserstein_threshold=100.0,
        correlation_difference=0.50,
    )

    assert 0.0 <= (
        gate.quality_score
    ) <= 1.0

    original = pd.DataFrame(
        {
            "MT_001": np.arange(
                100,
                dtype=float,
            ),
            "MT_002": np.arange(
                100,
                dtype=float,
            )
            * 2,
            "MT_003": np.arange(
                100,
                dtype=float,
            )
            * 3,
        }
    )

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

    recovered = recover_missing_values(
        corrupted,
        method="interpolate",
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
    )

    assert (
        recovered.remaining_missing_cells
        == 0
    )

    quality = recovery_quality(
        original=original,
        recovered=recovered.recovered,
        corrupted=corrupted,
        features=[
            "MT_001",
            "MT_002",
            "MT_003",
        ],
    )

    assert quality[
        "evaluated_cells"
    ] > 0

    assert quality[
        "rmse"
    ] < 1e-10


if __name__ == "__main__":
    test_person3_dataset_pipeline()
    test_person3_small_synthetic_pipeline()

    print(
        "Person 3 pipeline tests defined successfully."
    )