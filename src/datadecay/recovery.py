from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class RecoveryResult:
    """
    Result produced by a recovery operation.
    """

    recovered: pd.DataFrame
    method: str
    filled_cells: int
    remaining_missing_cells: int
    recovery_rate: float
    metadata: Dict

    def to_dict(self) -> Dict:
        """
        Convert recovery metadata into a serializable dictionary.
        """

        result = asdict(self)

        result["recovered"] = {
            "rows": int(
                self.recovered.shape[0]
            ),
            "columns": int(
                self.recovered.shape[1]
            ),
        }

        return result


def _validate_dataframe(
    df: pd.DataFrame,
) -> None:
    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )


def _resolve_features(
    df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ],
) -> list:
    """
    Resolve the columns that will be recovered.
    """

    if features is None:
        resolved = [
            column
            for column in df.columns
            if pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]
    else:
        resolved = list(features)

    if not resolved:
        raise ValueError(
            "No recovery features were provided."
        )

    missing = [
        feature
        for feature in resolved
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing recovery features: {missing}"
        )

    return resolved


def _count_missing(
    df: pd.DataFrame,
    features: Sequence[str],
) -> int:
    return int(
        df[list(features)]
        .isna()
        .sum()
        .sum()
    )


def _build_result(
    original: pd.DataFrame,
    recovered: pd.DataFrame,
    features: Sequence[str],
    method: str,
    metadata: Optional[Dict] = None,
) -> RecoveryResult:
    """
    Construct a standardized RecoveryResult.
    """

    before_missing = _count_missing(
        original,
        features,
    )

    after_missing = _count_missing(
        recovered,
        features,
    )

    filled_cells = max(
        before_missing - after_missing,
        0,
    )

    if before_missing == 0:
        recovery_rate = 1.0
    else:
        recovery_rate = (
            filled_cells
            / before_missing
        )

    return RecoveryResult(
        recovered=recovered,
        method=method,
        filled_cells=int(
            filled_cells
        ),
        remaining_missing_cells=int(
            after_missing
        ),
        recovery_rate=float(
            np.clip(
                recovery_rate,
                0.0,
                1.0,
            )
        ),
        metadata=metadata or {},
    )


def forward_fill_recovery(
    df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
    limit: Optional[int] = None,
) -> RecoveryResult:
    """
    Recover missing values using forward filling.

    Existing non-missing values are preserved.
    """

    _validate_dataframe(df)

    resolved_features = _resolve_features(
        df,
        features,
    )

    if limit is not None and limit <= 0:
        raise ValueError(
            "limit must be positive or None."
        )

    recovered = df.copy()

    recovered[
        resolved_features
    ] = recovered[
        resolved_features
    ].ffill(
        limit=limit
    )

    return _build_result(
        original=df,
        recovered=recovered,
        features=resolved_features,
        method="forward_fill",
        metadata={
            "limit": limit,
            "features": resolved_features,
        },
    )


def backward_fill_recovery(
    df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
    limit: Optional[int] = None,
) -> RecoveryResult:
    """
    Recover missing values using backward filling.
    """

    _validate_dataframe(df)

    resolved_features = _resolve_features(
        df,
        features,
    )

    if limit is not None and limit <= 0:
        raise ValueError(
            "limit must be positive or None."
        )

    recovered = df.copy()

    recovered[
        resolved_features
    ] = recovered[
        resolved_features
    ].bfill(
        limit=limit
    )

    return _build_result(
        original=df,
        recovered=recovered,
        features=resolved_features,
        method="backward_fill",
        metadata={
            "limit": limit,
            "features": resolved_features,
        },
    )


def interpolate_recovery(
    df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
    method: str = "linear",
    limit: Optional[int] = None,
) -> RecoveryResult:
    """
    Recover missing values using interpolation.

    Linear interpolation is the default and is appropriate for
    the project's ordered time-series measurements.
    """

    _validate_dataframe(df)

    resolved_features = _resolve_features(
        df,
        features,
    )

    if limit is not None and limit <= 0:
        raise ValueError(
            "limit must be positive or None."
        )

    recovered = df.copy()

    recovered[
        resolved_features
    ] = recovered[
        resolved_features
    ].interpolate(
        method=method,
        limit=limit,
        limit_direction="both",
    )

    return _build_result(
        original=df,
        recovered=recovered,
        features=resolved_features,
        method="interpolation",
        metadata={
            "interpolation_method": method,
            "limit": limit,
            "features": resolved_features,
        },
    )


def median_recovery(
    df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
) -> RecoveryResult:
    """
    Recover missing values using the median of each feature.
    """

    _validate_dataframe(df)

    resolved_features = _resolve_features(
        df,
        features,
    )

    recovered = df.copy()

    for feature in resolved_features:
        median = (
            recovered[feature]
            .median()
        )

        if pd.isna(median):
            continue

        recovered[feature] = (
            recovered[feature]
            .fillna(median)
        )

    return _build_result(
        original=df,
        recovered=recovered,
        features=resolved_features,
        method="median",
        metadata={
            "features": resolved_features,
        },
    )


def mean_recovery(
    df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
) -> RecoveryResult:
    """
    Recover missing values using the mean of each feature.
    """

    _validate_dataframe(df)

    resolved_features = _resolve_features(
        df,
        features,
    )

    recovered = df.copy()

    for feature in resolved_features:
        mean = (
            recovered[feature]
            .mean()
        )

        if pd.isna(mean):
            continue

        recovered[feature] = (
            recovered[feature]
            .fillna(mean)
        )

    return _build_result(
        original=df,
        recovered=recovered,
        features=resolved_features,
        method="mean",
        metadata={
            "features": resolved_features,
        },
    )


def synthetic_recovery(
    df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
) -> RecoveryResult:
    """
    Recover missing values using synthetic observations.

    For each feature, missing positions are filled by values from
    the corresponding synthetic feature. Synthetic values are
    consumed in deterministic row order.
    """

    _validate_dataframe(df)
    _validate_dataframe(synthetic_df)

    resolved_features = _resolve_features(
        df,
        features,
    )

    missing_synthetic = [
        feature
        for feature in resolved_features
        if feature not in synthetic_df.columns
    ]

    if missing_synthetic:
        raise ValueError(
            "Synthetic dataframe is missing "
            f"features: {missing_synthetic}"
        )

    if len(synthetic_df) == 0:
        raise ValueError(
            "Synthetic dataframe contains no rows."
        )

    recovered = df.copy()

    total_filled = 0

    for feature in resolved_features:
        missing_mask = (
            recovered[feature]
            .isna()
        )

        missing_count = int(
            missing_mask.sum()
        )

        if missing_count == 0:
            continue

        synthetic_values = (
            pd.to_numeric(
                synthetic_df[feature],
                errors="coerce",
            )
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        if len(synthetic_values) == 0:
            continue

        repetitions = int(
            np.ceil(
                missing_count
                / len(synthetic_values)
            )
        )

        replacement_values = np.tile(
            synthetic_values,
            repetitions,
        )[:missing_count]

        recovered.loc[
            missing_mask,
            feature
        ] = replacement_values

        total_filled += (
            len(replacement_values)
        )

    return _build_result(
        original=df,
        recovered=recovered,
        features=resolved_features,
        method="synthetic",
        metadata={
            "synthetic_rows": int(
                len(synthetic_df)
            ),
            "features": resolved_features,
            "filled_by_method": int(
                total_filled
            ),
        },
    )


def recover_missing_values(
    df: pd.DataFrame,
    method: str = "interpolate",
    features: Optional[
        Sequence[str]
    ] = None,
    synthetic_df: Optional[
        pd.DataFrame
    ] = None,
    **kwargs,
) -> RecoveryResult:
    """
    Dispatch missing-value recovery to the selected method.

    Supported methods:
        - forward_fill
        - backward_fill
        - interpolate
        - median
        - mean
        - synthetic
    """

    method = method.lower().strip()

    if method == "forward_fill":
        return forward_fill_recovery(
            df,
            features=features,
            **kwargs,
        )

    if method == "backward_fill":
        return backward_fill_recovery(
            df,
            features=features,
            **kwargs,
        )

    if method == "interpolate":
        return interpolate_recovery(
            df,
            features=features,
            **kwargs,
        )

    if method == "median":
        return median_recovery(
            df,
            features=features,
        )

    if method == "mean":
        return mean_recovery(
            df,
            features=features,
        )

    if method == "synthetic":
        if synthetic_df is None:
            raise ValueError(
                "synthetic_df is required when "
                "method='synthetic'."
            )

        return synthetic_recovery(
            df=df,
            synthetic_df=synthetic_df,
            features=features,
        )

    raise ValueError(
        "Unknown recovery method: "
        f"{method}"
    )


def create_missingness(
    df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
    missing_fraction: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a controlled missing-value corruption.

    The original dataframe is not modified.
    """

    _validate_dataframe(df)

    if not (
        0.0
        <= missing_fraction
        <= 1.0
    ):
        raise ValueError(
            "missing_fraction must be "
            "between 0 and 1."
        )

    resolved_features = _resolve_features(
        df,
        features,
    )

    corrupted = df.copy()

    rng = np.random.default_rng(
        random_state
    )

    for feature in resolved_features:
        valid_indices = (
            corrupted.index[
                corrupted[feature]
                .notna()
            ]
            .to_numpy()
        )

        if len(valid_indices) == 0:
            continue

        n_missing = int(
            round(
                len(valid_indices)
                * missing_fraction
            )
        )

        if n_missing == 0:
            continue

        selected = rng.choice(
            valid_indices,
            size=n_missing,
            replace=False,
        )

        corrupted.loc[
            selected,
            feature
        ] = np.nan

    return corrupted


def recovery_quality(
    original: pd.DataFrame,
    recovered: pd.DataFrame,
    corrupted: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
) -> Dict:
    """
    Evaluate recovery quality when the clean original dataframe
    is available.

    Metrics are calculated only over cells that were corrupted.
    """

    _validate_dataframe(original)
    _validate_dataframe(recovered)
    _validate_dataframe(corrupted)

    if (
        len(original)
        != len(recovered)
        or len(original)
        != len(corrupted)
    ):
        raise ValueError(
            "original, recovered, and corrupted "
            "must have the same number of rows."
        )

    resolved_features = _resolve_features(
        original,
        features,
    )

    total_corrupted = 0
    absolute_errors = []
    squared_errors = []

    for feature in resolved_features:
        corruption_mask = (
            corrupted[feature]
            .isna()
            & original[feature].notna()
        )

        if not corruption_mask.any():
            continue

        truth = original.loc[
            corruption_mask,
            feature,
        ].to_numpy(
            dtype=float
        )

        prediction = recovered.loc[
            corruption_mask,
            feature,
        ].to_numpy(
            dtype=float
        )

        valid = np.isfinite(
            truth
        ) & np.isfinite(
            prediction
        )

        truth = truth[valid]
        prediction = prediction[valid]

        if len(truth) == 0:
            continue

        errors = (
            prediction
            - truth
        )

        absolute_errors.extend(
            np.abs(errors).tolist()
        )

        squared_errors.extend(
            np.square(errors).tolist()
        )

        total_corrupted += len(
            truth
        )

    if total_corrupted == 0:
        return {
            "corrupted_cells": 0,
            "evaluated_cells": 0,
            "mae": None,
            "rmse": None,
        }

    mae = float(
        np.mean(
            absolute_errors
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                squared_errors
            )
        )
    )

    return {
        "corrupted_cells": int(
            total_corrupted
        ),
        "evaluated_cells": int(
            total_corrupted
        ),
        "mae": mae,
        "rmse": rmse,
    }


def compare_recovery_methods(
    original: pd.DataFrame,
    corrupted: pd.DataFrame,
    methods: Sequence[str],
    features: Optional[
        Sequence[str]
    ] = None,
    synthetic_df: Optional[
        pd.DataFrame
    ] = None,
) -> pd.DataFrame:
    """
    Compare multiple recovery methods against the known original
    data.
    """

    rows = []

    for method in methods:
        result = recover_missing_values(
            corrupted,
            method=method,
            features=features,
            synthetic_df=synthetic_df,
        )

        quality = recovery_quality(
            original=original,
            recovered=result.recovered,
            corrupted=corrupted,
            features=features,
        )

        rows.append(
            {
                "method": method,
                "filled_cells": (
                    result.filled_cells
                ),
                "remaining_missing_cells": (
                    result.remaining_missing_cells
                ),
                "recovery_rate": (
                    result.recovery_rate
                ),
                "mae": quality["mae"],
                "rmse": quality["rmse"],
            }
        )

    return pd.DataFrame(
        rows
    )


def recovery_summary(
    result: RecoveryResult,
) -> Dict:
    """
    Return a compact summary for reports.
    """

    return {
        "method": result.method,
        "filled_cells": int(
            result.filled_cells
        ),
        "remaining_missing_cells": int(
            result.remaining_missing_cells
        ),
        "recovery_rate": float(
            result.recovery_rate
        ),
        "metadata": result.metadata,
    }