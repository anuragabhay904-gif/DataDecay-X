"""
Dataset validation utilities for DataDecay-X.

This module validates structure, timestamps, numeric features,
missing values, and chronological ordering before data enters
the experiment pipeline.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Container for dataset validation results."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    row_count: int
    column_count: int
    feature_count: int

    def to_dict(self):
        """Return validation results as a dictionary."""

        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "feature_count": self.feature_count,
        }


def validate_dataframe(
    df,
    timestamp_column="timestamp",
    require_timestamp=True,
    require_numeric_features=True,
    allow_missing=False,
):
    """
    Validate a DataDecay-X dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to validate.

    timestamp_column : str
        Name of the timestamp column.

    require_timestamp : bool
        Whether a timestamp column is required.

    require_numeric_features : bool
        Whether at least one numeric feature is required.

    allow_missing : bool
        Whether missing numeric values are allowed.

    Returns
    -------
    ValidationResult
    """

    errors = []
    warnings = []

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        return ValidationResult(
            valid=False,
            errors=[
                "Input must be a pandas DataFrame."
            ],
            warnings=[],
            row_count=0,
            column_count=0,
            feature_count=0,
        )

    row_count = len(df)
    column_count = len(df.columns)

    if df.empty:
        errors.append(
            "DataFrame is empty."
        )

    if df.columns.duplicated().any():
        duplicates = (
            df.columns[
                df.columns.duplicated()
            ]
            .tolist()
        )

        errors.append(
            "Duplicate column names found: "
            f"{duplicates}"
        )

    if require_timestamp:

        if timestamp_column not in df.columns:
            errors.append(
                f"Required timestamp column "
                f"'{timestamp_column}' is missing."
            )

        else:

            timestamps = pd.to_datetime(
                df[timestamp_column],
                errors="coerce",
            )

            invalid_timestamp_count = int(
                timestamps.isna().sum()
            )

            if invalid_timestamp_count > 0:
                errors.append(
                    f"{invalid_timestamp_count} invalid "
                    "timestamp values found."
                )

            if (
                invalid_timestamp_count == 0
                and not timestamps.is_monotonic_increasing
            ):
                errors.append(
                    "Timestamps are not in chronological order."
                )

            duplicate_timestamps = int(
                timestamps.duplicated().sum()
            )

            if duplicate_timestamps > 0:
                warnings.append(
                    f"{duplicate_timestamps} duplicate "
                    "timestamps found."
                )

    feature_columns = [
        column
        for column in df.columns
        if column != timestamp_column
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    feature_count = len(
        feature_columns
    )

    if (
        require_numeric_features
        and feature_count == 0
    ):
        errors.append(
            "No numeric feature columns were found."
        )

    if feature_columns:

        missing_counts = (
            df[feature_columns]
            .isna()
            .sum()
        )

        total_missing = int(
            missing_counts.sum()
        )

        if total_missing > 0:

            if allow_missing:
                warnings.append(
                    f"{total_missing} missing numeric "
                    "values found."
                )
            else:
                errors.append(
                    f"{total_missing} missing numeric "
                    "values found."
                )

        infinite_counts = {}

        for column in feature_columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            invalid = int(
                (~np.isfinite(values)).sum()
            )

            if invalid > 0:
                infinite_counts[column] = (
                    invalid
                )

        if infinite_counts:
            errors.append(
                "NaN or infinite numeric values found: "
                f"{infinite_counts}"
            )

    valid = len(errors) == 0

    return ValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        row_count=row_count,
        column_count=column_count,
        feature_count=feature_count,
    )


def validate_temporal_frequency(
    df,
    timestamp_column="timestamp",
    expected_minutes=None,
):
    """
    Validate the timestamp frequency.

    Parameters
    ----------
    expected_minutes : int, optional
        Expected interval in minutes.

    Returns
    -------
    dict
        Frequency validation information.
    """

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' "
            "was not found."
        )

    timestamps = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Cannot calculate timestamp frequency "
            "because invalid timestamps exist."
        )

    if len(timestamps) < 2:
        return {
            "valid": True,
            "expected_minutes": expected_minutes,
            "observed_minutes": None,
            "irregular_intervals": 0,
        }

    differences = (
        timestamps
        .sort_values()
        .diff()
        .dropna()
    )

    mode = differences.mode()

    if mode.empty:
        observed = None
    else:
        observed = float(
            mode.iloc[0].total_seconds()
            / 60.0
        )

    irregular_intervals = 0

    if expected_minutes is not None:

        expected_delta = pd.Timedelta(
            minutes=expected_minutes
        )

        irregular_intervals = int(
            (differences != expected_delta)
            .sum()
        )

    return {
        "valid": (
            irregular_intervals == 0
            if expected_minutes is not None
            else True
        ),
        "expected_minutes": expected_minutes,
        "observed_minutes": observed,
        "irregular_intervals": irregular_intervals,
    }


def validate_feature_columns(
    df,
    features,
):
    """
    Validate that requested features exist and are numeric.
    """

    features = list(features)

    if not features:
        raise ValueError(
            "At least one feature must be supplied."
        )

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Requested features are missing: "
            f"{missing}"
        )

    non_numeric = [
        feature
        for feature in features
        if not pd.api.types.is_numeric_dtype(
            df[feature]
        )
    ]

    if non_numeric:
        raise ValueError(
            "Requested features must be numeric: "
            f"{non_numeric}"
        )

    return True


def validate_split(
    reference,
    monitoring,
    evaluation,
):
    """
    Validate chronological reference, monitoring, and evaluation
    splits.
    """

    datasets = {
        "reference": reference,
        "monitoring": monitoring,
        "evaluation": evaluation,
    }

    for name, dataset in datasets.items():

        if not isinstance(
            dataset,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

        if dataset.empty:
            raise ValueError(
                f"{name} split is empty."
            )

        if "timestamp" not in dataset.columns:
            raise ValueError(
                f"{name} split has no timestamp column."
            )

    reference_end = pd.to_datetime(
        reference["timestamp"]
    ).max()

    monitoring_start = pd.to_datetime(
        monitoring["timestamp"]
    ).min()

    monitoring_end = pd.to_datetime(
        monitoring["timestamp"]
    ).max()

    evaluation_start = pd.to_datetime(
        evaluation["timestamp"]
    ).min()

    if reference_end >= monitoring_start:
        raise ValueError(
            "Reference and monitoring periods overlap "
            "or are out of chronological order."
        )

    if monitoring_end >= evaluation_start:
        raise ValueError(
            "Monitoring and evaluation periods overlap "
            "or are out of chronological order."
        )

    return True


def validate_dataset(
    df,
    expected_features=None,
    timestamp_column="timestamp",
    expected_minutes=None,
    allow_missing=False,
):
    """
    Run the complete dataset validation suite.

    Returns
    -------
    ValidationResult
    """

    result = validate_dataframe(
        df,
        timestamp_column=timestamp_column,
        require_timestamp=True,
        require_numeric_features=True,
        allow_missing=allow_missing,
    )

    if not result.valid:
        return result

    errors = list(
        result.errors
    )

    warnings = list(
        result.warnings
    )

    if expected_features is not None:

        try:
            validate_feature_columns(
                df,
                expected_features,
            )

        except ValueError as exc:
            errors.append(
                str(exc)
            )

    frequency_result = (
        validate_temporal_frequency(
            df,
            timestamp_column=timestamp_column,
            expected_minutes=expected_minutes,
        )
    )

    if not frequency_result["valid"]:
        warnings.append(
            "Dataset contains irregular timestamp intervals."
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        row_count=result.row_count,
        column_count=result.column_count,
        feature_count=result.feature_count,
    )


def assert_valid_dataset(
    df,
    **kwargs,
):
    """
    Validate a dataset and raise ValueError when invalid.

    Returns the original dataframe when validation succeeds.
    """

    result = validate_dataset(
        df,
        **kwargs,
    )

    if not result.valid:
        message = (
            "Dataset validation failed:\n"
            + "\n".join(
                f"- {error}"
                for error in result.errors
            )
        )

        raise ValueError(
            message
        )

    return df


__all__ = [
    "ValidationResult",
    "validate_dataframe",
    "validate_temporal_frequency",
    "validate_feature_columns",
    "validate_split",
    "validate_dataset",
    "assert_valid_dataset",
]