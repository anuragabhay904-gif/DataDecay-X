"""
Synthetic-data constraints for DataDecay-X.

This module validates generated synthetic data against basic
structural and statistical constraints derived from the real
reference data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConstraintResult:
    """Result of a single synthetic-data constraint check."""

    name: str
    passed: bool
    details: dict

    def to_dict(self):
        """Return the result as a dictionary."""

        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }


def _numeric_features(
    reference_df,
    synthetic_df,
    features=None,
):
    """Validate and return common numeric features."""

    if not isinstance(
        reference_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "reference_df must be a pandas DataFrame."
        )

    if not isinstance(
        synthetic_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "synthetic_df must be a pandas DataFrame."
        )

    if features is None:
        features = [
            column
            for column in reference_df.columns
            if column in synthetic_df.columns
            and column != "timestamp"
            and pd.api.types.is_numeric_dtype(
                reference_df[column]
            )
            and pd.api.types.is_numeric_dtype(
                synthetic_df[column]
            )
        ]

    features = list(features)

    if not features:
        raise ValueError(
            "No common numeric features were found."
        )

    missing_reference = [
        feature
        for feature in features
        if feature not in reference_df.columns
    ]

    missing_synthetic = [
        feature
        for feature in features
        if feature not in synthetic_df.columns
    ]

    if missing_reference:
        raise ValueError(
            "Features missing from reference data: "
            f"{missing_reference}"
        )

    if missing_synthetic:
        raise ValueError(
            "Features missing from synthetic data: "
            f"{missing_synthetic}"
        )

    return features


def check_schema(
    reference_df,
    synthetic_df,
):
    """
    Check whether synthetic data preserves the required schema.

    The synthetic dataframe may contain fewer rows, but it should
    contain the same required columns as the reference dataframe.
    """

    reference_columns = list(
        reference_df.columns
    )

    synthetic_columns = list(
        synthetic_df.columns
    )

    missing = [
        column
        for column in reference_columns
        if column not in synthetic_columns
    ]

    extra = [
        column
        for column in synthetic_columns
        if column not in reference_columns
    ]

    passed = (
        len(missing) == 0
        and len(extra) == 0
    )

    return ConstraintResult(
        name="schema",
        passed=passed,
        details={
            "reference_columns": reference_columns,
            "synthetic_columns": synthetic_columns,
            "missing_columns": missing,
            "extra_columns": extra,
        },
    )


def check_missing_values(
    synthetic_df,
    features=None,
):
    """
    Check for missing values in synthetic numeric features.
    """

    if features is None:
        features = [
            column
            for column in synthetic_df.columns
            if column != "timestamp"
            and pd.api.types.is_numeric_dtype(
                synthetic_df[column]
            )
        ]

    features = list(features)

    missing_counts = {
        feature: int(
            synthetic_df[feature].isna().sum()
        )
        for feature in features
    }

    total_missing = sum(
        missing_counts.values()
    )

    return ConstraintResult(
        name="missing_values",
        passed=total_missing == 0,
        details={
            "missing_counts": missing_counts,
            "total_missing": total_missing,
        },
    )


def check_finite_values(
    synthetic_df,
    features=None,
):
    """
    Check that numeric synthetic values are finite.
    """

    if features is None:
        features = [
            column
            for column in synthetic_df.columns
            if column != "timestamp"
            and pd.api.types.is_numeric_dtype(
                synthetic_df[column]
            )
        ]

    invalid_counts = {}

    for feature in features:

        values = pd.to_numeric(
            synthetic_df[feature],
            errors="coerce",
        )

        invalid_counts[feature] = int(
            (~np.isfinite(values)).sum()
        )

    total_invalid = sum(
        invalid_counts.values()
    )

    return ConstraintResult(
        name="finite_values",
        passed=total_invalid == 0,
        details={
            "invalid_counts": invalid_counts,
            "total_invalid": total_invalid,
        },
    )


def check_non_negative(
    synthetic_df,
    features=None,
):
    """
    Check that synthetic values are non-negative.

    Electricity load values are expected to be non-negative,
    so this is an appropriate constraint for the project's
    electricity dataset.
    """

    if features is None:
        features = [
            column
            for column in synthetic_df.columns
            if column != "timestamp"
            and pd.api.types.is_numeric_dtype(
                synthetic_df[column]
            )
        ]

    violations = {}

    for feature in features:

        values = pd.to_numeric(
            synthetic_df[feature],
            errors="coerce",
        )

        count = int(
            (values < 0).sum()
        )

        violations[feature] = count

    total_violations = sum(
        violations.values()
    )

    return ConstraintResult(
        name="non_negative",
        passed=total_violations == 0,
        details={
            "negative_counts": violations,
            "total_negative": total_violations,
        },
    )


def check_range(
    reference_df,
    synthetic_df,
    features=None,
    tolerance=0.0,
):
    """
    Check whether synthetic values remain within the observed
    reference range, optionally allowing a tolerance.

    Parameters
    ----------
    tolerance : float
        Fractional extension of the reference range.

        For example, 0.05 permits the synthetic values to extend
        5% beyond the observed reference range.
    """

    if tolerance < 0:
        raise ValueError(
            "tolerance cannot be negative."
        )

    features = _numeric_features(
        reference_df,
        synthetic_df,
        features,
    )

    violations = {}
    bounds = {}

    for feature in features:

        reference_values = pd.to_numeric(
            reference_df[feature],
            errors="coerce",
        ).dropna()

        synthetic_values = pd.to_numeric(
            synthetic_df[feature],
            errors="coerce",
        ).dropna()

        if reference_values.empty:
            violations[feature] = 0
            bounds[feature] = {
                "lower": None,
                "upper": None,
            }
            continue

        minimum = float(
            reference_values.min()
        )

        maximum = float(
            reference_values.max()
        )

        spread = maximum - minimum

        lower = (
            minimum - spread * tolerance
        )

        upper = (
            maximum + spread * tolerance
        )

        below = int(
            (synthetic_values < lower).sum()
        )

        above = int(
            (synthetic_values > upper).sum()
        )

        violations[feature] = (
            below + above
        )

        bounds[feature] = {
            "lower": lower,
            "upper": upper,
        }

    total_violations = sum(
        violations.values()
    )

    return ConstraintResult(
        name="range",
        passed=total_violations == 0,
        details={
            "violations": violations,
            "bounds": bounds,
            "tolerance": tolerance,
        },
    )


def check_mean_deviation(
    reference_df,
    synthetic_df,
    features=None,
    max_relative_difference=0.20,
):
    """
    Check whether synthetic feature means remain reasonably close
    to reference means.
    """

    if max_relative_difference < 0:
        raise ValueError(
            "max_relative_difference cannot be negative."
        )

    features = _numeric_features(
        reference_df,
        synthetic_df,
        features,
    )

    differences = {}
    violations = {}

    for feature in features:

        reference_mean = float(
            pd.to_numeric(
                reference_df[feature],
                errors="coerce",
            ).mean()
        )

        synthetic_mean = float(
            pd.to_numeric(
                synthetic_df[feature],
                errors="coerce",
            ).mean()
        )

        denominator = max(
            abs(reference_mean),
            1e-12,
        )

        relative_difference = (
            abs(
                synthetic_mean
                - reference_mean
            )
            / denominator
        )

        differences[feature] = (
            relative_difference
        )

        violations[feature] = (
            relative_difference
            > max_relative_difference
        )

    passed = not any(
        violations.values()
    )

    return ConstraintResult(
        name="mean_deviation",
        passed=passed,
        details={
            "relative_differences": differences,
            "violations": violations,
            "threshold": max_relative_difference,
        },
    )


def check_std_deviation(
    reference_df,
    synthetic_df,
    features=None,
    max_relative_difference=0.30,
):
    """
    Check whether synthetic feature standard deviations remain
    reasonably close to reference standard deviations.
    """

    if max_relative_difference < 0:
        raise ValueError(
            "max_relative_difference cannot be negative."
        )

    features = _numeric_features(
        reference_df,
        synthetic_df,
        features,
    )

    differences = {}
    violations = {}

    for feature in features:

        reference_std = float(
            pd.to_numeric(
                reference_df[feature],
                errors="coerce",
            ).std()
        )

        synthetic_std = float(
            pd.to_numeric(
                synthetic_df[feature],
                errors="coerce",
            ).std()
        )

        denominator = max(
            abs(reference_std),
            1e-12,
        )

        relative_difference = (
            abs(
                synthetic_std
                - reference_std
            )
            / denominator
        )

        differences[feature] = (
            relative_difference
        )

        violations[feature] = (
            relative_difference
            > max_relative_difference
        )

    passed = not any(
        violations.values()
    )

    return ConstraintResult(
        name="std_deviation",
        passed=passed,
        details={
            "relative_differences": differences,
            "violations": violations,
            "threshold": max_relative_difference,
        },
    )


def validate_synthetic_constraints(
    reference_df,
    synthetic_df,
    features=None,
    range_tolerance=0.0,
    max_mean_difference=0.20,
    max_std_difference=0.30,
):
    """
    Run the complete synthetic-data constraint suite.

    Returns
    -------
    dict
        Constraint results and an overall pass/fail status.
    """

    features = _numeric_features(
        reference_df,
        synthetic_df,
        features,
    )

    results = [
        check_schema(
            reference_df,
            synthetic_df,
        ),
        check_missing_values(
            synthetic_df,
            features,
        ),
        check_finite_values(
            synthetic_df,
            features,
        ),
        check_non_negative(
            synthetic_df,
            features,
        ),
        check_range(
            reference_df,
            synthetic_df,
            features,
            tolerance=range_tolerance,
        ),
        check_mean_deviation(
            reference_df,
            synthetic_df,
            features,
            max_relative_difference=(
                max_mean_difference
            ),
        ),
        check_std_deviation(
            reference_df,
            synthetic_df,
            features,
            max_relative_difference=(
                max_std_difference
            ),
        ),
    ]

    passed = all(
        result.passed
        for result in results
    )

    return {
        "passed": passed,
        "features": features,
        "checks": [
            result.to_dict()
            for result in results
        ],
    }


__all__ = [
    "ConstraintResult",
    "check_schema",
    "check_missing_values",
    "check_finite_values",
    "check_non_negative",
    "check_range",
    "check_mean_deviation",
    "check_std_deviation",
    "validate_synthetic_constraints",
]