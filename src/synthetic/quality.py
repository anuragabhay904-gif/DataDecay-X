"""
Synthetic-data quality assessment for DataDecay-X.

This module evaluates whether generated synthetic data is
statistically and structurally similar enough to the real
reference data to be considered usable.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.datadecay.detection import (
    calculate_dsi,
    calculate_feature_metrics,
    classify_dsi,
)

from .constraints import (
    validate_synthetic_constraints,
)


@dataclass
class QualityResult:
    """Result of a synthetic-data quality assessment."""

    accepted: bool
    dsi: float
    severity: str
    constraint_passed: bool
    feature_metrics: dict
    constraint_results: dict
    reasons: list[str]

    def to_dict(self):
        """Return a serializable quality result."""

        return {
            "accepted": self.accepted,
            "dsi": self.dsi,
            "severity": self.severity,
            "constraint_passed": self.constraint_passed,
            "feature_metrics": self.feature_metrics,
            "constraint_results": self.constraint_results,
            "reasons": self.reasons,
        }


def _select_features(
    reference_df,
    synthetic_df,
    features=None,
):
    """Select common numeric features."""

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
            "No common numeric features are available."
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


def calculate_distribution_similarity(
    reference_df,
    synthetic_df,
    features=None,
):
    """
    Calculate feature-level distribution similarity metrics.

    The drift metrics are calculated by treating the reference
    data as the baseline and the synthetic data as the current
    distribution.
    """

    features = _select_features(
        reference_df,
        synthetic_df,
        features,
    )

    return calculate_feature_metrics(
        reference=reference_df,
        current=synthetic_df,
        features=features,
    )


def calculate_quality_score(
    feature_metrics,
):
    """
    Calculate the Data Shift Index for synthetic data quality.

    A lower DSI indicates that the synthetic distribution is
    closer to the reference distribution.
    """

    if not feature_metrics:
        raise ValueError(
            "feature_metrics cannot be empty."
        )

    return float(
        calculate_dsi(
            feature_metrics
        )
    )


def assess_synthetic_quality(
    reference_df,
    synthetic_df,
    features=None,
    dsi_threshold=0.50,
    range_tolerance=0.0,
    max_mean_difference=0.20,
    max_std_difference=0.30,
):
    """
    Assess synthetic data using distribution metrics and
    structural/statistical constraints.

    Synthetic data is accepted only when:

    1. All configured constraints pass.
    2. The synthetic-data DSI is at or below the configured
       threshold.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        Real/reference dataset.

    synthetic_df : pandas.DataFrame
        Generated synthetic dataset.

    features : iterable, optional
        Numeric features to assess.

    dsi_threshold : float
        Maximum acceptable DSI.

    range_tolerance : float
        Allowed extension beyond reference feature ranges.

    max_mean_difference : float
        Maximum relative mean difference.

    max_std_difference : float
        Maximum relative standard-deviation difference.

    Returns
    -------
    QualityResult
    """

    if dsi_threshold < 0:
        raise ValueError(
            "dsi_threshold cannot be negative."
        )

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

    if reference_df.empty:
        raise ValueError(
            "reference_df cannot be empty."
        )

    if synthetic_df.empty:
        raise ValueError(
            "synthetic_df cannot be empty."
        )

    features = _select_features(
        reference_df,
        synthetic_df,
        features,
    )

    feature_metrics = (
        calculate_distribution_similarity(
            reference_df,
            synthetic_df,
            features,
        )
    )

    dsi = calculate_quality_score(
        feature_metrics
    )

    severity = classify_dsi(
        dsi
    )

    constraint_results = (
        validate_synthetic_constraints(
            reference_df=reference_df,
            synthetic_df=synthetic_df,
            features=features,
            range_tolerance=range_tolerance,
            max_mean_difference=max_mean_difference,
            max_std_difference=max_std_difference,
        )
    )

    constraint_passed = bool(
        constraint_results["passed"]
    )

    reasons = []

    if dsi > dsi_threshold:
        reasons.append(
            "Synthetic distribution exceeds "
            f"the DSI threshold ({dsi:.4f} > "
            f"{dsi_threshold:.4f})."
        )

    if not constraint_passed:
        reasons.append(
            "One or more synthetic-data "
            "constraints failed."
        )

    if not reasons:
        reasons.append(
            "Synthetic data passed the configured "
            "quality gate."
        )

    accepted = (
        dsi <= dsi_threshold
        and constraint_passed
    )

    return QualityResult(
        accepted=accepted,
        dsi=dsi,
        severity=severity,
        constraint_passed=constraint_passed,
        feature_metrics=feature_metrics,
        constraint_results=constraint_results,
        reasons=reasons,
    )


def quality_gate(
    reference_df,
    synthetic_df,
    features=None,
    threshold=0.50,
):
    """
    Simplified quality-gate interface.

    Returns
    -------
    bool
        True when synthetic data passes the quality gate.
    """

    result = assess_synthetic_quality(
        reference_df=reference_df,
        synthetic_df=synthetic_df,
        features=features,
        dsi_threshold=threshold,
    )

    return result.accepted


def compare_synthetic_datasets(
    reference_df,
    synthetic_datasets,
    features=None,
    dsi_threshold=0.50,
):
    """
    Compare multiple synthetic datasets against the same
    reference dataset.

    Parameters
    ----------
    synthetic_datasets : dict
        Mapping of names to synthetic DataFrames.

    Returns
    -------
    dict
        QualityResult objects keyed by dataset name.
    """

    if not isinstance(
        synthetic_datasets,
        dict,
    ):
        raise TypeError(
            "synthetic_datasets must be a dictionary."
        )

    results = {}

    for name, synthetic_df in (
        synthetic_datasets.items()
    ):
        results[name] = assess_synthetic_quality(
            reference_df=reference_df,
            synthetic_df=synthetic_df,
            features=features,
            dsi_threshold=dsi_threshold,
        )

    return results


def quality_summary(
    result,
):
    """
    Return a compact summary of a QualityResult.
    """

    if not isinstance(
        result,
        QualityResult,
    ):
        raise TypeError(
            "result must be a QualityResult."
        )

    return {
        "accepted": result.accepted,
        "dsi": result.dsi,
        "severity": result.severity,
        "constraint_passed": result.constraint_passed,
        "reason_count": len(
            result.reasons
        ),
    }


__all__ = [
    "QualityResult",
    "calculate_distribution_similarity",
    "calculate_quality_score",
    "assess_synthetic_quality",
    "quality_gate",
    "compare_synthetic_datasets",
    "quality_summary",
]