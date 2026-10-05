"""
Drift metrics for DataDecay-X.

This legacy module provides a small wrapper around the project's
main drift-detection implementation in ``src.datadecay.detection``.
"""

import numpy as np
import pandas as pd

from src.datadecay.detection import (
    calculate_dsi,
    calculate_feature_metrics,
    calculate_ks,
    calculate_psi,
    calculate_wasserstein,
    classify_dsi,
)


def calculate_psi_metric(
    reference,
    current,
):
    """
    Calculate Population Stability Index (PSI).
    """

    return float(
        calculate_psi(
            reference,
            current,
        )
    )


def calculate_ks_metric(
    reference,
    current,
):
    """
    Calculate the Kolmogorov-Smirnov statistic.
    """

    result = calculate_ks(
        reference,
        current,
    )

    if isinstance(result, dict):
        return float(
            result.get(
                "statistic",
                result.get("ks", 0.0),
            )
        )

    return float(result)


def calculate_wasserstein_metric(
    reference,
    current,
):
    """
    Calculate the Wasserstein distance.
    """

    return float(
        calculate_wasserstein(
            reference,
            current,
        )
    )


def calculate_drift_metrics(
    reference_df,
    current_df,
    features=None,
):
    """
    Calculate drift metrics for multiple features.

    Returns
    -------
    dict
        Feature-level PSI, KS, and Wasserstein results.
    """

    if not isinstance(
        reference_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "reference_df must be a pandas DataFrame."
        )

    if not isinstance(
        current_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "current_df must be a pandas DataFrame."
        )

    if features is None:
        features = [
            column
            for column in reference_df.columns
            if column in current_df.columns
            and pd.api.types.is_numeric_dtype(
                reference_df[column]
            )
        ]

    features = list(features)

    if not features:
        raise ValueError(
            "No valid numeric features were provided."
        )

    missing_reference = [
        feature
        for feature in features
        if feature not in reference_df.columns
    ]

    missing_current = [
        feature
        for feature in features
        if feature not in current_df.columns
    ]

    if missing_reference:
        raise ValueError(
            "Features missing from reference data: "
            f"{missing_reference}"
        )

    if missing_current:
        raise ValueError(
            "Features missing from current data: "
            f"{missing_current}"
        )

    return calculate_feature_metrics(
        reference=reference_df,
        current=current_df,
        features=features,
    )


def calculate_drift_score(
    feature_metrics,
):
    """
    Calculate the Data Shift Index (DSI).
    """

    return float(
        calculate_dsi(
            feature_metrics
        )
    )


def classify_drift(
    dsi,
):
    """
    Convert a DSI score into a severity label.
    """

    return classify_dsi(
        float(dsi)
    )


def summarize_drift(
    feature_metrics,
):
    """
    Create a compact summary from feature-level metrics.

    The returned values are useful for reports and dashboards.
    """

    if not isinstance(
        feature_metrics,
        dict,
    ):
        raise TypeError(
            "feature_metrics must be a dictionary."
        )

    if not feature_metrics:
        return {
            "mean_psi": 0.0,
            "mean_ks": 0.0,
            "mean_wasserstein": 0.0,
            "max_psi": 0.0,
            "max_ks": 0.0,
            "max_wasserstein": 0.0,
            "dsi": 0.0,
            "severity": "LOW",
        }

    psi_values = []
    ks_values = []
    wasserstein_values = []

    for feature, metrics in feature_metrics.items():

        if not isinstance(metrics, dict):
            continue

        psi = metrics.get("psi")
        ks = metrics.get("ks")
        wasserstein = metrics.get(
            "wasserstein"
        )

        if psi is not None:
            psi_values.append(
                float(psi)
            )

        if ks is not None:
            ks_values.append(
                float(ks)
            )

        if wasserstein is not None:
            wasserstein_values.append(
                float(wasserstein)
            )

    def mean_or_zero(values):
        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    def max_or_zero(values):
        if not values:
            return 0.0

        return float(
            np.max(values)
        )

    dsi = calculate_drift_score(
        feature_metrics
    )

    return {
        "mean_psi": mean_or_zero(
            psi_values
        ),
        "mean_ks": mean_or_zero(
            ks_values
        ),
        "mean_wasserstein": mean_or_zero(
            wasserstein_values
        ),
        "max_psi": max_or_zero(
            psi_values
        ),
        "max_ks": max_or_zero(
            ks_values
        ),
        "max_wasserstein": max_or_zero(
            wasserstein_values
        ),
        "dsi": dsi,
        "severity": classify_drift(
            dsi
        ),
    }


__all__ = [
    "calculate_psi_metric",
    "calculate_ks_metric",
    "calculate_wasserstein_metric",
    "calculate_drift_metrics",
    "calculate_drift_score",
    "classify_drift",
    "summarize_drift",
]