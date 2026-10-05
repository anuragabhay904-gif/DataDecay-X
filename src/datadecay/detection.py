import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance


EPSILON = 1e-6


def _clean(values):
    """Convert values to finite numeric NumPy values."""
    values = pd.Series(values)

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values.dropna()

    values = values[
        np.isfinite(values)
    ]

    return values.to_numpy()


def calculate_psi(
    reference,
    current,
    bins=10,
):
    """
    Calculate Population Stability Index.

    Bins are determined from reference quantiles.
    """
    reference = _clean(reference)
    current = _clean(current)

    if (
        len(reference) == 0
        or len(current) == 0
    ):
        return np.nan

    if bins < 2:
        raise ValueError(
            "bins must be >= 2."
        )

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    edges = np.quantile(
        reference,
        quantiles,
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=edges,
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges,
    )

    reference_pct = (
        reference_counts
        / len(reference)
    )

    current_pct = (
        current_counts
        / len(current)
    )

    reference_pct = np.clip(
        reference_pct,
        EPSILON,
        None,
    )

    current_pct = np.clip(
        current_pct,
        EPSILON,
        None,
    )

    return float(
        np.sum(
            (
                current_pct
                - reference_pct
            )
            * np.log(
                current_pct
                / reference_pct
            )
        )
    )


def calculate_ks(
    reference,
    current,
):
    """
    Calculate the two-sample KS statistic.

    Returns
    -------
    statistic, p_value
    """
    reference = _clean(reference)
    current = _clean(current)

    if (
        len(reference) == 0
        or len(current) == 0
    ):
        return np.nan, np.nan

    result = ks_2samp(
        reference,
        current,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def calculate_wasserstein(
    reference,
    current,
):
    """Calculate first Wasserstein distance."""
    reference = _clean(reference)
    current = _clean(current)

    if (
        len(reference) == 0
        or len(current) == 0
    ):
        return np.nan

    return float(
        wasserstein_distance(
            reference,
            current,
        )
    )


def calculate_feature_metrics(
    reference_df,
    current_df,
    features,
    psi_bins=10,
):
    """
    Calculate PSI, KS and Wasserstein
    for every requested feature.
    """
    if not features:
        raise ValueError(
            "At least one feature is required."
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
            "Missing features in reference dataframe: "
            f"{missing_reference}"
        )

    if missing_current:
        raise ValueError(
            "Missing features in current dataframe: "
            f"{missing_current}"
        )

    rows = []

    for feature in features:
        reference = reference_df[
            feature
        ]

        current = current_df[
            feature
        ]

        psi_value = calculate_psi(
            reference,
            current,
            bins=psi_bins,
        )

        ks_value, p_value = calculate_ks(
            reference,
            current,
        )

        wasserstein_value = (
            calculate_wasserstein(
                reference,
                current,
            )
        )

        reference_values = _clean(
            reference
        )

        reference_std = np.std(
            reference_values
        )

        if (
            reference_std == 0
            or np.isnan(reference_std)
        ):
            normalized_wasserstein = 0.0

        elif np.isnan(
            wasserstein_value
        ):
            normalized_wasserstein = np.nan

        else:
            normalized_wasserstein = (
                wasserstein_value
                / reference_std
            )

        rows.append(
            {
                "feature": feature,
                "psi": psi_value,
                "ks_statistic": ks_value,
                "ks_pvalue": p_value,
                "wasserstein": wasserstein_value,
                "normalized_wasserstein":
                    normalized_wasserstein,
            }
        )

    return pd.DataFrame(rows)


def calculate_dsi(
    metrics,
    psi_reference=0.25,
    wasserstein_reference=1.0,
    weights=None,
):
    """
    Calculate the project Drift Severity Index.

    DSI combines normalized:
    - PSI
    - KS statistic
    - Wasserstein distance

    Reference values and weights are project
    configuration, not universal thresholds.
    """
    if metrics.empty:
        return np.nan

    required_columns = {
        "psi",
        "ks_statistic",
        "normalized_wasserstein",
    }

    missing = (
        required_columns
        - set(metrics.columns)
    )

    if missing:
        raise ValueError(
            "Missing DSI columns: "
            f"{sorted(missing)}"
        )

    if psi_reference <= 0:
        raise ValueError(
            "psi_reference must be > 0."
        )

    if wasserstein_reference <= 0:
        raise ValueError(
            "wasserstein_reference must be > 0."
        )

    if weights is None:
        weights = {
            "psi": 1 / 3,
            "ks": 1 / 3,
            "wasserstein": 1 / 3,
        }

    total_weight = sum(
        weights.values()
    )

    if total_weight <= 0:
        raise ValueError(
            "DSI weights must have positive total."
        )

    weights = {
        key: value / total_weight
        for key, value in weights.items()
    }

    psi_score = np.clip(
        metrics["psi"]
        / psi_reference,
        0,
        1,
    )

    ks_score = np.clip(
        metrics["ks_statistic"],
        0,
        1,
    )

    wasserstein_score = np.clip(
        metrics[
            "normalized_wasserstein"
        ]
        / wasserstein_reference,
        0,
        1,
    )

    feature_scores = (
        weights["psi"] * psi_score
        + weights["ks"] * ks_score
        + weights["wasserstein"]
        * wasserstein_score
    )

    return float(
        feature_scores.mean()
    )


def classify_dsi(
    dsi,
    low=0.20,
    moderate=0.50,
    high=0.75,
):
    """
    Convert DSI into project severity bands.

    LOW       < 0.20
    MODERATE  < 0.50
    HIGH      < 0.75
    SEVERE    >= 0.75
    """
    if pd.isna(dsi):
        return "UNKNOWN"

    if dsi < low:
        return "LOW"

    if dsi < moderate:
        return "MODERATE"

    if dsi < high:
        return "HIGH"

    return "SEVERE"