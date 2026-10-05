from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .detection import (
    calculate_feature_metrics,
    calculate_dsi,
)


@dataclass
class QualityGateResult:
    """
    Result produced by the synthetic-data quality gate.
    """

    accepted: bool
    quality_score: float
    threshold: float
    schema_valid: bool
    missingness_valid: bool
    distribution_valid: bool
    correlation_valid: bool
    feature_metrics: pd.DataFrame
    checks: Dict
    metadata: Dict

    def to_dict(self) -> Dict:
        """
        Convert the quality-gate result into a JSON-friendly
        dictionary.
        """

        result = asdict(self)

        result["feature_metrics"] = (
            self.feature_metrics.to_dict(
                orient="records"
            )
        )

        return result


def _validate_features(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    features: Sequence[str],
) -> None:
    """
    Validate that all requested features exist and are numeric.
    """

    if not isinstance(
        reference,
        pd.DataFrame,
    ):
        raise TypeError(
            "reference must be a pandas DataFrame."
        )

    if not isinstance(
        candidate,
        pd.DataFrame,
    ):
        raise TypeError(
            "candidate must be a pandas DataFrame."
        )

    if reference.empty:
        raise ValueError(
            "Reference dataframe is empty."
        )

    if candidate.empty:
        raise ValueError(
            "Candidate dataframe is empty."
        )

    if not features:
        raise ValueError(
            "At least one feature is required."
        )

    missing_reference = [
        feature
        for feature in features
        if feature not in reference.columns
    ]

    missing_candidate = [
        feature
        for feature in features
        if feature not in candidate.columns
    ]

    if missing_reference:
        raise ValueError(
            "Missing features in reference: "
            f"{missing_reference}"
        )

    if missing_candidate:
        raise ValueError(
            "Missing features in candidate: "
            f"{missing_candidate}"
        )

    non_numeric_reference = [
        feature
        for feature in features
        if not pd.api.types.is_numeric_dtype(
            reference[feature]
        )
    ]

    non_numeric_candidate = [
        feature
        for feature in features
        if not pd.api.types.is_numeric_dtype(
            candidate[feature]
        )
    ]

    if non_numeric_reference:
        raise TypeError(
            "Reference features must be numeric: "
            f"{non_numeric_reference}"
        )

    if non_numeric_candidate:
        raise TypeError(
            "Candidate features must be numeric: "
            f"{non_numeric_candidate}"
        )


def check_schema(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    features: Sequence[str],
) -> Dict:
    """
    Check that the candidate contains the required schema.
    """

    reference_columns = set(
        reference.columns
    )

    candidate_columns = set(
        candidate.columns
    )

    required_columns = set(
        features
    )

    missing_columns = sorted(
        required_columns
        - candidate_columns
    )

    extra_columns = sorted(
        candidate_columns
        - reference_columns
    )

    dtype_matches = all(
        str(
            reference[feature].dtype
        )
        == str(
            candidate[feature].dtype
        )
        for feature in features
    )

    valid = (
        len(missing_columns) == 0
        and dtype_matches
    )

    return {
        "valid": bool(valid),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "dtype_matches": bool(
            dtype_matches
        ),
    }


def check_missingness(
    candidate: pd.DataFrame,
    features: Sequence[str],
    max_missing_fraction: float = 0.01,
) -> Dict:
    """
    Check whether candidate missingness remains below a threshold.
    """

    if not (
        0.0
        <= max_missing_fraction
        <= 1.0
    ):
        raise ValueError(
            "max_missing_fraction must be "
            "between 0 and 1."
        )

    missing_fraction = {}

    for feature in features:
        fraction = float(
            candidate[feature]
            .isna()
            .mean()
        )

        missing_fraction[feature] = (
            fraction
        )

    maximum_missingness = (
        max(
            missing_fraction.values()
        )
        if missing_fraction
        else 0.0
    )

    valid = (
        maximum_missingness
        <= max_missing_fraction
    )

    return {
        "valid": bool(valid),
        "maximum_missing_fraction": (
            maximum_missingness
        ),
        "allowed_missing_fraction": (
            float(max_missing_fraction)
        ),
        "feature_missing_fraction": (
            missing_fraction
        ),
    }


def check_value_bounds(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    features: Sequence[str],
    lower_quantile: float = 0.001,
    upper_quantile: float = 0.999,
    tolerance: float = 0.05,
) -> Dict:
    """
    Check whether candidate values remain close to the empirical
    range of the reference data.

    A tolerance of 0.05 permits candidate values to extend
    slightly beyond the reference quantile range.
    """

    if not (
        0.0
        <= lower_quantile
        < upper_quantile
        <= 1.0
    ):
        raise ValueError(
            "Invalid quantile range."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    feature_results = {}

    all_valid = True

    for feature in features:
        reference_values = (
            pd.to_numeric(
                reference[feature],
                errors="coerce",
            )
            .dropna()
        )

        candidate_values = (
            pd.to_numeric(
                candidate[feature],
                errors="coerce",
            )
            .dropna()
        )

        if (
            reference_values.empty
            or candidate_values.empty
        ):
            feature_results[feature] = {
                "valid": False,
                "reason": "empty_values",
            }

            all_valid = False
            continue

        lower = float(
            reference_values.quantile(
                lower_quantile
            )
        )

        upper = float(
            reference_values.quantile(
                upper_quantile
            )
        )

        spread = max(
            upper - lower,
            1e-12,
        )

        allowed_lower = (
            lower - tolerance * spread
        )

        allowed_upper = (
            upper + tolerance * spread
        )

        candidate_min = float(
            candidate_values.min()
        )

        candidate_max = float(
            candidate_values.max()
        )

        valid = (
            candidate_min
            >= allowed_lower
            and candidate_max
            <= allowed_upper
        )

        feature_results[feature] = {
            "valid": bool(valid),
            "reference_lower": lower,
            "reference_upper": upper,
            "allowed_lower": allowed_lower,
            "allowed_upper": allowed_upper,
            "candidate_min": candidate_min,
            "candidate_max": candidate_max,
        }

        if not valid:
            all_valid = False

    return {
        "valid": bool(all_valid),
        "features": feature_results,
    }


def check_distribution(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    features: Sequence[str],
    psi_threshold: float = 0.25,
    wasserstein_threshold: float = 1.0,
) -> Dict:
    """
    Evaluate whether candidate feature distributions remain
    sufficiently close to the reference distributions.

    Wasserstein distance is evaluated using the normalized
    Wasserstein metric so that features with different numeric
    scales can be compared consistently.
    """

    metrics = calculate_feature_metrics(
        reference[
            list(features)
        ],
        candidate[
            list(features)
        ],
        features=list(features),
    )

    if metrics.empty:
        return {
            "valid": False,
            "metrics": metrics,
            "reason": "no_valid_metrics",
        }

    psi_valid = (
        metrics["psi"]
        <= psi_threshold
    )

    # Use normalized Wasserstein distance because the raw
    # Wasserstein distance depends on the scale of each feature.
    wasserstein_valid = (
        metrics[
            "normalized_wasserstein"
        ]
        <= wasserstein_threshold
    )

    valid = bool(
        psi_valid.all()
        and wasserstein_valid.all()
    )

    return {
        "valid": valid,
        "psi_threshold": float(
            psi_threshold
        ),
        "wasserstein_threshold": float(
            wasserstein_threshold
        ),
        "psi_pass_fraction": float(
            psi_valid.mean()
        ),
        "wasserstein_pass_fraction": float(
            wasserstein_valid.mean()
        ),
        "metrics": metrics,
    }


def check_correlation(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    features: Sequence[str],
    max_difference: float = 0.20,
) -> Dict:
    """
    Compare feature correlation structures between reference
    and candidate datasets.
    """

    if max_difference < 0:
        raise ValueError(
            "max_difference must be non-negative."
        )

    reference_corr = (
        reference[
            list(features)
        ]
        .corr()
        .fillna(0.0)
    )

    candidate_corr = (
        candidate[
            list(features)
        ]
        .corr()
        .fillna(0.0)
    )

    difference = (
        reference_corr
        - candidate_corr
    ).abs()

    values = difference.to_numpy(
        dtype=float
    )

    if len(values) == 0:
        maximum_difference = 0.0
        mean_difference = 0.0
    else:
        mask = ~np.eye(
            len(values),
            dtype=bool,
        )

        off_diagonal = values[
            mask
        ]

        if len(off_diagonal) == 0:
            maximum_difference = 0.0
            mean_difference = 0.0
        else:
            maximum_difference = float(
                np.max(
                    off_diagonal
                )
            )

            mean_difference = float(
                np.mean(
                    off_diagonal
                )
            )

    valid = (
        maximum_difference
        <= max_difference
    )

    return {
        "valid": bool(valid),
        "maximum_difference": (
            maximum_difference
        ),
        "mean_difference": (
            mean_difference
        ),
        "allowed_difference": float(
            max_difference
        ),
        "reference_correlation": (
            reference_corr
        ),
        "candidate_correlation": (
            candidate_corr
        ),
    }


def calculate_quality_score(
    schema_valid: bool,
    missingness_valid: bool,
    distribution_valid: bool,
    correlation_valid: bool,
    distribution_score: float,
) -> float:
    """
    Combine quality checks into a single interpretable score.
    """

    structural_score = np.mean(
        [
            float(schema_valid),
            float(missingness_valid),
            float(correlation_valid),
        ]
    )

    distribution_score = float(
        np.clip(
            distribution_score,
            0.0,
            1.0,
        )
    )

    score = (
        0.40 * structural_score
        + 0.60 * distribution_score
    )

    if not distribution_valid:
        score *= 0.90

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


def run_quality_gate(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    features: Sequence[str],
    threshold: float = 0.70,
    max_missing_fraction: float = 0.01,
    psi_threshold: float = 0.25,
    wasserstein_threshold: float = 1.0,
    correlation_difference: float = 0.20,
) -> QualityGateResult:
    """
    Run the complete synthetic-data quality gate.

    A candidate is accepted when:
    - schema is valid
    - missingness is acceptable
    - distribution checks pass
    - correlation structure is acceptable
    - final quality score meets the threshold
    """

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    _validate_features(
        reference,
        candidate,
        features,
    )

    schema_result = check_schema(
        reference,
        candidate,
        features,
    )

    missingness_result = check_missingness(
        candidate,
        features,
        max_missing_fraction=(
            max_missing_fraction
        ),
    )

    distribution_result = check_distribution(
        reference,
        candidate,
        features,
        psi_threshold=psi_threshold,
        wasserstein_threshold=(
            wasserstein_threshold
        ),
    )

    correlation_result = check_correlation(
        reference,
        candidate,
        features,
        max_difference=(
            correlation_difference
        ),
    )

    metrics = distribution_result[
        "metrics"
    ]

    if metrics.empty:
        distribution_score = 0.0
    else:
        try:
            dsi = calculate_dsi(
                metrics
            )

            distribution_score = float(
                1.0 - np.clip(
                    dsi,
                    0.0,
                    1.0,
                )
            )
        except (
            TypeError,
            ValueError,
            KeyError,
        ):
            distribution_score = 0.0

    quality_score = calculate_quality_score(
        schema_valid=schema_result[
            "valid"
        ],
        missingness_valid=missingness_result[
            "valid"
        ],
        distribution_valid=distribution_result[
            "valid"
        ],
        correlation_valid=correlation_result[
            "valid"
        ],
        distribution_score=(
            distribution_score
        ),
    )

    accepted = (
        schema_result["valid"]
        and missingness_result["valid"]
        and distribution_result["valid"]
        and correlation_result["valid"]
        and quality_score >= threshold
    )

    checks = {
        "schema": schema_result,
        "missingness": missingness_result,
        "distribution": {
            key: value
            for key, value in distribution_result.items()
            if key != "metrics"
        },
        "correlation": {
            key: value
            for key, value in correlation_result.items()
            if key not in {
                "reference_correlation",
                "candidate_correlation",
            }
        },
    }

    metadata = {
        "feature_count": len(
            features
        ),
        "reference_rows": int(
            len(reference)
        ),
        "candidate_rows": int(
            len(candidate)
        ),
        "threshold": float(
            threshold
        ),
        "distribution_score": float(
            distribution_score
        ),
    }

    return QualityGateResult(
        accepted=bool(accepted),
        quality_score=float(
            quality_score
        ),
        threshold=float(
            threshold
        ),
        schema_valid=bool(
            schema_result["valid"]
        ),
        missingness_valid=bool(
            missingness_result["valid"]
        ),
        distribution_valid=bool(
            distribution_result["valid"]
        ),
        correlation_valid=bool(
            correlation_result["valid"]
        ),
        feature_metrics=metrics,
        checks=checks,
        metadata=metadata,
    )


def quality_gate_summary(
    result: QualityGateResult,
) -> Dict:
    """
    Return a compact summary suitable for experiment reports.
    """

    return {
        "accepted": bool(
            result.accepted
        ),
        "quality_score": float(
            result.quality_score
        ),
        "threshold": float(
            result.threshold
        ),
        "schema_valid": bool(
            result.schema_valid
        ),
        "missingness_valid": bool(
            result.missingness_valid
        ),
        "distribution_valid": bool(
            result.distribution_valid
        ),
        "correlation_valid": bool(
            result.correlation_valid
        ),
    }


def save_quality_gate_report(
    result: QualityGateResult,
    path,
) -> None:
    """
    Save the quality-gate result as CSV/JSON-compatible
    tabular output depending on the supplied file extension.
    """

    path = str(path)

    if path.lower().endswith(".csv"):
        result.feature_metrics.to_csv(
            path,
            index=False,
        )
        return

    if path.lower().endswith(".json"):
        import json

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result.to_dict(),
                file,
                indent=4,
                default=str,
            )
        return

    raise ValueError(
        "Unsupported report format. "
        "Use .csv or .json."
    )