from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReliabilityResult:
    """
    Container for reliability analysis results.
    """

    model_name: str
    baseline_score: float
    drifted_score: float
    degradation: float
    degradation_ratio: float
    reliability_status: str

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "baseline_score": self.baseline_score,
            "drifted_score": self.drifted_score,
            "degradation": self.degradation,
            "degradation_ratio": self.degradation_ratio,
            "reliability_status": self.reliability_status,
        }


def _validate_score(
    score,
    name,
):
    """
    Validate a model performance score.
    """

    if score is None:
        raise ValueError(
            f"{name} cannot be None."
        )

    score = float(score)

    if not np.isfinite(score):
        raise ValueError(
            f"{name} must be finite."
        )

    return score


def calculate_degradation(
    baseline_score,
    current_score,
):
    """
    Calculate absolute performance degradation.

    Positive degradation means performance became worse.

    This assumes that larger performance scores
    represent better performance.
    """

    baseline_score = _validate_score(
        baseline_score,
        "baseline_score",
    )

    current_score = _validate_score(
        current_score,
        "current_score",
    )

    return float(
        baseline_score
        - current_score
    )


def calculate_degradation_ratio(
    baseline_score,
    current_score,
):
    """
    Calculate relative performance degradation.

    Returns the fraction of baseline performance
    that was lost.

    Example:
        baseline = 0.90
        current = 0.72

        degradation ratio = 0.20
    """

    baseline_score = _validate_score(
        baseline_score,
        "baseline_score",
    )

    current_score = _validate_score(
        current_score,
        "current_score",
    )

    if baseline_score == 0:
        return np.nan

    return float(
        (
            baseline_score
            - current_score
        )
        / abs(baseline_score)
    )


def classify_reliability(
    degradation_ratio,
    stable_threshold=0.05,
    warning_threshold=0.15,
    critical_threshold=0.30,
):
    """
    Classify model reliability based on
    relative performance degradation.

    LOW:
        degradation < 5%

    WARNING:
        degradation >= 5% and < 15%

    HIGH:
        degradation >= 15% and < 30%

    CRITICAL:
        degradation >= 30%
    """

    degradation_ratio = _validate_score(
        degradation_ratio,
        "degradation_ratio",
    )

    if stable_threshold < 0:
        raise ValueError(
            "stable_threshold must be >= 0."
        )

    if warning_threshold <= stable_threshold:
        raise ValueError(
            "warning_threshold must be greater "
            "than stable_threshold."
        )

    if critical_threshold <= warning_threshold:
        raise ValueError(
            "critical_threshold must be greater "
            "than warning_threshold."
        )

    if degradation_ratio < stable_threshold:
        return "STABLE"

    if degradation_ratio < warning_threshold:
        return "WARNING"

    if degradation_ratio < critical_threshold:
        return "HIGH"

    return "CRITICAL"


def evaluate_reliability(
    model_name,
    baseline_score,
    current_score,
):
    """
    Evaluate how much model performance changed
    between a baseline period and a later period.
    """

    baseline_score = _validate_score(
        baseline_score,
        "baseline_score",
    )

    current_score = _validate_score(
        current_score,
        "current_score",
    )

    degradation = calculate_degradation(
        baseline_score=baseline_score,
        current_score=current_score,
    )

    degradation_ratio = (
        calculate_degradation_ratio(
            baseline_score=baseline_score,
            current_score=current_score,
        )
    )

    if np.isnan(
        degradation_ratio
    ):
        status = "UNKNOWN"

    else:
        status = classify_reliability(
            degradation_ratio
        )

    return ReliabilityResult(
        model_name=model_name,
        baseline_score=baseline_score,
        drifted_score=current_score,
        degradation=degradation,
        degradation_ratio=degradation_ratio,
        reliability_status=status,
    )


def evaluate_multiple_models(
    baseline_scores,
    current_scores,
):
    """
    Evaluate reliability for multiple models.

    Parameters
    ----------
    baseline_scores:
        Dictionary mapping model names to
        baseline performance scores.

    current_scores:
        Dictionary mapping model names to
        later-period performance scores.
    """

    if not baseline_scores:
        raise ValueError(
            "baseline_scores cannot be empty."
        )

    if not current_scores:
        raise ValueError(
            "current_scores cannot be empty."
        )

    missing_current = [
        name
        for name in baseline_scores
        if name not in current_scores
    ]

    if missing_current:
        raise ValueError(
            "Missing current scores for models: "
            f"{missing_current}"
        )

    results = []

    for model_name in baseline_scores:

        result = evaluate_reliability(
            model_name=model_name,
            baseline_score=(
                baseline_scores[model_name]
            ),
            current_score=(
                current_scores[model_name]
            ),
        )

        results.append(
            result
        )

    return pd.DataFrame(
        [
            result.to_dict()
            for result in results
        ]
    )


def calculate_reliability_index(
    baseline_score,
    current_score,
):
    """
    Convert performance retention into a
    0-to-1 reliability index.

    1.0 means no degradation.

    0.0 means performance has fallen to zero
    relative to the baseline.

    Values are clipped to [0, 1].
    """

    baseline_score = _validate_score(
        baseline_score,
        "baseline_score",
    )

    current_score = _validate_score(
        current_score,
        "current_score",
    )

    if baseline_score == 0:
        return np.nan

    retention = (
        current_score
        / baseline_score
    )

    return float(
        np.clip(
            retention,
            0.0,
            1.0,
        )
    )


def reliability_summary(
    results,
):
    """
    Create a compact summary from reliability results.
    """

    if results is None:
        raise ValueError(
            "results cannot be None."
        )

    if isinstance(
        results,
        pd.DataFrame,
    ):

        if results.empty:
            return {
                "model_count": 0,
                "mean_degradation": None,
                "mean_degradation_ratio": None,
                "mean_reliability_index": None,
            }

        reliability_indices = []

        for _, row in results.iterrows():

            reliability_indices.append(
                calculate_reliability_index(
                    row[
                        "baseline_score"
                    ],
                    row[
                        "drifted_score"
                    ],
                )
            )

        return {
            "model_count": int(
                len(results)
            ),
            "mean_degradation": float(
                results[
                    "degradation"
                ].mean()
            ),
            "mean_degradation_ratio": float(
                results[
                    "degradation_ratio"
                ].mean()
            ),
            "mean_reliability_index": float(
                np.nanmean(
                    reliability_indices
                )
            ),
        }

    raise TypeError(
        "results must be a pandas DataFrame."
    )


def compare_reliability_by_period(
    baseline_results,
    drifted_results,
    score_column,
):
    """
    Compare model scores across two periods.

    Parameters
    ----------
    baseline_results:
        DataFrame containing baseline model scores.

    drifted_results:
        DataFrame containing later-period scores.

    score_column:
        Name of the performance column.
    """

    required = {
        "model_name",
        score_column,
    }

    missing_baseline = (
        required
        - set(
            baseline_results.columns
        )
    )

    missing_drifted = (
        required
        - set(
            drifted_results.columns
        )
    )

    if missing_baseline:
        raise ValueError(
            "Missing baseline columns: "
            f"{sorted(missing_baseline)}"
        )

    if missing_drifted:
        raise ValueError(
            "Missing drifted columns: "
            f"{sorted(missing_drifted)}"
        )

    baseline = (
        baseline_results[
            [
                "model_name",
                score_column,
            ]
        ]
        .rename(
            columns={
                score_column:
                    "baseline_score"
            }
        )
    )

    drifted = (
        drifted_results[
            [
                "model_name",
                score_column,
            ]
        ]
        .rename(
            columns={
                score_column:
                    "drifted_score"
            }
        )
    )

    merged = baseline.merge(
        drifted,
        on="model_name",
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No common models were found "
            "between the two periods."
        )

    merged[
        "degradation"
    ] = (
        merged["baseline_score"]
        - merged["drifted_score"]
    )

    merged[
        "degradation_ratio"
    ] = (
        merged["degradation"]
        / merged["baseline_score"].abs()
    )

    merged[
        "reliability_index"
    ] = np.clip(
        merged["drifted_score"]
        / merged["baseline_score"],
        0.0,
        1.0,
    )

    merged[
        "reliability_status"
    ] = merged[
        "degradation_ratio"
    ].apply(
        classify_reliability
    )

    return merged