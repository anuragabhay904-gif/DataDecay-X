"""
Model recovery helpers for DataDecay-X.

This module provides simple model-recovery utilities for the
legacy model layer. It focuses on retraining a supplied model
using updated reference data and evaluating the recovered model
on a chronological evaluation period.

The actual data-value recovery algorithms live in
``src.datadecay.recovery``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RecoveryResult:
    """Container for a model recovery run."""

    model_name: str
    original_metrics: dict
    recovered_metrics: dict
    improvement: dict
    recovered_model: object

    def to_dict(self):
        """Return serializable recovery information."""

        return {
            "model_name": self.model_name,
            "original_metrics": self.original_metrics,
            "recovered_metrics": self.recovered_metrics,
            "improvement": self.improvement,
        }


def _metric_improvement(
    original,
    recovered,
    higher_is_better=True,
):
    """
    Calculate relative improvement between two metric values.

    For metrics where larger is better, improvement is:

        recovered - original

    For metrics where smaller is better, improvement is:

        original - recovered
    """

    original = float(original)
    recovered = float(recovered)

    if higher_is_better:
        return recovered - original

    return original - recovered


def calculate_recovery_improvement(
    original_metrics,
    recovered_metrics,
):
    """
    Compare original and recovered model metrics.

    The direction of improvement is determined from the metric
    name:

    Higher is better:
        accuracy
        f1
        f1_weighted
        r2

    Lower is better:
        mae
        mse
        rmse
        mape
    """

    higher_is_better_metrics = {
        "accuracy",
        "f1",
        "f1_weighted",
        "r2",
    }

    lower_is_better_metrics = {
        "mae",
        "mse",
        "rmse",
        "mape",
    }

    shared_metrics = (
        set(original_metrics)
        & set(recovered_metrics)
    )

    improvement = {}

    for metric in sorted(shared_metrics):

        if metric in higher_is_better_metrics:
            higher_is_better = True

        elif metric in lower_is_better_metrics:
            higher_is_better = False

        else:
            # Unknown metrics are treated as higher-is-better
            # unless the caller explicitly handles them elsewhere.
            higher_is_better = True

        improvement[metric] = _metric_improvement(
            original_metrics[metric],
            recovered_metrics[metric],
            higher_is_better=higher_is_better,
        )

    return improvement


def recover_model(
    model_factory,
    X_original,
    y_original,
    X_recovery,
    y_recovery,
):
    """
    Retrain a model using recovery data.

    Parameters
    ----------
    model_factory : callable
        Function that creates a fresh model instance.

    X_original, y_original
        Original training data.

    X_recovery, y_recovery
        Updated/recovery training data.

    Returns
    -------
    tuple
        ``(original_model, recovered_model)``
    """

    if not callable(model_factory):
        raise TypeError(
            "model_factory must be callable."
        )

    original_model = model_factory()
    recovered_model = model_factory()

    original_model.fit(
        X_original,
        y_original,
    )

    recovered_model.fit(
        X_recovery,
        y_recovery,
    )

    return (
        original_model,
        recovered_model,
    )


def evaluate_recovery(
    original_model,
    recovered_model,
    X_evaluation,
    y_evaluation,
    evaluation_function,
    model_name=None,
):
    """
    Evaluate an original model against a recovered model.

    ``evaluation_function`` should accept:

        evaluation_function(y_true, y_pred)

    and return a dictionary of metrics.
    """

    if not hasattr(
        original_model,
        "predict",
    ):
        raise TypeError(
            "original_model must provide predict()."
        )

    if not hasattr(
        recovered_model,
        "predict",
    ):
        raise TypeError(
            "recovered_model must provide predict()."
        )

    if not callable(evaluation_function):
        raise TypeError(
            "evaluation_function must be callable."
        )

    original_predictions = (
        original_model.predict(
            X_evaluation
        )
    )

    recovered_predictions = (
        recovered_model.predict(
            X_evaluation
        )
    )

    original_metrics = evaluation_function(
        y_evaluation,
        original_predictions,
    )

    recovered_metrics = evaluation_function(
        y_evaluation,
        recovered_predictions,
    )

    improvement = calculate_recovery_improvement(
        original_metrics,
        recovered_metrics,
    )

    if model_name is None:
        model_name = type(
            recovered_model
        ).__name__

    return RecoveryResult(
        model_name=model_name,
        original_metrics=original_metrics,
        recovered_metrics=recovered_metrics,
        improvement=improvement,
        recovered_model=recovered_model,
    )


def recovery_success(
    result,
    required_metrics=None,
):
    """
    Determine whether recovery improved the requested metrics.

    Parameters
    ----------
    result : RecoveryResult
        Recovery evaluation result.

    required_metrics : iterable, optional
        Metrics that must improve. If omitted, all available
        improvement values are considered.

    Returns
    -------
    bool
        True when all requested metrics improved.
    """

    if not isinstance(
        result,
        RecoveryResult,
    ):
        raise TypeError(
            "result must be a RecoveryResult."
        )

    if required_metrics is None:
        required_metrics = (
            result.improvement.keys()
        )

    required_metrics = list(
        required_metrics
    )

    if not required_metrics:
        return False

    for metric in required_metrics:

        if metric not in result.improvement:
            raise ValueError(
                f"Recovery metric '{metric}' "
                "was not available."
            )

        if result.improvement[metric] <= 0:
            return False

    return True


def summarize_recovery(
    results,
):
    """
    Convert multiple RecoveryResult objects into dictionaries.
    """

    summaries = []

    for result in results:

        if not isinstance(
            result,
            RecoveryResult,
        ):
            raise TypeError(
                "All results must be RecoveryResult instances."
            )

        summaries.append(
            result.to_dict()
        )

    return summaries


__all__ = [
    "RecoveryResult",
    "calculate_recovery_improvement",
    "recover_model",
    "evaluate_recovery",
    "recovery_success",
    "summarize_recovery",
]