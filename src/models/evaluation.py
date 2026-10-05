"""
Model evaluation helpers for DataDecay-X.

This module provides lightweight evaluation functions for the
legacy model layer. The main ``src.datadecay.evaluation`` module
contains the project's broader evaluation implementation; these
helpers provide a clean interface for the legacy models package.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class ModelEvaluation:
    """Container for model evaluation results."""

    model_name: str
    task: str
    metrics: dict

    def to_dict(self):
        """Return evaluation results as a dictionary."""

        result = {
            "model_name": self.model_name,
            "task": self.task,
        }

        result.update(self.metrics)

        return result


def _validate_predictions(
    y_true,
    y_pred,
):
    """Validate true and predicted values."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)

    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate empty predictions."
        )

    return y_true, y_pred


def regression_metrics(
    y_true,
    y_pred,
):
    """
    Calculate standard regression metrics.

    Returns
    -------
    dict
        MAE, MSE, RMSE, and R2.
    """

    y_true, y_pred = _validate_predictions(
        y_true,
        y_pred,
    )

    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    if not np.isfinite(y_true).all():
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "y_pred contains NaN or infinite values."
        )

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "mse": float(mse),
        "rmse": float(
            np.sqrt(mse)
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


def classification_metrics(
    y_true,
    y_pred,
):
    """
    Calculate standard classification metrics.

    Returns
    -------
    dict
        Accuracy and weighted F1 score.
    """

    y_true, y_pred = _validate_predictions(
        y_true,
        y_pred,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def evaluate_model(
    model,
    X,
    y,
    task="regression",
    model_name=None,
):
    """
    Evaluate a fitted model on a dataset.

    Parameters
    ----------
    model
        Fitted model exposing ``predict``.

    X
        Evaluation features.

    y
        Ground-truth target.

    task
        ``regression`` or ``classification``.

    model_name
        Optional display name for the model.
    """

    if not hasattr(model, "predict"):
        raise TypeError(
            "model must provide a predict() method."
        )

    task = task.lower()

    if task not in {
        "regression",
        "classification",
    }:
        raise ValueError(
            "task must be 'regression' or 'classification'."
        )

    predictions = model.predict(X)

    if task == "regression":
        metrics = regression_metrics(
            y,
            predictions,
        )
    else:
        metrics = classification_metrics(
            y,
            predictions,
        )

    if model_name is None:
        model_name = type(model).__name__

    return ModelEvaluation(
        model_name=model_name,
        task=task,
        metrics=metrics,
    )


def evaluate_predictions(
    y_true,
    y_pred,
    task="regression",
    model_name="model",
):
    """
    Evaluate already-generated predictions.
    """

    task = task.lower()

    if task == "regression":
        metrics = regression_metrics(
            y_true,
            y_pred,
        )

    elif task == "classification":
        metrics = classification_metrics(
            y_true,
            y_pred,
        )

    else:
        raise ValueError(
            "task must be 'regression' or 'classification'."
        )

    return ModelEvaluation(
        model_name=model_name,
        task=task,
        metrics=metrics,
    )


def compare_models(
    evaluations,
    primary_metric,
    higher_is_better=True,
):
    """
    Compare multiple ModelEvaluation objects.

    Parameters
    ----------
    evaluations : iterable
        ModelEvaluation instances.

    primary_metric : str
        Metric used for ranking.

    higher_is_better : bool
        Whether larger metric values represent better
        performance.

    Returns
    -------
    list[dict]
        Evaluations sorted by the requested metric.
    """

    evaluations = list(evaluations)

    if not evaluations:
        return []

    rows = []

    for evaluation in evaluations:

        if not isinstance(
            evaluation,
            ModelEvaluation,
        ):
            raise TypeError(
                "All items must be ModelEvaluation instances."
            )

        if primary_metric not in evaluation.metrics:
            raise ValueError(
                f"Metric '{primary_metric}' is not available "
                f"for model '{evaluation.model_name}'."
            )

        rows.append(
            evaluation.to_dict()
        )

    return sorted(
        rows,
        key=lambda row: row[primary_metric],
        reverse=higher_is_better,
    )


def evaluation_table(
    evaluations,
):
    """
    Convert model evaluations into a list of dictionaries.

    This is intentionally dependency-light so the result can
    easily be converted to a pandas DataFrame by callers.
    """

    return [
        evaluation.to_dict()
        for evaluation in evaluations
    ]


__all__ = [
    "ModelEvaluation",
    "regression_metrics",
    "classification_metrics",
    "evaluate_model",
    "evaluate_predictions",
    "compare_models",
    "evaluation_table",
]