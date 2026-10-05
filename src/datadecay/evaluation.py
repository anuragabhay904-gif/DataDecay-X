from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    """
    Container for model evaluation results.
    """

    model_name: str
    metrics: dict
    metadata: dict

    def to_dict(self):
        return {
            "model_name": self.model_name,
            **self.metrics,
            **self.metadata,
        }


def _validate_arrays(
    y_true,
    y_pred,
):
    """
    Validate actual and predicted arrays.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)

    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Evaluation arrays cannot be empty."
        )

    if not np.isfinite(y_true).all():
        raise ValueError(
            "y_true contains non-finite values."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "y_pred contains non-finite values."
        )

    return (
        y_true,
        y_pred,
    )


def mean_absolute_error(
    y_true,
    y_pred,
):
    """
    Mean Absolute Error.
    """

    y_true, y_pred = _validate_arrays(
        y_true,
        y_pred,
    )

    return float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )


def mean_squared_error(
    y_true,
    y_pred,
):
    """
    Mean Squared Error.
    """

    y_true, y_pred = _validate_arrays(
        y_true,
        y_pred,
    )

    return float(
        np.mean(
            (
                y_true - y_pred
            ) ** 2
        )
    )


def root_mean_squared_error(
    y_true,
    y_pred,
):
    """
    Root Mean Squared Error.
    """

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    return float(
        np.sqrt(mse)
    )


def mean_absolute_percentage_error(
    y_true,
    y_pred,
):
    """
    Mean Absolute Percentage Error.

    Zero-valued actual observations are excluded
    from the percentage calculation.
    """

    y_true, y_pred = _validate_arrays(
        y_true,
        y_pred,
    )

    non_zero = (
        y_true != 0
    )

    if not non_zero.any():
        return np.nan

    percentage_error = (
        np.abs(
            (
                y_true[non_zero]
                - y_pred[non_zero]
            )
            / y_true[non_zero]
        )
        * 100.0
    )

    return float(
        np.mean(
            percentage_error
        )
    )


def r_squared(
    y_true,
    y_pred,
):
    """
    Calculate R-squared.
    """

    y_true, y_pred = _validate_arrays(
        y_true,
        y_pred,
    )

    total_sum_squares = np.sum(
        (
            y_true
            - np.mean(y_true)
        ) ** 2
    )

    if total_sum_squares == 0:
        return 0.0

    residual_sum_squares = np.sum(
        (
            y_true
            - y_pred
        ) ** 2
    )

    return float(
        1
        - (
            residual_sum_squares
            / total_sum_squares
        )
    )


def classification_accuracy(
    y_true,
    y_pred,
):
    """
    Calculate classification accuracy.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Classification arrays cannot be empty."
        )

    return float(
        np.mean(
            y_true == y_pred
        )
    )


def classification_f1(
    y_true,
    y_pred,
):
    """
    Calculate weighted F1 score.

    Implemented without requiring the model
    layer so the evaluation module remains
    reusable.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have "
            "the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Classification arrays cannot be empty."
        )

    labels = np.unique(
        np.concatenate(
            [
                y_true,
                y_pred,
            ]
        )
    )

    weighted_f1 = 0.0

    total_support = 0

    for label in labels:

        true_positive = np.sum(
            (
                y_true == label
            )
            & (
                y_pred == label
            )
        )

        false_positive = np.sum(
            (
                y_true != label
            )
            & (
                y_pred == label
            )
        )

        false_negative = np.sum(
            (
                y_true == label
            )
            & (
                y_pred != label
            )
        )

        support = np.sum(
            y_true == label
        )

        if (
            true_positive
            + false_positive
            == 0
        ):
            precision = 0.0

        else:
            precision = (
                true_positive
                / (
                    true_positive
                    + false_positive
                )
            )

        if (
            true_positive
            + false_negative
            == 0
        ):
            recall = 0.0

        else:
            recall = (
                true_positive
                / (
                    true_positive
                    + false_negative
                )
            )

        if (
            precision + recall
            == 0
        ):
            f1 = 0.0

        else:
            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        weighted_f1 += (
            f1 * support
        )

        total_support += support

    if total_support == 0:
        return 0.0

    return float(
        weighted_f1
        / total_support
    )


def evaluate_regression(
    y_true,
    y_pred,
    model_name="model",
):
    """
    Evaluate a regression model.
    """

    metrics = {
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": root_mean_squared_error(
            y_true,
            y_pred,
        ),
        "mape": mean_absolute_percentage_error(
            y_true,
            y_pred,
        ),
        "r2": r_squared(
            y_true,
            y_pred,
        ),
    }

    return EvaluationResult(
        model_name=model_name,
        metrics=metrics,
        metadata={
            "task_type": "regression",
            "sample_count": len(
                y_true
            ),
        },
    )


def evaluate_classification(
    y_true,
    y_pred,
    model_name="model",
):
    """
    Evaluate a classification model.
    """

    metrics = {
        "accuracy": classification_accuracy(
            y_true,
            y_pred,
        ),
        "f1": classification_f1(
            y_true,
            y_pred,
        ),
    }

    return EvaluationResult(
        model_name=model_name,
        metrics=metrics,
        metadata={
            "task_type": "classification",
            "sample_count": len(
                y_true
            ),
        },
    )


def evaluate_predictions(
    y_true,
    y_pred,
    task_type,
    model_name="model",
):
    """
    Evaluate predictions according to task type.
    """

    task_type = (
        task_type.lower()
        .strip()
    )

    if task_type == "regression":

        return evaluate_regression(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
        )

    if task_type == "classification":

        return evaluate_classification(
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_name,
        )

    raise ValueError(
        "task_type must be either "
        "'regression' or 'classification'."
    )


def results_to_dataframe(
    results,
):
    """
    Convert EvaluationResult objects
    into a dataframe.
    """

    if not results:
        return pd.DataFrame()

    rows = [
        result.to_dict()
        for result in results
    ]

    return pd.DataFrame(
        rows
    )


def compare_models(
    y_true,
    predictions,
    task_type,
):
    """
    Compare multiple model prediction sets.

    Parameters
    ----------
    y_true:
        Ground-truth values.

    predictions:
        Dictionary mapping model names
        to prediction arrays.

    task_type:
        'classification' or 'regression'.
    """

    if not predictions:
        raise ValueError(
            "At least one prediction set "
            "is required."
        )

    results = []

    for model_name, y_pred in (
        predictions.items()
    ):

        result = evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            task_type=task_type,
            model_name=model_name,
        )

        results.append(
            result
        )

    return results_to_dataframe(
        results
    )


def temporal_evaluation(
    model,
    reference_X,
    reference_y,
    evaluation_X,
    evaluation_y,
    task_type,
    model_name="model",
):
    """
    Train on the historical/reference period
    and evaluate on a later chronological period.

    No random shuffling is performed.
    """

    model.fit(
        reference_X,
        reference_y,
    )

    predictions = model.predict(
        evaluation_X
    )

    return evaluate_predictions(
        y_true=evaluation_y,
        y_pred=predictions,
        task_type=task_type,
        model_name=model_name,
    )