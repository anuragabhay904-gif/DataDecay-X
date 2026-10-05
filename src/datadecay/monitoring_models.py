from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import (
    evaluate_predictions,
)
from .reliability import (
    evaluate_reliability,
)


@dataclass
class MonitoringResult:
    """
    Combined model-monitoring result.
    """

    model_name: str
    period: str
    performance: dict
    reliability: dict
    drift_severity: str
    dsi: float

    def to_dict(self):
        result = {
            "model_name": self.model_name,
            "period": self.period,
            "drift_severity": self.drift_severity,
            "dsi": self.dsi,
        }

        result.update(
            self.performance
        )

        result.update(
            {
                f"reliability_{key}": value
                for key, value in self.reliability.items()
            }
        )

        return result


def validate_monitoring_inputs(
    y_true,
    y_pred,
):
    """
    Validate actual and predicted values.
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
            "Monitoring inputs cannot be empty."
        )

    return (
        y_true,
        y_pred,
    )


def monitor_model_performance(
    y_true,
    y_pred,
    task_type,
    model_name="model",
):
    """
    Calculate model performance for one period.
    """

    (
        y_true,
        y_pred,
    ) = validate_monitoring_inputs(
        y_true,
        y_pred,
    )

    result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        task_type=task_type,
        model_name=model_name,
    )

    return result.to_dict()


def monitor_model_reliability(
    model_name,
    baseline_score,
    current_score,
):
    """
    Calculate reliability degradation
    between baseline and current performance.
    """

    result = evaluate_reliability(
        model_name=model_name,
        baseline_score=baseline_score,
        current_score=current_score,
    )

    return result.to_dict()


def build_monitoring_result(
    model_name,
    period,
    performance,
    reliability,
    drift_severity="UNKNOWN",
    dsi=np.nan,
):
    """
    Combine performance, reliability and
    drift information into one monitoring result.
    """

    return MonitoringResult(
        model_name=model_name,
        period=period,
        performance=performance,
        reliability=reliability,
        drift_severity=drift_severity,
        dsi=float(dsi)
        if not pd.isna(dsi)
        else np.nan,
    )


def monitor_model(
    model,
    reference_X,
    reference_y,
    current_X,
    current_y,
    task_type,
    model_name="model",
    period="current",
    drift_severity="UNKNOWN",
    dsi=np.nan,
):
    """
    Train a model on the reference period,
    then evaluate it on reference and current data.

    The current period is evaluated without
    retraining the model.

    This is important for measuring degradation
    caused by changing data.
    """

    model.fit(
        reference_X,
        reference_y,
    )

    reference_predictions = (
        model.predict(
            reference_X
        )
    )

    current_predictions = (
        model.predict(
            current_X
        )
    )

    reference_performance = (
        evaluate_predictions(
            y_true=reference_y,
            y_pred=reference_predictions,
            task_type=task_type,
            model_name=model_name,
        )
    )

    current_performance = (
        evaluate_predictions(
            y_true=current_y,
            y_pred=current_predictions,
            task_type=task_type,
            model_name=model_name,
        )
    )

    reference_dict = (
        reference_performance.to_dict()
    )

    current_dict = (
        current_performance.to_dict()
    )

    score_name = (
        _primary_score_name(
            task_type
        )
    )

    baseline_score = (
        reference_dict[score_name]
    )

    current_score = (
        current_dict[score_name]
    )

    reliability = (
        monitor_model_reliability(
            model_name=model_name,
            baseline_score=baseline_score,
            current_score=current_score,
        )
    )

    return build_monitoring_result(
        model_name=model_name,
        period=period,
        performance=current_dict,
        reliability=reliability,
        drift_severity=drift_severity,
        dsi=dsi,
    )


def _primary_score_name(
    task_type,
):
    """
    Select the primary performance score
    used for reliability calculations.

    Higher is better for both selected scores.
    """

    task_type = (
        task_type.lower()
        .strip()
    )

    if task_type == "classification":
        return "accuracy"

    if task_type == "regression":
        return "r2"

    raise ValueError(
        "task_type must be either "
        "'classification' or 'regression'."
    )


def compare_model_monitoring(
    models,
    reference_X,
    reference_y,
    current_X,
    current_y,
    task_type,
    period="current",
    drift_severity="UNKNOWN",
    dsi=np.nan,
):
    """
    Monitor several models using the same
    chronological reference/current periods.

    Parameters
    ----------
    models:
        Dictionary mapping model names to
        unfitted model objects.
    """

    if not models:
        raise ValueError(
            "At least one model is required."
        )

    results = []

    for model_name, model in (
        models.items()
    ):

        result = monitor_model(
            model=model,
            reference_X=reference_X,
            reference_y=reference_y,
            current_X=current_X,
            current_y=current_y,
            task_type=task_type,
            model_name=model_name,
            period=period,
            drift_severity=drift_severity,
            dsi=dsi,
        )

        results.append(
            result.to_dict()
        )

    return pd.DataFrame(
        results
    )


def summarize_model_monitoring(
    monitoring_results,
):
    """
    Produce a compact summary of model monitoring.
    """

    if monitoring_results is None:
        raise ValueError(
            "monitoring_results cannot be None."
        )

    if not isinstance(
        monitoring_results,
        pd.DataFrame,
    ):
        raise TypeError(
            "monitoring_results must be "
            "a pandas DataFrame."
        )

    if monitoring_results.empty:
        return {
            "model_count": 0,
            "mean_dsi": None,
            "mean_reliability_index": None,
            "mean_degradation": None,
        }

    reliability_column = (
        "reliability_reliability_index"
    )

    degradation_column = (
        "reliability_degradation"
    )

    dsi_column = "dsi"

    summary = {
        "model_count": int(
            len(monitoring_results)
        ),
        "mean_dsi": float(
            monitoring_results[
                dsi_column
            ].mean()
        ),
    }

    if reliability_column in (
        monitoring_results.columns
    ):
        summary[
            "mean_reliability_index"
        ] = float(
            monitoring_results[
                reliability_column
            ].mean()
        )
    else:
        summary[
            "mean_reliability_index"
        ] = None

    if degradation_column in (
        monitoring_results.columns
    ):
        summary[
            "mean_degradation"
        ] = float(
            monitoring_results[
                degradation_column
            ].mean()
        )
    else:
        summary[
            "mean_degradation"
        ] = None

    return summary