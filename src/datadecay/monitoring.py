from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .detection import (
    calculate_feature_metrics,
    calculate_dsi,
    classify_dsi,
)


def monitor_window(
    reference_df,
    current_df,
    features,
    psi_bins=10,
    psi_reference=0.25,
    wasserstein_reference=1.0,
    weights=None,
):
    """
    Monitor one temporal window against a reference period.

    Returns
    -------
    metrics
        Feature-level drift metrics.

    dsi
        Drift Severity Index.

    severity
        LOW / MODERATE / HIGH / SEVERE / UNKNOWN.
    """

    metrics = calculate_feature_metrics(
        reference_df=reference_df,
        current_df=current_df,
        features=features,
        psi_bins=psi_bins,
    )

    dsi = calculate_dsi(
        metrics,
        psi_reference=psi_reference,
        wasserstein_reference=wasserstein_reference,
        weights=weights,
    )

    severity = classify_dsi(
        dsi
    )

    return (
        metrics,
        dsi,
        severity,
    )


def summarize_monitoring(
    metrics,
    dsi,
    severity,
):
    """
    Convert monitoring results into
    a compact summary dictionary.
    """

    if metrics.empty:
        return {
            "dsi": dsi,
            "severity": severity,
            "feature_count": 0,
            "mean_psi": None,
            "max_psi": None,
            "mean_ks": None,
            "max_ks": None,
            "mean_wasserstein": None,
            "max_wasserstein": None,
        }

    return {
        "dsi": float(dsi),
        "severity": severity,
        "feature_count": int(
            len(metrics)
        ),
        "mean_psi": float(
            metrics["psi"].mean()
        ),
        "max_psi": float(
            metrics["psi"].max()
        ),
        "mean_ks": float(
            metrics["ks_statistic"].mean()
        ),
        "max_ks": float(
            metrics["ks_statistic"].max()
        ),
        "mean_wasserstein": float(
            metrics["wasserstein"].mean()
        ),
        "max_wasserstein": float(
            metrics["wasserstein"].max()
        ),
    }


class ReliabilityMonitor:
    """
    Monitor predictive model reliability over time.

    Reliability is based on classification performance
    metrics and their change relative to a stored baseline.
    """

    def __init__(
        self,
        baseline_metrics=None,
    ):
        self.baseline_metrics = (
            dict(baseline_metrics)
            if baseline_metrics is not None
            else None
        )

    def calculate_metrics(
        self,
        y_true,
        y_pred,
    ):
        """
        Calculate predictive reliability metrics.
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred must have "
                "the same length."
            )

        if len(y_true) == 0:
            raise ValueError(
                "y_true and y_pred cannot be empty."
            )

        accuracy = float(
            accuracy_score(
                y_true,
                y_pred,
            )
        )

        precision = float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        )

        recall = float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        )

        f1 = float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": matrix,
            "mean_confidence": accuracy,
        }

    def set_baseline(
        self,
        y_true,
        y_pred,
    ):
        """
        Calculate and store baseline reliability metrics.
        """

        self.baseline_metrics = (
            self.calculate_metrics(
                y_true,
                y_pred,
            )
        )

        return self.baseline_metrics

    def calculate_reliability_degradation(
        self,
        current_metrics,
    ):
        """
        Calculate average performance degradation
        relative to the stored baseline.

        The result is clipped to [0, 1].
        """

        if self.baseline_metrics is None:
            raise RuntimeError(
                "Baseline metrics must be set before "
                "calculating reliability degradation."
            )

        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]

        degradations = []

        for name in metric_names:
            baseline = float(
                self.baseline_metrics[name]
            )
            current = float(
                current_metrics[name]
            )

            degradations.append(
                max(
                    0.0,
                    baseline - current,
                )
            )

        return float(
            np.clip(
                np.mean(degradations),
                0.0,
                1.0,
            )
        )

    def calculate_model_health(
        self,
        current_metrics,
    ):
        """
        Calculate model health as the inverse
        of reliability degradation.
        """

        degradation = (
            self.calculate_reliability_degradation(
                current_metrics
            )
        )

        return float(
            np.clip(
                1.0 - degradation,
                0.0,
                1.0,
            )
        )

    def monitor_windows(
        self,
        windows,
    ):
        """
        Monitor multiple chronological prediction windows.

        Each window must contain:
            window_id
            y_true
            y_pred
        """

        history = []

        for window in windows:
            metrics = self.calculate_metrics(
                window["y_true"],
                window["y_pred"],
            )

            if self.baseline_metrics is None:
                self.baseline_metrics = dict(
                    metrics
                )

            degradation = (
                self.calculate_reliability_degradation(
                    metrics
                )
            )

            health = (
                self.calculate_model_health(
                    metrics
                )
            )

            history.append(
                {
                    "window_id": window[
                        "window_id"
                    ],
                    "accuracy": metrics[
                        "accuracy"
                    ],
                    "precision": metrics[
                        "precision"
                    ],
                    "recall": metrics[
                        "recall"
                    ],
                    "f1": metrics[
                        "f1"
                    ],
                    "reliability_degradation": (
                        degradation
                    ),
                    "model_health": health,
                }
            )

        return pd.DataFrame(history)