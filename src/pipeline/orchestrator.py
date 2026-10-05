"""
DataDecay-X pipeline orchestration.

This module provides a lightweight orchestration layer for
running the major DataDecay-X stages in a controlled order:

1. Load and validate the dataset.
2. Create chronological reference/monitoring/evaluation splits.
3. Select useful numeric features.
4. Simulate controlled drift.
5. Calculate drift metrics and DSI.
6. Optionally evaluate synthetic-data quality.
7. Optionally perform recovery.
8. Return structured results to the caller.

The individual algorithms remain implemented in
src.datadecay. This module only coordinates them.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.datadecay import (
    DriftSimulator,
    calculate_dsi,
    calculate_feature_metrics,
    classify_dsi,
    load_electricity_dataset,
    numeric_features,
    temporal_split,
)


@dataclass
class PipelineResult:
    """
    Container for the outputs of a DataDecay-X pipeline run.
    """

    dataset: pd.DataFrame
    reference: pd.DataFrame
    monitoring: pd.DataFrame
    evaluation: pd.DataFrame

    features: list[str]

    scenario: str | None = None
    drifted_monitoring: pd.DataFrame | None = None

    feature_metrics: dict = field(
        default_factory=dict
    )

    dsi: float | None = None
    severity: str | None = None

    metadata: object | None = None

    def summary(self):
        """
        Return a compact dictionary describing the pipeline run.
        """

        return {
            "rows": len(self.dataset),
            "reference_rows": len(self.reference),
            "monitoring_rows": len(self.monitoring),
            "evaluation_rows": len(self.evaluation),
            "features": self.features,
            "scenario": self.scenario,
            "dsi": self.dsi,
            "severity": self.severity,
        }


class DataDecayPipeline:
    """
    Orchestrate the main DataDecay-X workflow.
    """

    def __init__(
        self,
        seed=42,
        train_fraction=0.60,
        monitor_fraction=0.20,
        feature_count=5,
    ):
        self.seed = seed
        self.train_fraction = train_fraction
        self.monitor_fraction = monitor_fraction
        self.feature_count = feature_count

        self.simulator = DriftSimulator(
            seed=seed
        )

    @staticmethod
    def load_dataset(path):
        """
        Load the electricity dataset.
        """

        return load_electricity_dataset(
            path
        )

    def split_dataset(self, df):
        """
        Create the chronological reference, monitoring,
        and evaluation periods.
        """

        return temporal_split(
            df,
            train_fraction=self.train_fraction,
            monitor_fraction=self.monitor_fraction,
        )

    def select_features(
        self,
        df,
        count=None,
    ):
        """
        Select the highest-variance numeric features.
        """

        if count is None:
            count = self.feature_count

        features = numeric_features(
            df,
            excluded=["timestamp"],
        )

        if not features:
            raise ValueError(
                "No numeric features were found."
            )

        variance = (
            df[features]
            .std()
            .dropna()
        )

        variance = variance[
            variance > 0
        ]

        if variance.empty:
            raise ValueError(
                "No non-zero-variance numeric "
                "features were found."
            )

        return (
            variance
            .sort_values(
                ascending=False
            )
            .head(count)
            .index
            .tolist()
        )

    def simulate_drift(
        self,
        scenario,
        reference,
        monitoring,
        features,
        magnitude=0.30,
        start_window=None,
        end_window=None,
        cycle_length=1000,
    ):
        """
        Apply one of the supported controlled drift scenarios.
        """

        scenario = scenario.lower()

        if scenario == "stable":

            drifted, metadata = (
                self.simulator.stable(
                    df=monitoring,
                    features=features,
                )
            )

        elif scenario == "gradual":

            if start_window is None:
                start_window = (
                    len(monitoring) // 4
                )

            if end_window is None:
                end_window = (
                    len(monitoring) * 3 // 4
                )

            drifted, metadata = (
                self.simulator.gradual(
                    df=monitoring,
                    features=features,
                    magnitude=magnitude,
                    start_window=start_window,
                    end_window=end_window,
                    baseline_df=reference,
                )
            )

        elif scenario == "abrupt":

            if start_window is None:
                start_window = (
                    len(monitoring) // 2
                )

            drifted, metadata = (
                self.simulator.abrupt(
                    df=monitoring,
                    features=features,
                    magnitude=magnitude,
                    start_window=start_window,
                    baseline_df=reference,
                )
            )

        elif scenario == "recurring":

            drifted, metadata = (
                self.simulator.recurring(
                    df=monitoring,
                    features=features,
                    magnitude=magnitude,
                    cycle_length=cycle_length,
                    baseline_df=reference,
                )
            )

        else:
            raise ValueError(
                "Unknown drift scenario: "
                f"{scenario}. Supported scenarios are: "
                "stable, gradual, abrupt, recurring."
            )

        return drifted, metadata

    def detect_drift(
        self,
        reference,
        current,
        features,
    ):
        """
        Calculate per-feature drift metrics, DSI, and severity.
        """

        feature_metrics = (
            calculate_feature_metrics(
                reference=reference,
                current=current,
                features=features,
            )
        )

        dsi = calculate_dsi(
            feature_metrics
        )

        severity = classify_dsi(
            dsi
        )

        return (
            feature_metrics,
            dsi,
            severity,
        )

    def run(
        self,
        df,
        scenario="stable",
        magnitude=0.30,
        start_window=None,
        end_window=None,
        cycle_length=1000,
    ):
        """
        Run the complete core drift-monitoring pipeline.

        Parameters
        ----------
        df : pandas.DataFrame
            Chronologically ordered source dataset.

        scenario : str
            One of stable, gradual, abrupt, recurring.

        magnitude : float
            Controlled drift magnitude.

        start_window : int, optional
            Drift start position.

        end_window : int, optional
            Drift end position for gradual drift.

        cycle_length : int
            Cycle length for recurring drift.

        Returns
        -------
        PipelineResult
        """

        if df is None:
            raise ValueError(
                "Input dataframe cannot be None."
            )

        if df.empty:
            raise ValueError(
                "Input dataframe cannot be empty."
            )

        reference, monitoring, evaluation = (
            self.split_dataset(df)
        )

        features = self.select_features(
            reference
        )

        (
            drifted_monitoring,
            metadata,
        ) = self.simulate_drift(
            scenario=scenario,
            reference=reference,
            monitoring=monitoring,
            features=features,
            magnitude=magnitude,
            start_window=start_window,
            end_window=end_window,
            cycle_length=cycle_length,
        )

        (
            feature_metrics,
            dsi,
            severity,
        ) = self.detect_drift(
            reference=reference,
            current=drifted_monitoring,
            features=features,
        )

        return PipelineResult(
            dataset=df,
            reference=reference,
            monitoring=monitoring,
            evaluation=evaluation,
            features=features,
            scenario=scenario,
            drifted_monitoring=drifted_monitoring,
            feature_metrics=feature_metrics,
            dsi=dsi,
            severity=severity,
            metadata=metadata,
        )


def run_pipeline(
    dataset_path,
    scenario="stable",
    seed=42,
    feature_count=5,
    magnitude=0.30,
):
    """
    Convenience function for running the pipeline from a dataset
    path.
    """

    dataset_path = Path(
        dataset_path
    )

    pipeline = DataDecayPipeline(
        seed=seed,
        feature_count=feature_count,
    )

    df = pipeline.load_dataset(
        dataset_path
    )

    return pipeline.run(
        df=df,
        scenario=scenario,
        magnitude=magnitude,
    )


__all__ = [
    "PipelineResult",
    "DataDecayPipeline",
    "run_pipeline",
]