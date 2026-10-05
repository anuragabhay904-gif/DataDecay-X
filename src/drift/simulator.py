"""
Legacy drift simulator for DataDecay-X.

This module provides a compatibility layer around the main
``src.datadecay.simulator`` implementation while exposing a
simple interface for the legacy ``src.drift`` package.
"""

from dataclasses import dataclass

import pandas as pd

from src.datadecay.simulator import DriftSimulator


@dataclass
class DriftResult:
    """
    Result of a controlled drift simulation.
    """

    scenario: str
    data: pd.DataFrame
    metadata: object

    def to_dict(self):
        """Return the result metadata as a dictionary."""

        if hasattr(
            self.metadata,
            "__dict__",
        ):
            metadata = dict(
                self.metadata.__dict__
            )
        else:
            metadata = self.metadata

        return {
            "scenario": self.scenario,
            "metadata": metadata,
        }


class LegacyDriftSimulator:
    """
    Compatibility wrapper for the main DataDecay-X drift simulator.
    """

    def __init__(self, seed=42):
        self.seed = seed

        self.simulator = DriftSimulator(
            seed=seed
        )

    @staticmethod
    def _validate_dataframe(df):
        """Validate simulation input."""

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "df cannot be empty."
            )

        return df

    @staticmethod
    def _validate_features(
        df,
        features,
    ):
        """Validate selected drift features."""

        if features is None:
            features = [
                column
                for column in df.columns
                if column != "timestamp"
                and pd.api.types.is_numeric_dtype(
                    df[column]
                )
            ]

        features = list(features)

        if not features:
            raise ValueError(
                "At least one numeric feature is required."
            )

        missing = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                "Features not found in dataframe: "
                f"{missing}"
            )

        non_numeric = [
            feature
            for feature in features
            if not pd.api.types.is_numeric_dtype(
                df[feature]
            )
        ]

        if non_numeric:
            raise ValueError(
                "Drift features must be numeric: "
                f"{non_numeric}"
            )

        return features

    def stable(
        self,
        df,
        features=None,
    ):
        """Generate the stable scenario."""

        df = self._validate_dataframe(df)

        features = self._validate_features(
            df,
            features,
        )

        data, metadata = (
            self.simulator.stable(
                df=df,
                features=features,
            )
        )

        return DriftResult(
            scenario="stable",
            data=data,
            metadata=metadata,
        )

    def gradual(
        self,
        df,
        features=None,
        magnitude=0.30,
        start_window=None,
        end_window=None,
        baseline_df=None,
    ):
        """Generate gradual drift."""

        df = self._validate_dataframe(df)

        features = self._validate_features(
            df,
            features,
        )

        if start_window is None:
            start_window = len(df) // 4

        if end_window is None:
            end_window = (
                len(df) * 3 // 4
            )

        if start_window < 0:
            raise ValueError(
                "start_window cannot be negative."
            )

        if end_window <= start_window:
            raise ValueError(
                "end_window must be greater than start_window."
            )

        if end_window > len(df):
            raise ValueError(
                "end_window cannot exceed dataframe length."
            )

        data, metadata = (
            self.simulator.gradual(
                df=df,
                features=features,
                magnitude=magnitude,
                start_window=start_window,
                end_window=end_window,
                baseline_df=baseline_df,
            )
        )

        return DriftResult(
            scenario="gradual",
            data=data,
            metadata=metadata,
        )

    def abrupt(
        self,
        df,
        features=None,
        magnitude=0.30,
        start_window=None,
        baseline_df=None,
    ):
        """Generate abrupt drift."""

        df = self._validate_dataframe(df)

        features = self._validate_features(
            df,
            features,
        )

        if start_window is None:
            start_window = len(df) // 2

        if not 0 <= start_window < len(df):
            raise ValueError(
                "start_window must be within the dataframe."
            )

        data, metadata = (
            self.simulator.abrupt(
                df=df,
                features=features,
                magnitude=magnitude,
                start_window=start_window,
                baseline_df=baseline_df,
            )
        )

        return DriftResult(
            scenario="abrupt",
            data=data,
            metadata=metadata,
        )

    def recurring(
        self,
        df,
        features=None,
        magnitude=0.30,
        cycle_length=1000,
        baseline_df=None,
    ):
        """Generate recurring drift."""

        df = self._validate_dataframe(df)

        features = self._validate_features(
            df,
            features,
        )

        if cycle_length <= 0:
            raise ValueError(
                "cycle_length must be greater than zero."
            )

        data, metadata = (
            self.simulator.recurring(
                df=df,
                features=features,
                magnitude=magnitude,
                cycle_length=cycle_length,
                baseline_df=baseline_df,
            )
        )

        return DriftResult(
            scenario="recurring",
            data=data,
            metadata=metadata,
        )

    def simulate(
        self,
        scenario,
        df,
        features=None,
        magnitude=0.30,
        start_window=None,
        end_window=None,
        cycle_length=1000,
        baseline_df=None,
    ):
        """
        Run a named drift scenario.

        Parameters
        ----------
        scenario : str
            One of ``stable``, ``gradual``, ``abrupt``,
            or ``recurring``.

        df : pandas.DataFrame
            Monitoring data.

        features : iterable, optional
            Numeric columns affected by drift.

        magnitude : float
            Drift magnitude.

        start_window : int, optional
            Drift start position.

        end_window : int, optional
            End position for gradual drift.

        cycle_length : int
            Cycle length for recurring drift.

        baseline_df : pandas.DataFrame, optional
            Reference data used to establish the baseline
            distribution.

        Returns
        -------
        DriftResult
        """

        if not isinstance(
            scenario,
            str,
        ):
            raise TypeError(
                "scenario must be a string."
            )

        scenario = scenario.strip().lower()

        if scenario == "stable":
            return self.stable(
                df=df,
                features=features,
            )

        if scenario == "gradual":
            return self.gradual(
                df=df,
                features=features,
                magnitude=magnitude,
                start_window=start_window,
                end_window=end_window,
                baseline_df=baseline_df,
            )

        if scenario == "abrupt":
            return self.abrupt(
                df=df,
                features=features,
                magnitude=magnitude,
                start_window=start_window,
                baseline_df=baseline_df,
            )

        if scenario == "recurring":
            return self.recurring(
                df=df,
                features=features,
                magnitude=magnitude,
                cycle_length=cycle_length,
                baseline_df=baseline_df,
            )

        raise ValueError(
            f"Unknown drift scenario: {scenario}. "
            "Supported scenarios are: stable, gradual, "
            "abrupt, recurring."
        )


def simulate_drift(
    df,
    scenario,
    features=None,
    magnitude=0.30,
    seed=42,
    start_window=None,
    end_window=None,
    cycle_length=1000,
    baseline_df=None,
):
    """
    Convenience function for running a drift scenario.
    """

    simulator = LegacyDriftSimulator(
        seed=seed
    )

    return simulator.simulate(
        scenario=scenario,
        df=df,
        features=features,
        magnitude=magnitude,
        start_window=start_window,
        end_window=end_window,
        cycle_length=cycle_length,
        baseline_df=baseline_df,
    )


__all__ = [
    "DriftResult",
    "LegacyDriftSimulator",
    "simulate_drift",
]