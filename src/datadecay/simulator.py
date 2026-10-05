from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class DriftMetadata:
    """
    Ground-truth description of an injected drift event.
    """

    scenario: str
    affected_features: List[str]
    start_window: Optional[int]
    end_window: Optional[int]
    magnitude: float
    seed: int
    schedule: str
    configuration: dict

    def to_dict(self):
        return asdict(self)


class DriftSimulator:
    """
    Controlled temporal feature-distribution drift simulator.

    Supported scenarios:
    - stable
    - gradual
    - abrupt
    - recurring

    Only specified feature columns are modified.
    Timestamp and unrelated columns remain unchanged.
    """

    VALID_SCENARIOS = {
        "stable",
        "gradual",
        "abrupt",
        "recurring",
    }

    def __init__(self, seed=42):
        self.seed = int(seed)

        self.rng = np.random.default_rng(
            self.seed
        )

    def _validate_features(
        self,
        df,
        features,
        target_column=None,
    ):
        if not features:
            raise ValueError(
                "At least one drift feature is required."
            )

        missing = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Features not found: {missing}"
            )

        if target_column is not None:
            if target_column in features:
                raise ValueError(
                    "Target column cannot be drifted."
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
                "Drift simulation currently expects "
                "numeric features. "
                f"Non-numeric: {non_numeric}"
            )

    @staticmethod
    def _shift_value(
        series,
        magnitude,
    ):
        """
        Convert relative drift magnitude into
        an absolute feature shift using standard deviation.
        """
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        std = values.std()

        if pd.isna(std) or std == 0:
            return 0.0

        return float(
            magnitude * std
        )

    @staticmethod
    def _validate_magnitude(magnitude):
        if magnitude < 0:
            raise ValueError(
                "magnitude must be >= 0."
            )

    def stable(
        self,
        df,
        features,
        target_column=None,
    ):
        """
        Stable scenario: no distribution modification.
        """
        self._validate_features(
            df,
            features,
            target_column,
        )

        output = df.copy()

        metadata = DriftMetadata(
            scenario="stable",
            affected_features=[],
            start_window=None,
            end_window=None,
            magnitude=0.0,
            seed=self.seed,
            schedule="no_shift",
            configuration={},
        )

        return output, metadata

    def gradual(
        self,
        df,
        features,
        magnitude=0.20,
        start_window=0,
        end_window=None,
        target_column=None,
        baseline_df=None,
    ):
        """
        Gradual drift.

        Drift increases linearly from zero to the requested
        magnitude between start_window and end_window.
        """
        self._validate_features(
            df,
            features,
            target_column,
        )

        self._validate_magnitude(
            magnitude
        )

        output = df.copy()

        n = len(output)

        if n == 0:
            raise ValueError(
                "Cannot simulate drift on an empty dataframe."
            )

        if end_window is None:
            end_window = n - 1

        start_window = int(start_window)
        end_window = int(end_window)

        if start_window < 0:
            raise ValueError(
                "start_window must be >= 0."
            )

        if start_window >= n:
            raise ValueError(
                "start_window must be smaller than "
                "the dataframe length."
            )

        end_window = min(
            end_window,
            n - 1,
        )

        if end_window <= start_window:
            raise ValueError(
                "end_window must be greater than "
                "start_window."
            )

        if baseline_df is None:
            baseline_df = df.iloc[
                :start_window
            ]

        for feature in features:
            if feature not in baseline_df.columns:
                raise ValueError(
                    f"Baseline dataframe does not contain "
                    f"feature '{feature}'."
                )

            shift_amount = self._shift_value(
                baseline_df[feature],
                magnitude,
            )

            progress = np.zeros(n)

            drift_length = (
                end_window
                - start_window
                + 1
            )

            progress[
                start_window:end_window + 1
            ] = np.linspace(
                0.0,
                1.0,
                drift_length,
            )

            progress[
                end_window + 1:
            ] = 1.0

            values = output[
                feature
            ].astype(float)

            output[feature] = (
                values
                + progress * shift_amount
            )

        metadata = DriftMetadata(
            scenario="gradual",
            affected_features=list(features),
            start_window=start_window,
            end_window=end_window,
            magnitude=float(magnitude),
            seed=self.seed,
            schedule="linear_progression",
            configuration={
                "start_window": start_window,
                "end_window": end_window,
            },
        )

        return output, metadata

    def abrupt(
        self,
        df,
        features,
        magnitude=0.50,
        start_window=None,
        target_column=None,
        baseline_df=None,
    ):
        """
        Abrupt drift.

        A fixed distribution shift is applied from
        start_window onward.
        """
        self._validate_features(
            df,
            features,
            target_column,
        )

        self._validate_magnitude(
            magnitude
        )

        output = df.copy()

        n = len(output)

        if n == 0:
            raise ValueError(
                "Cannot simulate drift on an empty dataframe."
            )

        if start_window is None:
            start_window = n // 2

        start_window = int(start_window)

        if not 0 <= start_window < n:
            raise ValueError(
                "start_window must be within the dataframe."
            )

        if baseline_df is None:
            baseline_df = df.iloc[
                :start_window
            ]

        for feature in features:
            if feature not in baseline_df.columns:
                raise ValueError(
                    f"Baseline dataframe does not contain "
                    f"feature '{feature}'."
                )

            shift_amount = self._shift_value(
                baseline_df[feature],
                magnitude,
            )

            values = output[
                feature
            ].astype(float)

            output.loc[
                output.index[start_window:],
                feature,
            ] = (
                values.iloc[start_window:]
                + shift_amount
            ).values

        metadata = DriftMetadata(
            scenario="abrupt",
            affected_features=list(features),
            start_window=start_window,
            end_window=None,
            magnitude=float(magnitude),
            seed=self.seed,
            schedule="step_change",
            configuration={
                "start_window": start_window,
            },
        )

        return output, metadata

    def recurring(
        self,
        df,
        features,
        magnitude=0.30,
        cycle_length=4,
        target_column=None,
        baseline_df=None,
    ):
        """
        Recurring drift.

        Alternates between baseline and shifted regimes.
        """
        self._validate_features(
            df,
            features,
            target_column,
        )

        self._validate_magnitude(
            magnitude
        )

        if cycle_length < 2:
            raise ValueError(
                "cycle_length must be >= 2."
            )

        output = df.copy()

        n = len(output)

        if n == 0:
            raise ValueError(
                "Cannot simulate drift on an empty dataframe."
            )

        if baseline_df is None:
            baseline_df = df

        regime = (
            np.arange(n)
            // int(cycle_length)
        ) % 2

        for feature in features:
            if feature not in baseline_df.columns:
                raise ValueError(
                    f"Baseline dataframe does not contain "
                    f"feature '{feature}'."
                )

            shift_amount = self._shift_value(
                baseline_df[feature],
                magnitude,
            )

            values = output[
                feature
            ].astype(float)

            output[feature] = (
                values
                + regime * shift_amount
            )

        metadata = DriftMetadata(
            scenario="recurring",
            affected_features=list(features),
            start_window=0,
            end_window=n - 1,
            magnitude=float(magnitude),
            seed=self.seed,
            schedule="alternating_regimes",
            configuration={
                "cycle_length": int(
                    cycle_length
                )
            },
        )

        return output, metadata

    def simulate(
        self,
        df,
        scenario,
        features,
        magnitude=0.20,
        target_column=None,
        **kwargs,
    ):
        """
        Generic interface for all drift scenarios.
        """
        scenario = (
            scenario
            .lower()
            .strip()
        )

        if scenario not in self.VALID_SCENARIOS:
            raise ValueError(
                "scenario must be one of: "
                "stable, gradual, abrupt, recurring"
            )

        if scenario == "stable":
            return self.stable(
                df,
                features,
                target_column,
            )

        if scenario == "gradual":
            return self.gradual(
                df,
                features,
                magnitude=magnitude,
                target_column=target_column,
                **kwargs,
            )

        if scenario == "abrupt":
            return self.abrupt(
                df,
                features,
                magnitude=magnitude,
                target_column=target_column,
                **kwargs,
            )

        if scenario == "recurring":
            return self.recurring(
                df,
                features,
                magnitude=magnitude,
                target_column=target_column,
                **kwargs,
            )

        raise RuntimeError(
            "Unexpected drift scenario."
        )