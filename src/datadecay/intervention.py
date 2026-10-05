"""
DataDecay-X
Person 2 — Intervention Engine

intervention.py

Converts reliability monitoring and forecasting results into
a rule-based system state:

    STABLE
       ↓
    WATCH
       ↓
    RECOVERY RECOMMENDED

This module intentionally keeps intervention decisions
rule-based and transparent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class InterventionResult:
    """
    Result returned by the intervention engine.
    """

    state: str
    model_health: float
    reliability_degradation: float
    forecasted_health: Optional[float]
    forecast_direction: str
    reason: str

    def to_dict(self) -> dict:
        """Return the result as a dictionary."""

        return {
            "state": self.state,
            "model_health": self.model_health,
            "reliability_degradation":
                self.reliability_degradation,
            "forecasted_health":
                self.forecasted_health,
            "forecast_direction":
                self.forecast_direction,
            "reason": self.reason,
        }


class InterventionEngine:
    """
    Rule-based intervention decision engine.

    States:

        STABLE
            Model reliability is currently healthy.

        WATCH
            Reliability shows meaningful degradation or
            a concerning forecast trend.

        RECOVERY RECOMMENDED
            Reliability degradation is sufficiently severe
            to recommend the recovery stage.

    The thresholds are explicit and configurable.
    """

    STATES = (
        "STABLE",
        "WATCH",
        "RECOVERY RECOMMENDED",
    )

    def __init__(
        self,
        watch_health_threshold: float = 0.80,
        recovery_health_threshold: float = 0.60,
        watch_degradation_threshold: float = 0.20,
        recovery_degradation_threshold: float = 0.40,
    ) -> None:

        self.watch_health_threshold = float(
            watch_health_threshold
        )

        self.recovery_health_threshold = float(
            recovery_health_threshold
        )

        self.watch_degradation_threshold = float(
            watch_degradation_threshold
        )

        self.recovery_degradation_threshold = float(
            recovery_degradation_threshold
        )

        self._validate_thresholds()

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_thresholds(self) -> None:
        """
        Validate intervention thresholds.
        """

        thresholds = {
            "watch_health_threshold":
                self.watch_health_threshold,
            "recovery_health_threshold":
                self.recovery_health_threshold,
            "watch_degradation_threshold":
                self.watch_degradation_threshold,
            "recovery_degradation_threshold":
                self.recovery_degradation_threshold,
        }

        for name, value in thresholds.items():

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

        if (
            self.recovery_health_threshold
            >= self.watch_health_threshold
        ):
            raise ValueError(
                "recovery_health_threshold must be "
                "lower than watch_health_threshold."
            )

        if (
            self.recovery_degradation_threshold
            <= self.watch_degradation_threshold
        ):
            raise ValueError(
                "recovery_degradation_threshold must be "
                "greater than watch_degradation_threshold."
            )

    @staticmethod
    def _validate_score(
        value: float,
        name: str,
    ) -> float:
        """
        Validate a score expected to be in [0, 1].
        """

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

        return value

    # ------------------------------------------------------------------
    # BASIC STATE DECISION
    # ------------------------------------------------------------------

    def determine_state(
        self,
        model_health: float,
        reliability_degradation: float,
    ) -> str:
        """
        Determine the current intervention state.

        Recovery is recommended if either:
            - model health is at or below the recovery threshold
            - degradation is at or above the recovery threshold

        Watch is triggered if either:
            - model health is at or below the watch threshold
            - degradation is at or above the watch threshold

        Otherwise:
            STABLE
        """

        health = self._validate_score(
            model_health,
            "model_health",
        )

        degradation = self._validate_score(
            reliability_degradation,
            "reliability_degradation",
        )

        if (
            health
            <= self.recovery_health_threshold
            or degradation
            >= self.recovery_degradation_threshold
        ):
            return "RECOVERY RECOMMENDED"

        if (
            health
            <= self.watch_health_threshold
            or degradation
            >= self.watch_degradation_threshold
        ):
            return "WATCH"

        return "STABLE"

    # ------------------------------------------------------------------
    # FORECAST STATE
    # ------------------------------------------------------------------

    def determine_forecast_state(
        self,
        forecasted_health: Sequence[float],
    ) -> str:
        """
        Determine the state suggested by future predicted health.

        The lowest forecasted health value is used because
        intervention should account for the worst projected
        reliability within the forecast horizon.

        Returns:

            STABLE
            WATCH
            RECOVERY RECOMMENDED
        """

        values = np.asarray(
            forecasted_health,
            dtype=float,
        ).ravel()

        if len(values) == 0:
            raise ValueError(
                "forecasted_health cannot be empty."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "forecasted_health must contain "
                "only finite values."
            )

        values = np.clip(
            values,
            0.0,
            1.0,
        )

        lowest_health = float(
            np.min(values)
        )

        if (
            lowest_health
            <= self.recovery_health_threshold
        ):
            return "RECOVERY RECOMMENDED"

        if (
            lowest_health
            <= self.watch_health_threshold
        ):
            return "WATCH"

        return "STABLE"

    # ------------------------------------------------------------------
    # COMBINED DECISION
    # ------------------------------------------------------------------

    def decide(
        self,
        model_health: float,
        reliability_degradation: float,
        forecasted_health: Optional[
            Sequence[float]
        ] = None,
        forecast_direction: Optional[str] = None,
    ) -> InterventionResult:
        """
        Combine current reliability and forecast information
        into one intervention recommendation.

        Current reliability is the primary decision signal.

        Forecasting can escalate a current STABLE state to
        WATCH or RECOVERY RECOMMENDED if future reliability
        is projected to deteriorate sufficiently.
        """

        health = self._validate_score(
            model_health,
            "model_health",
        )

        degradation = self._validate_score(
            reliability_degradation,
            "reliability_degradation",
        )

        current_state = self.determine_state(
            model_health=health,
            reliability_degradation=degradation,
        )

        forecasted_value = None
        forecast_state = "STABLE"

        if forecasted_health is not None:

            forecast_values = np.asarray(
                forecasted_health,
                dtype=float,
            ).ravel()

            if len(forecast_values) == 0:
                raise ValueError(
                    "forecasted_health cannot be empty."
                )

            forecast_values = np.clip(
                forecast_values,
                0.0,
                1.0,
            )

            forecasted_value = float(
                np.min(forecast_values)
            )

            forecast_state = (
                self.determine_forecast_state(
                    forecast_values
                )
            )

        # Current performance has priority.
        state = self._highest_risk_state(
            current_state,
            forecast_state,
        )

        if forecast_direction is None:

            if (
                forecasted_health is not None
                and len(forecasted_health) >= 2
            ):

                first = float(
                    forecasted_health[0]
                )

                last = float(
                    forecasted_health[-1]
                )

                if last < first:
                    forecast_direction = (
                        "declining"
                    )

                elif last > first:
                    forecast_direction = (
                        "improving"
                    )

                else:
                    forecast_direction = (
                        "stable"
                    )

            else:
                forecast_direction = "unknown"

        reason = self._build_reason(
            state=state,
            model_health=health,
            degradation=degradation,
            forecasted_health=forecasted_value,
            forecast_direction=forecast_direction,
        )

        return InterventionResult(
            state=state,
            model_health=health,
            reliability_degradation=degradation,
            forecasted_health=forecasted_value,
            forecast_direction=forecast_direction,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # STATE PRIORITY
    # ------------------------------------------------------------------

    @classmethod
    def _highest_risk_state(
        cls,
        state_a: str,
        state_b: str,
    ) -> str:
        """
        Return the more severe state.

        This is not a model ranking; it is simply the
        predefined severity order of the intervention states.
        """

        priority = {
            "STABLE": 0,
            "WATCH": 1,
            "RECOVERY RECOMMENDED": 2,
        }

        if state_a not in priority:
            raise ValueError(
                f"Unknown intervention state: {state_a}"
            )

        if state_b not in priority:
            raise ValueError(
                f"Unknown intervention state: {state_b}"
            )

        if (
            priority[state_a]
            >= priority[state_b]
        ):
            return state_a

        return state_b

    # ------------------------------------------------------------------
    # REASON
    # ------------------------------------------------------------------

    def _build_reason(
        self,
        state: str,
        model_health: float,
        degradation: float,
        forecasted_health: Optional[float],
        forecast_direction: str,
    ) -> str:
        """
        Build a human-readable explanation of the rule decision.
        """

        if state == "STABLE":

            return (
                "Current model reliability is within "
                "the stable range and the available "
                "forecast does not cross the watch "
                "threshold."
            )

        if state == "WATCH":

            reasons = []

            if (
                model_health
                <= self.watch_health_threshold
            ):
                reasons.append(
                    "current model health is below "
                    "the watch threshold"
                )

            if (
                degradation
                >= self.watch_degradation_threshold
            ):
                reasons.append(
                    "current reliability degradation "
                    "is above the watch threshold"
                )

            if (
                forecasted_health is not None
                and forecasted_health
                <= self.watch_health_threshold
            ):
                reasons.append(
                    "forecasted model health enters "
                    "the watch range"
                )

            if (
                forecast_direction
                == "declining"
            ):
                reasons.append(
                    "the forecast indicates a declining "
                    "reliability trend"
                )

            return (
                "WATCH: "
                + "; ".join(reasons)
                + "."
            )

        reasons = []

        if (
            model_health
            <= self.recovery_health_threshold
        ):
            reasons.append(
                "current model health is at or below "
                "the recovery threshold"
            )

        if (
            degradation
            >= self.recovery_degradation_threshold
        ):
            reasons.append(
                "current reliability degradation "
                "has reached the recovery threshold"
            )

        if (
            forecasted_health is not None
            and forecasted_health
            <= self.recovery_health_threshold
        ):
            reasons.append(
                "forecasted model health reaches "
                "the recovery range"
            )

        return (
            "RECOVERY RECOMMENDED: "
            + "; ".join(reasons)
            + "."
        )

    # ------------------------------------------------------------------
    # TEMPORAL DECISIONS
    # ------------------------------------------------------------------

    def evaluate_history(
        self,
        monitoring_history: pd.DataFrame,
        forecasted_health: Optional[
            Sequence[float]
        ] = None,
    ) -> pd.DataFrame:
        """
        Convert monitoring history into intervention states.

        Required columns:

            model_health
            reliability_degradation

        Optional:

            window_id
            timestamp

        The rows remain in chronological order.
        """

        required_columns = {
            "model_health",
            "reliability_degradation",
        }

        missing = (
            required_columns
            - set(monitoring_history.columns)
        )

        if missing:
            raise ValueError(
                "Monitoring history is missing "
                "required columns: "
                + ", ".join(sorted(missing))
            )

        results = []

        forecast_values = None

        if forecasted_health is not None:

            forecast_values = np.asarray(
                forecasted_health,
                dtype=float,
            ).ravel()

        for index, row in monitoring_history.iterrows():

            result = self.decide(
                model_health=float(
                    row["model_health"]
                ),
                reliability_degradation=float(
                    row[
                        "reliability_degradation"
                    ]
                ),
                forecasted_health=None,
            )

            output = {
                "window_id": row.get(
                    "window_id",
                    index,
                ),
                "timestamp": row.get(
                    "timestamp",
                    None,
                ),
                "model_health":
                    result.model_health,
                "reliability_degradation":
                    result.reliability_degradation,
                "state":
                    result.state,
                "reason":
                    result.reason,
            }

            results.append(output)

        result_df = pd.DataFrame(
            results
        )

        # Forecast applies to the overall future horizon,
        # rather than being assigned to individual historical
        # windows.
        if forecast_values is not None:

            forecast_state = (
                self.determine_forecast_state(
                    forecast_values
                )
            )

            result_df.attrs[
                "forecast_state"
            ] = forecast_state

            result_df.attrs[
                "forecasted_health"
            ] = float(
                np.min(
                    forecast_values
                )
            )

        return result_df

    # ------------------------------------------------------------------
    # STATE SUMMARY
    # ------------------------------------------------------------------

    def state_summary(
        self,
        state_history: Sequence[str],
    ) -> dict:
        """
        Summarize intervention states observed across windows.
        """

        if state_history is None:
            raise ValueError(
                "state_history cannot be None."
            )

        states = list(
            state_history
        )

        if not states:
            return {
                "latest_state": None,
                "stable_windows": 0,
                "watch_windows": 0,
                "recovery_recommended_windows": 0,
            }

        invalid = [
            state
            for state in states
            if state not in self.STATES
        ]

        if invalid:
            raise ValueError(
                "Invalid intervention states: "
                + ", ".join(
                    sorted(set(invalid))
                )
            )

        return {
            "latest_state": states[-1],
            "stable_windows": states.count(
                "STABLE"
            ),
            "watch_windows": states.count(
                "WATCH"
            ),
            "recovery_recommended_windows":
                states.count(
                    "RECOVERY RECOMMENDED"
                ),
        }


# ----------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ----------------------------------------------------------------------

def recommend_intervention(
    model_health: float,
    reliability_degradation: float,
    forecasted_health: Optional[
        Sequence[float]
    ] = None,
) -> dict:
    """
    Convenience function for obtaining an intervention
    recommendation.
    """

    engine = InterventionEngine()

    result = engine.decide(
        model_health=model_health,
        reliability_degradation=(
            reliability_degradation
        ),
        forecasted_health=forecasted_health,
    )

    return result.to_dict()