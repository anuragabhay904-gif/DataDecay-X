"""
Controlled drift scenarios for DataDecay-X.

This module provides reusable definitions for the four project
drift scenarios:

- stable
- gradual
- abrupt
- recurring

The actual numerical drift application is delegated to the
main ``src.datadecay.simulator`` implementation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftScenario:
    """
    Configuration describing one drift scenario.
    """

    name: str
    description: str
    magnitude: float = 0.30
    start_fraction: float | None = None
    end_fraction: float | None = None
    cycle_length: int | None = None

    def validate(self):
        """Validate the scenario configuration."""

        if not self.name:
            raise ValueError(
                "Scenario name cannot be empty."
            )

        if self.magnitude < 0:
            raise ValueError(
                "Drift magnitude cannot be negative."
            )

        if self.start_fraction is not None:
            if not 0 <= self.start_fraction <= 1:
                raise ValueError(
                    "start_fraction must be between 0 and 1."
                )

        if self.end_fraction is not None:
            if not 0 <= self.end_fraction <= 1:
                raise ValueError(
                    "end_fraction must be between 0 and 1."
                )

        if (
            self.start_fraction is not None
            and self.end_fraction is not None
            and self.start_fraction >= self.end_fraction
        ):
            raise ValueError(
                "start_fraction must be smaller than "
                "end_fraction."
            )

        if self.cycle_length is not None:
            if self.cycle_length <= 0:
                raise ValueError(
                    "cycle_length must be greater than zero."
                )

        return True


STABLE = DriftScenario(
    name="stable",
    description=(
        "No controlled distribution shift is applied."
    ),
    magnitude=0.0,
)


GRADUAL = DriftScenario(
    name="gradual",
    description=(
        "Drift increases progressively over a defined "
        "portion of the monitoring period."
    ),
    magnitude=0.30,
    start_fraction=0.25,
    end_fraction=0.75,
)


ABRUPT = DriftScenario(
    name="abrupt",
    description=(
        "A step change is introduced at a defined point "
        "in the monitoring period."
    ),
    magnitude=0.30,
    start_fraction=0.50,
)


RECURRING = DriftScenario(
    name="recurring",
    description=(
        "The data alternates between baseline and shifted "
        "regimes over repeated cycles."
    ),
    magnitude=0.30,
    cycle_length=1000,
)


SCENARIOS = {
    "stable": STABLE,
    "gradual": GRADUAL,
    "abrupt": ABRUPT,
    "recurring": RECURRING,
}


def get_scenario(name):
    """
    Return a configured drift scenario by name.
    """

    if not isinstance(name, str):
        raise TypeError(
            "Scenario name must be a string."
        )

    key = name.strip().lower()

    if key not in SCENARIOS:
        raise ValueError(
            f"Unknown drift scenario: {name}. "
            f"Available scenarios: "
            f"{', '.join(SCENARIOS)}"
        )

    scenario = SCENARIOS[key]
    scenario.validate()

    return scenario


def list_scenarios():
    """Return the names of all supported scenarios."""

    return list(SCENARIOS.keys())


def scenario_parameters(name):
    """
    Return a scenario configuration as a dictionary.
    """

    scenario = get_scenario(name)

    return {
        "name": scenario.name,
        "description": scenario.description,
        "magnitude": scenario.magnitude,
        "start_fraction": scenario.start_fraction,
        "end_fraction": scenario.end_fraction,
        "cycle_length": scenario.cycle_length,
    }


def calculate_window_positions(
    scenario,
    monitoring_length,
):
    """
    Convert fractional scenario positions into row positions.

    Parameters
    ----------
    scenario : DriftScenario or str
        Scenario configuration or scenario name.

    monitoring_length : int
        Number of rows in the monitoring period.

    Returns
    -------
    dict
        Calculated start/end positions and cycle length.
    """

    if monitoring_length <= 0:
        raise ValueError(
            "monitoring_length must be greater than zero."
        )

    if isinstance(scenario, str):
        scenario = get_scenario(scenario)

    if not isinstance(
        scenario,
        DriftScenario,
    ):
        raise TypeError(
            "scenario must be a DriftScenario or string."
        )

    start_window = None
    end_window = None

    if scenario.start_fraction is not None:
        start_window = int(
            monitoring_length
            * scenario.start_fraction
        )

    if scenario.end_fraction is not None:
        end_window = int(
            monitoring_length
            * scenario.end_fraction
        )

    return {
        "start_window": start_window,
        "end_window": end_window,
        "cycle_length": scenario.cycle_length,
    }


def build_scenario_schedule(
    scenario,
    monitoring_length,
):
    """
    Build a simple ground-truth schedule describing when drift
    should be active.

    Returns
    -------
    list[dict]
        Window-level ground-truth schedule.
    """

    if monitoring_length <= 0:
        raise ValueError(
            "monitoring_length must be greater than zero."
        )

    if isinstance(scenario, str):
        scenario = get_scenario(scenario)

    positions = calculate_window_positions(
        scenario,
        monitoring_length,
    )

    schedule = []

    if scenario.name == "stable":

        for index in range(monitoring_length):
            schedule.append(
                {
                    "window": index,
                    "drift_active": False,
                    "regime": "stable",
                }
            )

        return schedule

    if scenario.name == "gradual":

        start = positions["start_window"]
        end = positions["end_window"]

        for index in range(monitoring_length):

            if index < start:
                active = False
                regime = "baseline"

            elif index <= end:
                active = True
                regime = "gradual"

            else:
                active = True
                regime = "drifted"

            schedule.append(
                {
                    "window": index,
                    "drift_active": active,
                    "regime": regime,
                }
            )

        return schedule

    if scenario.name == "abrupt":

        start = positions["start_window"]

        for index in range(monitoring_length):

            active = index >= start

            schedule.append(
                {
                    "window": index,
                    "drift_active": active,
                    "regime": (
                        "drifted"
                        if active
                        else "baseline"
                    ),
                }
            )

        return schedule

    if scenario.name == "recurring":

        cycle_length = (
            positions["cycle_length"]
        )

        for index in range(monitoring_length):

            cycle_position = (
                index % cycle_length
            )

            active = (
                cycle_position
                >= cycle_length // 2
            )

            schedule.append(
                {
                    "window": index,
                    "drift_active": active,
                    "regime": (
                        "drifted"
                        if active
                        else "baseline"
                    ),
                }
            )

        return schedule

    raise ValueError(
        f"Unsupported scenario: {scenario.name}"
    )


__all__ = [
    "DriftScenario",
    "STABLE",
    "GRADUAL",
    "ABRUPT",
    "RECURRING",
    "SCENARIOS",
    "get_scenario",
    "list_scenarios",
    "scenario_parameters",
    "calculate_window_positions",
    "build_scenario_schedule",
]