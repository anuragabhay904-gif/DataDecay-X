"""
Tests for the legacy drift package.

These tests verify that the four controlled drift scenarios
preserve dataframe structure and produce reproducible results.
"""

import numpy as np
import pandas as pd

from src.drift.scenarios import (
    ABRUPT,
    GRADUAL,
    RECURRING,
    STABLE,
    get_scenario,
    list_scenarios,
)
from src.drift.simulator import (
    LegacyDriftSimulator,
)


def make_test_dataframe(
    rows=1000,
):
    """Create a deterministic test time series."""

    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-01",
                periods=rows,
                freq="15min",
            ),
            "feature_1": rng.normal(
                100,
                10,
                rows,
            ),
            "feature_2": rng.normal(
                50,
                5,
                rows,
            ),
            "feature_3": rng.normal(
                200,
                20,
                rows,
            ),
        }
    )


def test_scenario_registry():
    """All required drift scenarios must be available."""

    scenarios = list_scenarios()

    assert "stable" in scenarios
    assert "gradual" in scenarios
    assert "abrupt" in scenarios
    assert "recurring" in scenarios

    assert get_scenario("stable") == STABLE
    assert get_scenario("gradual") == GRADUAL
    assert get_scenario("abrupt") == ABRUPT
    assert get_scenario("recurring") == RECURRING


def test_scenario_validation():
    """Configured scenarios should contain valid parameters."""

    assert STABLE.validate()
    assert GRADUAL.validate()
    assert ABRUPT.validate()
    assert RECURRING.validate()


def test_stable_scenario():
    """Stable drift must leave feature values unchanged."""

    df = make_test_dataframe()

    simulator = LegacyDriftSimulator(
        seed=42
    )

    result = simulator.stable(
        df,
        features=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
    )

    assert result.scenario == "stable"
    assert result.data.shape == df.shape
    assert list(
        result.data.columns
    ) == list(
        df.columns
    )

    pd.testing.assert_frame_equal(
        result.data,
        df,
    )


def test_gradual_scenario():
    """Gradual drift should preserve schema and modify data."""

    df = make_test_dataframe()

    simulator = LegacyDriftSimulator(
        seed=42
    )

    result = simulator.gradual(
        df,
        features=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        magnitude=0.30,
        start_window=250,
        end_window=750,
        baseline_df=df,
    )

    assert result.scenario == "gradual"
    assert result.data.shape == df.shape

    assert list(
        result.data.columns
    ) == list(
        df.columns
    )

    assert not np.allclose(
        result.data[
            [
                "feature_1",
                "feature_2",
                "feature_3",
            ]
        ].to_numpy(),
        df[
            [
                "feature_1",
                "feature_2",
                "feature_3",
            ]
        ].to_numpy(),
    )


def test_abrupt_scenario():
    """Abrupt drift should introduce a change after the start."""

    df = make_test_dataframe()

    simulator = LegacyDriftSimulator(
        seed=42
    )

    result = simulator.abrupt(
        df,
        features=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        magnitude=0.30,
        start_window=500,
        baseline_df=df,
    )

    assert result.scenario == "abrupt"
    assert result.data.shape == df.shape

    before = result.data.iloc[
        :500
    ][
        [
            "feature_1",
            "feature_2",
            "feature_3",
        ]
    ]

    original_before = df.iloc[
        :500
    ][
        [
            "feature_1",
            "feature_2",
            "feature_3",
        ]
    ]

    pd.testing.assert_frame_equal(
        before,
        original_before,
    )

    after = result.data.iloc[
        500:
    ][
        [
            "feature_1",
            "feature_2",
            "feature_3",
        ]
    ]

    original_after = df.iloc[
        500:
    ][
        [
            "feature_1",
            "feature_2",
            "feature_3",
        ]
    ]

    assert not np.allclose(
        after.to_numpy(),
        original_after.to_numpy(),
    )


def test_recurring_scenario():
    """Recurring drift should preserve structure and modify regimes."""

    df = make_test_dataframe()

    simulator = LegacyDriftSimulator(
        seed=42
    )

    result = simulator.recurring(
        df,
        features=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
        magnitude=0.30,
        cycle_length=200,
        baseline_df=df,
    )

    assert result.scenario == "recurring"
    assert result.data.shape == df.shape

    assert list(
        result.data.columns
    ) == list(
        df.columns
    )

    assert not np.allclose(
        result.data[
            [
                "feature_1",
                "feature_2",
                "feature_3",
            ]
        ].to_numpy(),
        df[
            [
                "feature_1",
                "feature_2",
                "feature_3",
            ]
        ].to_numpy(),
    )


def test_reproducibility():
    """The same seed and input should produce the same result."""

    df = make_test_dataframe()

    simulator_a = LegacyDriftSimulator(
        seed=42
    )

    simulator_b = LegacyDriftSimulator(
        seed=42
    )

    result_a = simulator_a.abrupt(
        df,
        features=["feature_1", "feature_2"],
        magnitude=0.30,
        start_window=500,
        baseline_df=df,
    )

    result_b = simulator_b.abrupt(
        df,
        features=["feature_1", "feature_2"],
        magnitude=0.30,
        start_window=500,
        baseline_df=df,
    )

    pd.testing.assert_frame_equal(
        result_a.data,
        result_b.data,
    )


def test_simulate_dispatcher():
    """The generic simulator should dispatch all scenarios."""

    df = make_test_dataframe()

    simulator = LegacyDriftSimulator(
        seed=42
    )

    for scenario in [
        "stable",
        "gradual",
        "abrupt",
        "recurring",
    ]:

        result = simulator.simulate(
            scenario=scenario,
            df=df,
            features=[
                "feature_1",
                "feature_2",
            ],
            magnitude=0.30,
            start_window=500,
            end_window=750,
            cycle_length=200,
            baseline_df=df,
        )

        assert result.scenario == scenario
        assert result.data.shape == df.shape


def test_invalid_scenario():
    """Unknown scenario names should raise ValueError."""

    df = make_test_dataframe()

    simulator = LegacyDriftSimulator(
        seed=42
    )

    try:
        simulator.simulate(
            scenario="unknown",
            df=df,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for an unknown scenario."
        )


def test_invalid_feature():
    """Missing feature names should raise ValueError."""

    df = make_test_dataframe()

    simulator = LegacyDriftSimulator(
        seed=42
    )

    try:
        simulator.abrupt(
            df,
            features=["does_not_exist"],
            baseline_df=df,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for a missing feature."
        )