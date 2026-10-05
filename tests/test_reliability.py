"""
Tests for DataDecay-X Person 2.

Tests:
    - Predictive models
    - Reliability monitoring
    - Reliability degradation
    - Model health
    - Persistence forecasting
    - Linear trend forecasting
    - Intervention decisions
"""

import numpy as np
import pandas as pd
import pytest

from src.datadecay.models import ModelManager
from src.datadecay.monitoring import ReliabilityMonitor
from src.datadecay.forecasting import ReliabilityForecaster
from src.datadecay.intervention import InterventionEngine


# ---------------------------------------------------------------------
# SAMPLE DATA
# ---------------------------------------------------------------------

@pytest.fixture
def classification_data():
    """
    Small deterministic binary classification dataset.
    """

    X = pd.DataFrame(
        {
            "feature_1": [
                0.1, 0.2, 0.3, 0.4,
                0.6, 0.7, 0.8, 0.9,
                0.15, 0.25, 0.75, 0.85,
            ],
            "feature_2": [
                0.2, 0.1, 0.4, 0.3,
                0.7, 0.8, 0.6, 0.9,
                0.1, 0.3, 0.8, 0.7,
            ],
        }
    )

    y = np.array(
        [
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 1, 1,
        ]
    )

    return X, y


# ---------------------------------------------------------------------
# MODEL TESTS
# ---------------------------------------------------------------------

def test_model_manager_initialization():
    """
    ModelManager should initialize the required core models.
    """

    manager = ModelManager()

    assert manager is not None

    expected_models = {
        "logistic_regression",
        "svm",
        "naive_bayes",
    }

    assert expected_models.issubset(
        set(manager.models.keys())
    )


def test_models_can_train(classification_data):
    """
    All required models should train successfully.
    """

    X, y = classification_data

    manager = ModelManager()

    manager.train_all(
        X,
        y,
    )

    for name in [
        "logistic_regression",
        "svm",
        "naive_bayes",
    ]:
        assert name in manager.trained_models


def test_models_can_predict(classification_data):
    """
    Trained models should produce predictions.
    """

    X, y = classification_data

    manager = ModelManager()

    manager.train_all(
        X,
        y,
    )

    predictions = manager.predict_all(X)

    assert isinstance(
        predictions,
        dict,
    )

    for name, prediction in predictions.items():

        assert len(prediction) == len(y)


# ---------------------------------------------------------------------
# MONITORING TESTS
# ---------------------------------------------------------------------

def test_reliability_metrics():
    """
    ReliabilityMonitor should calculate all required
    predictive metrics.
    """

    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 1, 1]
    )

    monitor = ReliabilityMonitor()

    metrics = monitor.calculate_metrics(
        y_true,
        y_pred,
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert "mean_confidence" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_baseline_is_set():
    """
    Baseline reliability should be stored correctly.
    """

    y_true = np.array(
        [0, 0, 1, 1]
    )

    y_pred = np.array(
        [0, 0, 1, 1]
    )

    monitor = ReliabilityMonitor()

    baseline = monitor.set_baseline(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert monitor.baseline_metrics is not None

    assert baseline["accuracy"] == pytest.approx(
        1.0
    )

    assert baseline["f1"] == pytest.approx(
        1.0
    )


def test_no_degradation_at_baseline():
    """
    A model performing exactly at baseline should have
    zero reliability degradation.
    """

    baseline = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }

    monitor = ReliabilityMonitor(
        baseline_metrics=baseline
    )

    degradation = (
        monitor.calculate_reliability_degradation(
            baseline
        )
    )

    assert degradation == pytest.approx(
        0.0
    )


def test_degradation_increases_when_performance_falls():
    """
    Lower current performance should produce positive
    reliability degradation.
    """

    baseline = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }

    current = {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }

    monitor = ReliabilityMonitor(
        baseline_metrics=baseline
    )

    degradation = (
        monitor.calculate_reliability_degradation(
            current
        )
    )

    assert degradation > 0.0
    assert degradation <= 1.0


def test_model_health_is_inverse_of_degradation():
    """
    Model health should decrease when degradation increases.
    """

    baseline = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }

    current = {
        "accuracy": 0.8,
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
    }

    monitor = ReliabilityMonitor(
        baseline_metrics=baseline
    )

    degradation = (
        monitor.calculate_reliability_degradation(
            current
        )
    )

    health = (
        monitor.calculate_model_health(
            current
        )
    )

    assert health == pytest.approx(
        1.0 - degradation
    )


def test_temporal_monitoring():
    """
    Multiple temporal windows should be monitored in
    the supplied chronological order.
    """

    monitor = ReliabilityMonitor()

    windows = [
        {
            "window_id": 1,
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 0, 1, 1],
        },
        {
            "window_id": 2,
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 1, 1, 1],
        },
        {
            "window_id": 3,
            "y_true": [0, 0, 1, 1],
            "y_pred": [1, 1, 1, 0],
        },
    ]

    history = monitor.monitor_windows(
        windows
    )

    assert len(history) == 3

    assert list(
        history["window_id"]
    ) == [1, 2, 3]

    assert (
        history.iloc[0]["model_health"]
        >=
        history.iloc[2]["model_health"]
    )


# ---------------------------------------------------------------------
# FORECASTING TESTS
# ---------------------------------------------------------------------

def test_persistence_forecast():
    """
    Persistence forecasting should repeat the latest
    observed reliability value.
    """

    history = [
        0.95,
        0.90,
        0.85,
    ]

    forecaster = ReliabilityForecaster()

    result = forecaster.persistence(
        history,
        horizon=4,
    )

    assert len(result.values) == 4

    assert np.allclose(
        result.values,
        [
            0.85,
            0.85,
            0.85,
            0.85,
        ],
    )


def test_linear_trend_forecast_declines():
    """
    A declining historical reliability sequence should
    produce a declining linear trend forecast.
    """

    history = [
        1.00,
        0.90,
        0.80,
        0.70,
    ]

    forecaster = ReliabilityForecaster()

    result = forecaster.linear_trend(
        history,
        horizon=3,
    )

    assert result.slope < 0

    assert (
        result.values[-1]
        <
        result.values[0]
    )


def test_forecast_values_are_bounded():
    """
    Reliability forecasts must remain in [0, 1].
    """

    history = [
        0.9,
        0.8,
        0.7,
        0.6,
    ]

    forecaster = ReliabilityForecaster()

    result = forecaster.linear_trend(
        history,
        horizon=10,
    )

    assert np.all(
        result.values >= 0.0
    )

    assert np.all(
        result.values <= 1.0
    )


def test_forecast_table():
    """
    Forecast table should contain both core forecasting
    methods.
    """

    history = [
        0.95,
        0.90,
        0.85,
    ]

    forecaster = ReliabilityForecaster()

    table = forecaster.forecast_table(
        history,
        horizon=5,
    )

    assert len(table) == 5

    assert "forecast_step" in table.columns
    assert "persistence" in table.columns
    assert "linear_trend" in table.columns


# ---------------------------------------------------------------------
# INTERVENTION TESTS
# ---------------------------------------------------------------------

def test_stable_intervention():
    """
    Healthy model should remain STABLE.
    """

    engine = InterventionEngine()

    result = engine.decide(
        model_health=0.95,
        reliability_degradation=0.05,
    )

    assert result.state == "STABLE"


def test_watch_intervention():
    """
    Moderate degradation should produce WATCH.
    """

    engine = InterventionEngine()

    result = engine.decide(
        model_health=0.75,
        reliability_degradation=0.25,
    )

    assert result.state == "WATCH"


def test_recovery_recommendation():
    """
    Severe degradation should produce
    RECOVERY RECOMMENDED.
    """

    engine = InterventionEngine()

    result = engine.decide(
        model_health=0.50,
        reliability_degradation=0.50,
    )

    assert (
        result.state
        == "RECOVERY RECOMMENDED"
    )


def test_forecast_can_trigger_watch():
    """
    A sufficiently poor future reliability forecast
    should escalate the intervention state.
    """

    engine = InterventionEngine()

    result = engine.decide(
        model_health=0.90,
        reliability_degradation=0.10,
        forecasted_health=[
            0.85,
            0.80,
            0.75,
        ],
    )

    assert result.state == "WATCH"


def test_forecast_can_trigger_recovery():
    """
    A severe forecast should be capable of triggering
    RECOVERY RECOMMENDED.
    """

    engine = InterventionEngine()

    result = engine.decide(
        model_health=0.90,
        reliability_degradation=0.10,
        forecasted_health=[
            0.80,
            0.65,
            0.55,
        ],
    )

    assert (
        result.state
        == "RECOVERY RECOMMENDED"
    )


def test_state_summary():
    """
    Intervention history should be summarized correctly.
    """

    engine = InterventionEngine()

    states = [
        "STABLE",
        "WATCH",
        "WATCH",
        "RECOVERY RECOMMENDED",
    ]

    summary = engine.state_summary(
        states
    )

    assert (
        summary["latest_state"]
        == "RECOVERY RECOMMENDED"
    )

    assert (
        summary["stable_windows"]
        == 1
    )

    assert (
        summary["watch_windows"]
        == 2
    )

    assert (
        summary[
            "recovery_recommended_windows"
        ]
        == 1
    )