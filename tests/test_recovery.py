"""
Tests for model recovery helpers in DataDecay-X.
"""

import numpy as np

from src.models.baseline import MeanRegressor
from src.models.evaluation import regression_metrics
from src.models.recovery import (
    RecoveryResult,
    calculate_recovery_improvement,
    evaluate_recovery,
    recover_model,
    recovery_success,
    summarize_recovery,
)


def make_regression_data(
    rows=100,
):
    """Create deterministic regression data."""

    rng = np.random.default_rng(42)

    X = rng.normal(
        0,
        1,
        size=(rows, 3),
    )

    y = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.5 * X[:, 2]
        + rng.normal(
            0,
            0.1,
            rows,
        )
    )

    return X, y


def test_mean_regressor_recovery():
    """A recovered MeanRegressor should be fitted correctly."""

    X, y = make_regression_data()

    original_model, recovered_model = recover_model(
        model_factory=MeanRegressor,
        X_original=X[:60],
        y_original=y[:60],
        X_recovery=X,
        y_recovery=y,
    )

    assert original_model.is_fitted
    assert recovered_model.is_fitted

    original_predictions = original_model.predict(
        X[60:]
    )

    recovered_predictions = recovered_model.predict(
        X[60:]
    )

    assert len(
        original_predictions
    ) == len(X[60:])

    assert len(
        recovered_predictions
    ) == len(X[60:])


def test_recover_model_returns_two_models():
    """recover_model should return original and recovered models."""

    X, y = make_regression_data()

    original_model, recovered_model = recover_model(
        model_factory=MeanRegressor,
        X_original=X[:50],
        y_original=y[:50],
        X_recovery=X[50:],
        y_recovery=y[50:],
    )

    assert isinstance(
        original_model,
        MeanRegressor,
    )

    assert isinstance(
        recovered_model,
        MeanRegressor,
    )


def test_calculate_recovery_improvement_higher_is_better():
    """Higher-is-better metrics should increase positively."""

    original = {
        "accuracy": 0.70,
        "f1_weighted": 0.65,
        "r2": 0.40,
    }

    recovered = {
        "accuracy": 0.80,
        "f1_weighted": 0.75,
        "r2": 0.60,
    }

    improvement = calculate_recovery_improvement(
        original,
        recovered,
    )

    assert np.isclose(
        improvement["accuracy"],
        0.10,
    )

    assert np.isclose(
        improvement["f1_weighted"],
        0.10,
    )

    assert np.isclose(
        improvement["r2"],
        0.20,
    )


def test_calculate_recovery_improvement_lower_is_better():
    """Lower-is-better metrics should improve when they decrease."""

    original = {
        "mae": 10.0,
        "mse": 100.0,
        "rmse": 10.0,
        "mape": 0.20,
    }

    recovered = {
        "mae": 8.0,
        "mse": 64.0,
        "rmse": 8.0,
        "mape": 0.15,
    }

    improvement = calculate_recovery_improvement(
        original,
        recovered,
    )

    assert np.isclose(
        improvement["mae"],
        2.0,
    )

    assert np.isclose(
        improvement["mse"],
        36.0,
    )

    assert np.isclose(
        improvement["rmse"],
        2.0,
    )

    assert np.isclose(
        improvement["mape"],
        0.05,
    )


def test_evaluate_recovery():
    """Recovery evaluation should return a RecoveryResult."""

    X, y = make_regression_data()

    original_model, recovered_model = recover_model(
        model_factory=MeanRegressor,
        X_original=X[:60],
        y_original=y[:60],
        X_recovery=X,
        y_recovery=y,
    )

    result = evaluate_recovery(
        original_model=original_model,
        recovered_model=recovered_model,
        X_evaluation=X[60:],
        y_evaluation=y[60:],
        evaluation_function=regression_metrics,
        model_name="MeanBaseline",
    )

    assert isinstance(
        result,
        RecoveryResult,
    )

    assert result.model_name == (
        "MeanBaseline"
    )

    assert "mae" in result.original_metrics
    assert "rmse" in result.original_metrics
    assert "r2" in result.original_metrics

    assert "mae" in result.recovered_metrics
    assert "rmse" in result.recovered_metrics
    assert "r2" in result.recovered_metrics

    assert "mae" in result.improvement
    assert "rmse" in result.improvement
    assert "r2" in result.improvement


def test_recovery_result_to_dict():
    """RecoveryResult should serialize its result fields."""

    result = RecoveryResult(
        model_name="test_model",
        original_metrics={
            "mae": 10.0,
        },
        recovered_metrics={
            "mae": 8.0,
        },
        improvement={
            "mae": 2.0,
        },
        recovered_model=None,
    )

    data = result.to_dict()

    assert data["model_name"] == "test_model"
    assert data["original_metrics"]["mae"] == 10.0
    assert data["recovered_metrics"]["mae"] == 8.0
    assert data["improvement"]["mae"] == 2.0

    # The fitted model itself should not be included in the
    # serialized dictionary.
    assert "recovered_model" not in data


def test_recovery_success():
    """Positive improvements should produce a successful recovery."""

    result = RecoveryResult(
        model_name="test_model",
        original_metrics={
            "mae": 10.0,
            "rmse": 12.0,
        },
        recovered_metrics={
            "mae": 8.0,
            "rmse": 10.0,
        },
        improvement={
            "mae": 2.0,
            "rmse": 2.0,
        },
        recovered_model=None,
    )

    assert recovery_success(
        result,
        required_metrics=[
            "mae",
            "rmse",
        ],
    )


def test_recovery_failure_when_metric_does_not_improve():
    """Recovery should fail when a required metric gets worse."""

    result = RecoveryResult(
        model_name="test_model",
        original_metrics={
            "mae": 10.0,
            "rmse": 12.0,
        },
        recovered_metrics={
            "mae": 11.0,
            "rmse": 10.0,
        },
        improvement={
            "mae": -1.0,
            "rmse": 2.0,
        },
        recovered_model=None,
    )

    assert not recovery_success(
        result,
        required_metrics=[
            "mae",
            "rmse",
        ],
    )


def test_recovery_success_with_all_metrics():
    """When no metrics are specified, all available metrics are used."""

    result = RecoveryResult(
        model_name="test_model",
        original_metrics={
            "mae": 10.0,
            "r2": 0.20,
        },
        recovered_metrics={
            "mae": 8.0,
            "r2": 0.50,
        },
        improvement={
            "mae": 2.0,
            "r2": 0.30,
        },
        recovered_model=None,
    )

    assert recovery_success(
        result
    )


def test_recovery_success_rejects_empty_requirements():
    """An empty required metric list cannot establish success."""

    result = RecoveryResult(
        model_name="test_model",
        original_metrics={
            "mae": 10.0,
        },
        recovered_metrics={
            "mae": 8.0,
        },
        improvement={
            "mae": 2.0,
        },
        recovered_model=None,
    )

    assert not recovery_success(
        result,
        required_metrics=[],
    )


def test_summarize_recovery():
    """Multiple recovery results should be summarized."""

    result_a = RecoveryResult(
        model_name="model_a",
        original_metrics={
            "mae": 10.0,
        },
        recovered_metrics={
            "mae": 8.0,
        },
        improvement={
            "mae": 2.0,
        },
        recovered_model=None,
    )

    result_b = RecoveryResult(
        model_name="model_b",
        original_metrics={
            "mae": 12.0,
        },
        recovered_metrics={
            "mae": 9.0,
        },
        improvement={
            "mae": 3.0,
        },
        recovered_model=None,
    )

    summary = summarize_recovery(
        [
            result_a,
            result_b,
        ]
    )

    assert len(summary) == 2

    assert summary[0]["model_name"] == (
        "model_a"
    )

    assert summary[1]["model_name"] == (
        "model_b"
    )


def test_recovery_with_custom_model_factory():
    """Recovery should work with a caller-provided model factory."""

    X, y = make_regression_data()

    def create_model():
        return MeanRegressor()

    original_model, recovered_model = recover_model(
        model_factory=create_model,
        X_original=X[:50],
        y_original=y[:50],
        X_recovery=X,
        y_recovery=y,
    )

    assert original_model.is_fitted
    assert recovered_model.is_fitted


def test_recovery_evaluation_is_deterministic():
    """Repeated evaluation of the same models should be identical."""

    X, y = make_regression_data()

    original_model, recovered_model = recover_model(
        model_factory=MeanRegressor,
        X_original=X[:60],
        y_original=y[:60],
        X_recovery=X,
        y_recovery=y,
    )

    result_a = evaluate_recovery(
        original_model,
        recovered_model,
        X[60:],
        y[60:],
        regression_metrics,
        model_name="MeanBaseline",
    )

    result_b = evaluate_recovery(
        original_model,
        recovered_model,
        X[60:],
        y[60:],
        regression_metrics,
        model_name="MeanBaseline",
    )

    assert result_a.original_metrics == (
        result_b.original_metrics
    )

    assert result_a.recovered_metrics == (
        result_b.recovered_metrics
    )

    assert result_a.improvement == (
        result_b.improvement
    )