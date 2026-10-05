from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


from src.datadecay import (
    load_electricity_dataset,
    numeric_features,
    temporal_split,
)

from src.datadecay.models import (
    BaselineModel,
)

from src.datadecay.evaluation import (
    evaluate_predictions,
    evaluate_regression,
    results_to_dataframe,
    temporal_evaluation,
)

from src.datadecay.reliability import (
    calculate_degradation,
    calculate_degradation_ratio,
    calculate_reliability_index,
    classify_reliability,
    evaluate_reliability,
    evaluate_multiple_models,
)

from src.datadecay.forecasting import (
    PersistenceForecaster,
    LinearTrendForecaster,
    forecast_series,
    calculate_forecast_metrics,
)

from src.datadecay.monitoring_models import (
    monitor_model,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)


def select_features(
    reference,
    count=5,
):
    features = numeric_features(
        reference
    )

    variance = (
        reference[features]
        .std()
        .dropna()
    )

    variance = variance[
        variance > 0
    ]

    return (
        variance
        .sort_values(
            ascending=False
        )
        .head(count)
        .index
        .tolist()
    )


def build_target(
    df,
    features,
):
    """
    Build a next-step mean-load target.
    """

    X = (
        df[features]
        .copy()
    )

    y = (
        df[features]
        .mean(axis=1)
        .shift(-1)
    )

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    return (
        X.loc[valid]
        .reset_index(drop=True),
        y.loc[valid]
        .reset_index(drop=True),
    )


def _check_baseline_model(
    X,
    y,
):
    print(
        "Testing baseline model..."
    )

    model = BaselineModel(
        task_type="regression",
        random_state=42,
    )

    model.fit(
        X,
        y,
    )

    predictions = model.predict(
        X
    )

    assert (
        len(predictions)
        == len(y)
    )

    assert np.isfinite(
        predictions
    ).all()

    print(
        "Baseline model: PASS"
    )

    return model, predictions

def test_baseline_model(
    X,
    y,
):
    print(
        "Testing baseline model..."
    )

    _check_baseline_model(
        X,
        y,
    )

    print(
        "Baseline model: PASS"
    )

def _check_evaluation(
    y,
    predictions,
):
    print(
        "Testing evaluation..."
    )

    result = evaluate_regression(
        y_true=y,
        y_pred=predictions,
        model_name="baseline",
    )

    result_dict = result.to_dict()

    required = {
        "model_name",
        "mae",
        "rmse",
        "mape",
        "r2",
        "task_type",
        "sample_count",
    }

    assert required.issubset(
        result_dict.keys()
    )

    assert np.isfinite(
        result_dict["mae"]
    )

    assert np.isfinite(
        result_dict["rmse"]
    )

    assert np.isfinite(
        result_dict["r2"]
    )

    print(
        "Evaluation: PASS"
    )

    return result

def test_evaluation(
    y,
    predictions,
):
    print(
        "Testing evaluation..."
    )

    _check_evaluation(
        y,
        predictions,
    )

    print(
        "Evaluation: PASS"
    )

def test_reliability():
    print(
        "Testing reliability..."
    )

    baseline = 1.0

    current = 0.8

    degradation = (
        calculate_degradation(
            baseline,
            current,
        )
    )

    ratio = (
        calculate_degradation_ratio(
            baseline,
            current,
        )
    )

    index = (
        calculate_reliability_index(
            baseline,
            current,
        )
    )

    status = (
        classify_reliability(
            ratio
        )
    )

    assert np.isclose(
        degradation,
        0.2,
    )

    assert np.isclose(
        ratio,
        0.2,
    )

    assert np.isclose(
        index,
        0.8,
    )

    assert (
        status == "HIGH"
    )

    result = evaluate_reliability(
        model_name="baseline",
        baseline_score=baseline,
        current_score=current,
    )

    assert (
        result.model_name
        == "baseline"
    )

    results = evaluate_multiple_models(
        baseline_scores={
            "model_a": 1.0,
            "model_b": 0.9,
        },
        current_scores={
            "model_a": 0.8,
            "model_b": 0.7,
        },
    )

    assert (
        len(results)
        == 2
    )

    print(
        "Reliability: PASS"
    )


def test_forecasting(
    y,
):
    print(
        "Testing forecasting..."
    )

    values = y.iloc[
        :1000
    ].to_numpy(
        dtype=float
    )

    # ---------------------------------------------------------
    # PERSISTENCE
    # ---------------------------------------------------------

    persistence = (
        PersistenceForecaster()
    )

    persistence.fit(
        values
    )

    persistence_predictions = (
        persistence.predict(
            10
        )
    )

    assert (
        len(
            persistence_predictions
        )
        == 10
    )

    assert np.allclose(
        persistence_predictions,
        values[-1],
    )

    # ---------------------------------------------------------
    # LINEAR TREND
    # ---------------------------------------------------------

    trend = (
        LinearTrendForecaster(
            lookback=100
        )
    )

    trend.fit(
        values
    )

    trend_predictions = (
        trend.predict(
            10
        )
    )

    assert (
        len(
            trend_predictions
        )
        == 10
    )

    assert np.isfinite(
        trend_predictions
    ).all()

    # ---------------------------------------------------------
    # FUNCTION API
    # ---------------------------------------------------------

    persistence_result = (
        forecast_series(
            values=values,
            horizon=10,
            method="persistence",
        )
    )

    trend_result = (
        forecast_series(
            values=values,
            horizon=10,
            method="linear_trend",
            lookback=100,
        )
    )

    assert (
        persistence_result.horizon
        == 10
    )

    assert (
        trend_result.horizon
        == 10
    )

    print(
        "Forecasting: PASS"
    )


def test_forecast_metrics():
    print(
        "Testing forecast metrics..."
    )

    actual = np.array(
        [
            10.0,
            20.0,
            30.0,
            40.0,
        ]
    )

    predicted = np.array(
        [
            11.0,
            18.0,
            33.0,
            36.0,
        ]
    )

    metrics = calculate_forecast_metrics(
        actual=actual,
        predicted=predicted,
    )

    assert (
        "mae"
        in metrics
    )

    assert (
        "rmse"
        in metrics
    )

    assert (
        "mape"
        in metrics
    )

    assert (
        metrics["mae"]
        > 0
    )

    assert (
        metrics["rmse"]
        > 0
    )

    print(
        "Forecast metrics: PASS"
    )


def test_temporal_evaluation(
    X,
    y,
):
    print(
        "Testing temporal evaluation..."
    )

    split_index = int(
        len(X) * 0.70
    )

    reference_X = (
        X.iloc[
            :split_index
        ]
        .copy()
    )

    reference_y = (
        y.iloc[
            :split_index
        ]
        .copy()
    )

    evaluation_X = (
        X.iloc[
            split_index:
        ]
        .copy()
    )

    evaluation_y = (
        y.iloc[
            split_index:
        ]
        .copy()
    )

    model = BaselineModel(
        task_type="regression",
        random_state=42,
    )

    result = temporal_evaluation(
        model=model,
        reference_X=reference_X,
        reference_y=reference_y,
        evaluation_X=evaluation_X,
        evaluation_y=evaluation_y,
        task_type="regression",
        model_name="temporal_baseline",
    )

    assert (
        result.model_name
        == "temporal_baseline"
    )

    assert (
        result.metrics["mae"]
        >= 0
    )

    assert (
        result.metrics["rmse"]
        >= 0
    )

    print(
        "Temporal evaluation: PASS"
    )


def test_monitoring_bridge(
    X,
    y,
):
    print(
        "Testing model monitoring bridge..."
    )

    split_index = int(
        len(X) * 0.70
    )

    reference_X = (
        X.iloc[
            :split_index
        ]
        .copy()
    )

    reference_y = (
        y.iloc[
            :split_index
        ]
        .copy()
    )

    current_X = (
        X.iloc[
            split_index:
        ]
        .copy()
    )

    current_y = (
        y.iloc[
            split_index:
        ]
        .copy()
    )

    model = BaselineModel(
        task_type="regression",
        random_state=42,
    )

    result = monitor_model(
        model=model,
        reference_X=reference_X,
        reference_y=reference_y,
        current_X=current_X,
        current_y=current_y,
        task_type="regression",
        model_name="baseline",
        period="evaluation",
        drift_severity="UNKNOWN",
        dsi=np.nan,
    )

    result_dict = result.to_dict()

    assert (
        result_dict["model_name"]
        == "baseline"
    )

    assert (
        "mae"
        in result_dict
    )

    assert (
        "rmse"
        in result_dict
    )

    assert (
        "r2"
        in result_dict
    )

    assert (
        "reliability_baseline_score"
        in result_dict
    )

    assert (
        "reliability_drifted_score"
        in result_dict
    )

    print(
        "Monitoring bridge: PASS"
    )


def main():

    print("=" * 80)

    print(
        "DataDecay-X | Person 2 Model Test"
    )

    print("=" * 80)

    print()

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    df = load_electricity_dataset(
        DATASET_PATH
    )

    (
        reference,
        monitoring,
        evaluation,
    ) = temporal_split(
        df,
        train_fraction=0.60,
        monitor_fraction=0.20,
    )

    features = select_features(
        reference,
        count=5,
    )

    print(
        "Selected features:",
        features,
    )

    # ---------------------------------------------------------
    # BUILD SUPERVISED DATA
    # ---------------------------------------------------------

    X_reference, y_reference = (
        build_target(
            reference,
            features,
        )
    )

    X_monitoring, y_monitoring = (
        build_target(
            monitoring,
            features,
        )
    )

    X_evaluation, y_evaluation = (
        build_target(
            evaluation,
            features,
        )
    )

    print(
        "Reference supervised shape:",
        X_reference.shape,
    )

    print(
        "Monitoring supervised shape:",
        X_monitoring.shape,
    )

    print(
        "Evaluation supervised shape:",
        X_evaluation.shape,
    )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    model, predictions = (
        _check_baseline_model(
            X_reference,
            y_reference,
        )
    )

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------

    result = _check_evaluation(
        y_reference,
        predictions,
    )

    dataframe = results_to_dataframe(
        [result]
    )

    assert (
        len(dataframe)
        == 1
    )

    # ---------------------------------------------------------
    # RELIABILITY
    # ---------------------------------------------------------

    test_reliability()

    # ---------------------------------------------------------
    # FORECASTING
    # ---------------------------------------------------------

    test_forecasting(
        y_reference
    )

    test_forecast_metrics()

    # ---------------------------------------------------------
    # TEMPORAL EVALUATION
    # ---------------------------------------------------------

    combined_X = pd.concat(
        [
            X_reference,
            X_monitoring,
            X_evaluation,
        ],
        ignore_index=True,
    )

    combined_y = pd.concat(
        [
            y_reference,
            y_monitoring,
            y_evaluation,
        ],
        ignore_index=True,
    )

    test_temporal_evaluation(
        combined_X,
        combined_y,
    )

    # ---------------------------------------------------------
    # MONITORING
    # ---------------------------------------------------------

    test_monitoring_bridge(
        combined_X,
        combined_y,
    )

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print()

    print("=" * 80)

    print(
        "ALL PERSON 2 MODEL TESTS PASSED"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()