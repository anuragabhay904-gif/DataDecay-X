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
    DriftSimulator,
    calculate_feature_metrics,
    calculate_dsi,
    classify_dsi,
)

from src.datadecay.models import (
    BaselineModel,
)

from src.datadecay.evaluation import (
    evaluate_predictions,
    temporal_evaluation,
)

from src.datadecay.forecasting import (
    forecast_series,
    calculate_forecast_metrics,
)

from src.datadecay.reliability import (
    evaluate_reliability,
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

    selected = (
        variance
        .sort_values(
            ascending=False
        )
        .head(count)
        .index
        .tolist()
    )

    if len(selected) != count:
        raise AssertionError(
            "Could not select the required "
            "number of features."
        )

    return selected


def build_supervised_data(
    df,
    features,
):
    """
    Build a next-step regression target.

    Target:
        mean load of selected features
        at the next timestamp.
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

    X = (
        X.loc[valid]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid]
        .reset_index(drop=True)
    )

    return (
        X,
        y,
    )


def _check_temporal_structure(
    df,
):
    (
        reference,
        monitoring,
        evaluation,
    ) = temporal_split(
        df,
        train_fraction=0.60,
        monitor_fraction=0.20,
    )

    assert len(reference) > 0
    assert len(monitoring) > 0
    assert len(evaluation) > 0

    assert (
        reference["timestamp"].iloc[-1]
        < monitoring["timestamp"].iloc[0]
    )

    assert (
        monitoring["timestamp"].iloc[-1]
        < evaluation["timestamp"].iloc[0]
    )

    return (
        reference,
        monitoring,
        evaluation,
    )


def test_temporal_structure(
    df,
):
    print(
        "Checking temporal structure..."
    )

    (
        reference,
        monitoring,
        evaluation,
    ) = _check_temporal_structure(
        df
    )

    assert len(reference) > 0
    assert len(monitoring) > 0
    assert len(evaluation) > 0

    print(
        "Temporal structure: PASS"
    )


def _check_baseline_model(
    X,
    y,
):
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

    result = evaluate_predictions(
        y_true=y,
        y_pred=predictions,
        task_type="regression",
        model_name="baseline",
    )

    assert (
        result.metrics["mae"]
        >= 0
    )

    assert (
        result.metrics["rmse"]
        >= 0
    )

    assert np.isfinite(
        result.metrics["r2"]
    )

    return model


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


def test_forecasting(
    y,
):
    print(
        "Testing forecasting..."
    )

    values = y.iloc[
        :2000
    ].to_numpy(
        dtype=float
    )

    persistence = forecast_series(
        values=values,
        horizon=20,
        method="persistence",
    )

    trend = forecast_series(
        values=values,
        horizon=20,
        method="linear_trend",
        lookback=500,
    )

    assert (
        len(
            persistence.predictions
        )
        == 20
    )

    assert (
        len(
            trend.predictions
        )
        == 20
    )

    assert np.isfinite(
        persistence.predictions
    ).all()

    assert np.isfinite(
        trend.predictions
    ).all()

    actual = (
        y.iloc[
            2000:2020
        ]
        .to_numpy(
            dtype=float
        )
    )

    persistence_metrics = (
        calculate_forecast_metrics(
            actual=actual,
            predicted=(
                persistence.predictions
            ),
        )
    )

    trend_metrics = (
        calculate_forecast_metrics(
            actual=actual,
            predicted=(
                trend.predictions
            ),
        )
    )

    assert (
        persistence_metrics["mae"]
        >= 0
    )

    assert (
        persistence_metrics["rmse"]
        >= 0
    )

    assert (
        trend_metrics["mae"]
        >= 0
    )

    assert (
        trend_metrics["rmse"]
        >= 0
    )

    print(
        "Forecasting: PASS"
    )


def test_reliability():
    print(
        "Testing reliability..."
    )

    result = evaluate_reliability(
        model_name="baseline",
        baseline_score=0.90,
        current_score=0.72,
    )

    assert (
        result.model_name
        == "baseline"
    )

    assert np.isclose(
        result.degradation,
        0.18,
    )

    assert np.isclose(
        result.degradation_ratio,
        0.20,
    )

    assert (
        result.reliability_status
        == "HIGH"
    )

    print(
        "Reliability: PASS"
    )


def _check_drift_and_model_monitoring(
    reference,
    monitoring,
    features,
):
    simulator = DriftSimulator(
        seed=42
    )

    drifted, metadata = (
        simulator.abrupt(
            df=monitoring.copy(),
            features=features,
            magnitude=0.30,
            start_window=(
                len(monitoring) // 2
            ),
            baseline_df=reference,
        )
    )

    drift_metrics = (
        calculate_feature_metrics(
            reference_df=reference,
            current_df=drifted,
            features=features,
        )
    )

    dsi = calculate_dsi(
        drift_metrics
    )

    severity = classify_dsi(
        dsi
    )

    assert (
        dsi >= 0
    )

    assert isinstance(
        severity,
        str,
    )

    reference_X, reference_y = (
        build_supervised_data(
            reference,
            features,
        )
    )

    current_X, current_y = (
        build_supervised_data(
            drifted,
            features,
        )
    )

    model = BaselineModel(
        task_type="regression",
        random_state=42,
    )

    monitoring_result = monitor_model(
        model=model,
        reference_X=reference_X,
        reference_y=reference_y,
        current_X=current_X,
        current_y=current_y,
        task_type="regression",
        model_name="baseline",
        period="drifted",
        drift_severity=severity,
        dsi=dsi,
    )

    result = monitoring_result.to_dict()

    assert (
        result["model_name"]
        == "baseline"
    )

    assert (
        result["dsi"]
        >= 0
    )

    assert (
        result["drift_severity"]
        == severity
    )

    assert (
        "mae"
        in result
    )

    assert (
        "rmse"
        in result
    )

    assert (
        "r2"
        in result
    )

    assert (
        "reliability_baseline_score"
        in result
    )

    assert (
        "reliability_drifted_score"
        in result
    )

    return result


def test_drift_and_model_monitoring(
    reference,
    monitoring,
    features,
):
    print(
        "Testing drift-to-model monitoring..."
    )

    _check_drift_and_model_monitoring(
        reference=reference,
        monitoring=monitoring,
        features=features,
    )

    print(
        "Drift-to-model monitoring: PASS"
    )


def main():

    print("=" * 80)

    print(
        "DataDecay-X | Person 2 Full Pipeline Test"
    )

    print("=" * 80)

    print()

    df = load_electricity_dataset(
        DATASET_PATH
    )

    assert not df.empty

    print(
        "Dataset loading: PASS"
    )

    (
        reference,
        monitoring,
        evaluation,
    ) = _check_temporal_structure(
        df
    )

    features = select_features(
        reference,
        count=5,
    )

    print()

    print(
        "Selected features:"
    )

    for feature in features:
        print(
            " -",
            feature,
        )

    (
        reference_X,
        reference_y,
    ) = build_supervised_data(
        reference,
        features,
    )

    (
        monitoring_X,
        monitoring_y,
    ) = build_supervised_data(
        monitoring,
        features,
    )

    (
        evaluation_X,
        evaluation_y,
    ) = build_supervised_data(
        evaluation,
        features,
    )

    assert (
        len(reference_X)
        == len(reference_y)
    )

    assert (
        len(monitoring_X)
        == len(monitoring_y)
    )

    assert (
        len(evaluation_X)
        == len(evaluation_y)
    )

    print(
        "Supervised data preparation: PASS"
    )

    model = _check_baseline_model(
        reference_X,
        reference_y,
    )

    print(
        "Baseline model: PASS"
    )

    print(
        "Testing chronological model evaluation..."
    )

    temporal_result = (
        temporal_evaluation(
            model=BaselineModel(
                task_type="regression",
                random_state=42,
            ),
            reference_X=reference_X,
            reference_y=reference_y,
            evaluation_X=evaluation_X,
            evaluation_y=evaluation_y,
            task_type="regression",
            model_name="temporal_baseline",
        )
    )

    assert (
        temporal_result.metrics["mae"]
        >= 0
    )

    assert (
        temporal_result.metrics["rmse"]
        >= 0
    )

    print(
        "Chronological model evaluation: PASS"
    )

    test_forecasting(
        reference_y
    )

    test_reliability()

    monitoring_result = (
        _check_drift_and_model_monitoring(
            reference=reference,
            monitoring=monitoring,
            features=features,
        )
    )

    assert isinstance(
        monitoring_result,
        dict,
    )

    print()

    print("=" * 80)

    print(
        "PERSON 2 FULL PIPELINE TEST PASSED"
    )

    print("=" * 80)

    print()

    print(
        "Reference rows:",
        len(reference),
    )

    print(
        "Monitoring rows:",
        len(monitoring),
    )

    print(
        "Evaluation rows:",
        len(evaluation),
    )

    print(
        "Features:",
        len(features),
    )


if __name__ == "__main__":
    main()