from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
    ensure_dir,
    save_json,
)

from src.datadecay.evaluation import (
    evaluate_predictions,
)


DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "person2"
)

SEED = 42

FEATURE_COUNT = 5

DRIFT_MAGNITUDE = 0.30


def select_features(
    reference,
    count=FEATURE_COUNT,
):
    """
    Select numeric features with non-zero variance.
    """

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

    if len(selected) < count:
        raise ValueError(
            "Not enough usable numeric features."
        )

    return selected


def build_classification_target(
    df,
    features,
):
    """
    Build a binary target representing whether
    the next timestamp's mean load is above the
    current timestamp's mean load.

    This creates a classification task suitable
    for Logistic Regression, SVM, and Naive Bayes.
    """

    current_mean = (
        df[features]
        .mean(axis=1)
    )

    next_mean = (
        current_mean
        .shift(-1)
    )

    target = (
        (next_mean > current_mean)
        .astype(int)
    )

    target.iloc[-1] = np.nan

    return target


def prepare_supervised_data(
    df,
    features,
):
    """
    Create feature matrix and binary target.
    """

    X = (
        df[features]
        .copy()
    )

    y = build_classification_target(
        df,
        features,
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
        .astype(int)
        .reset_index(drop=True)
    )

    return (
        X,
        y,
    )


def create_drifted_period(
    reference,
    features,
):
    """
    Create an artificial drifted copy of the
    monitoring period.

    This is used to test model degradation
    under controlled distribution change.
    """

    simulator = DriftSimulator(
        seed=SEED
    )

    drifted, metadata = (
        simulator.abrupt(
            df=reference.copy(),
            features=features,
            magnitude=DRIFT_MAGNITUDE,
            start_window=len(reference) // 2,
            baseline_df=reference,
        )
    )

    return (
        drifted,
        metadata,
    )


def calculate_drift_summary(
    reference,
    current,
    features,
):
    """
    Calculate drift metrics for the
    Person 2 experiment.
    """

    metrics = calculate_feature_metrics(
        reference_df=reference,
        current_df=current,
        features=features,
    )

    dsi = calculate_dsi(
        metrics
    )

    severity = classify_dsi(
        dsi
    )

    return (
        metrics,
        dsi,
        severity,
    )


def build_models():
    """
    Build the three classification models
    required for the Person 2 experiment.
    """

    return {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    SVC(),
                ),
            ]
        ),
        "naive_bayes": GaussianNB(),
    }


def calculate_reliability(
    baseline_metrics,
    drifted_metrics,
):
    """
    Calculate model reliability degradation.

    F1 is the primary reliability metric.
    """

    baseline_f1 = float(
        baseline_metrics["f1"]
    )

    drifted_f1 = float(
        drifted_metrics["f1"]
    )

    degradation = (
        baseline_f1
        - drifted_f1
    )

    if baseline_f1 > 0:
        degradation_ratio = (
            degradation
            / baseline_f1
        )
    else:
        degradation_ratio = np.nan

    if np.isnan(
        degradation_ratio
    ):
        status = "UNKNOWN"
    elif degradation_ratio < 0.05:
        status = "STABLE"
    elif degradation_ratio < 0.15:
        status = "WARNING"
    elif degradation_ratio < 0.30:
        status = "HIGH"
    else:
        status = "CRITICAL"

    return {
        "baseline_f1": baseline_f1,
        "drifted_f1": drifted_f1,
        "degradation": float(
            degradation
        ),
        "degradation_ratio": (
            float(degradation_ratio)
            if not np.isnan(
                degradation_ratio
            )
            else np.nan
        ),
        "reliability_status": status,
    }


def evaluate_models(
    reference,
    current,
    features,
):
    """
    Train each classification model on the
    reference period and evaluate it on both
    reference and drifted periods.
    """

    reference_X, reference_y = (
        prepare_supervised_data(
            reference,
            features,
        )
    )

    current_X, current_y = (
        prepare_supervised_data(
            current,
            features,
        )
    )

    models = build_models()

    results = {}

    for model_name, model in models.items():

        print()
        print(
            f"Evaluating {model_name}..."
        )

        model.fit(
            reference_X,
            reference_y,
        )

        reference_predictions = (
            model.predict(
                reference_X
            )
        )

        current_predictions = (
            model.predict(
                current_X
            )
        )

        reference_result = (
            evaluate_predictions(
                y_true=reference_y,
                y_pred=reference_predictions,
                task_type="classification",
                model_name=model_name,
            )
        )

        current_result = (
            evaluate_predictions(
                y_true=current_y,
                y_pred=current_predictions,
                task_type="classification",
                model_name=model_name,
            )
        )

        reference_metrics = (
            reference_result.to_dict()
        )

        current_metrics = (
            current_result.to_dict()
        )

        reliability = calculate_reliability(
            baseline_metrics=reference_metrics,
            drifted_metrics=current_metrics,
        )

        results[model_name] = {
            "reference": reference_metrics,
            "drifted": current_metrics,
            "reliability": reliability,
        }

        print(
            "  Reference F1:",
            round(
                reference_metrics["f1"],
                4,
            ),
        )

        print(
            "  Drifted F1:",
            round(
                current_metrics["f1"],
                4,
            ),
        )

        print(
            "  Reliability:",
            reliability[
                "reliability_status"
            ],
        )

    return results


def save_dataframe(
    dataframe,
    path,
):
    ensure_dir(
        path.parent
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def main():

    print("=" * 80)

    print(
        "DataDecay-X | Person 2 Experiment"
    )

    print("=" * 80)

    print()

    ensure_dir(
        RESULTS_DIR
    )

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print(
        "Loading dataset..."
    )

    df = load_electricity_dataset(
        DATASET_PATH
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    # ---------------------------------------------------------
    # TEMPORAL SPLIT
    # ---------------------------------------------------------

    (
        reference,
        monitoring,
        evaluation,
    ) = temporal_split(
        df,
        train_fraction=0.60,
        monitor_fraction=0.20,
    )

    print()

    print(
        "Reference:",
        reference.shape,
    )

    print(
        "Monitoring:",
        monitoring.shape,
    )

    print(
        "Evaluation:",
        evaluation.shape,
    )

    # ---------------------------------------------------------
    # FEATURE SELECTION
    # ---------------------------------------------------------

    features = select_features(
        reference
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

    # ---------------------------------------------------------
    # CREATE CONTROLLED DRIFT
    # ---------------------------------------------------------

    print()

    print(
        "Creating controlled drift..."
    )

    drifted_monitoring, drift_metadata = (
        create_drifted_period(
            monitoring,
            features,
        )
    )

    # ---------------------------------------------------------
    # DETECT DRIFT
    # ---------------------------------------------------------

    (
        drift_metrics,
        drift_dsi,
        drift_severity,
    ) = calculate_drift_summary(
        reference=reference,
        current=drifted_monitoring,
        features=features,
    )

    print()

    print(
        "Drift DSI:",
        round(
            drift_dsi,
            4,
        ),
    )

    print(
        "Drift severity:",
        drift_severity,
    )

    # ---------------------------------------------------------
    # MODEL EVALUATION
    # ---------------------------------------------------------

    print()

    print(
        "Evaluating classification models..."
    )

    model_results = evaluate_models(
        reference=reference,
        current=drifted_monitoring,
        features=features,
    )

    # ---------------------------------------------------------
    # FINAL MODEL SUMMARY
    # ---------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "MODEL RELIABILITY SUMMARY"
    )

    print(
        "=" * 80
    )

    for model_name, result in (
        model_results.items()
    ):

        reliability = (
            result["reliability"]
        )

        print()

        print(
            model_name
        )

        print(
            "  Reference F1:",
            round(
                result["reference"]["f1"],
                4,
            ),
        )

        print(
            "  Drifted F1:",
            round(
                result["drifted"]["f1"],
                4,
            ),
        )

        print(
            "  Degradation:",
            round(
                reliability["degradation"],
                4,
            ),
        )

        ratio = (
            reliability[
                "degradation_ratio"
            ]
        )

        if np.isnan(ratio):
            print(
                "  Degradation ratio: nan"
            )
        else:
            print(
                "  Degradation ratio:",
                round(
                    ratio,
                    4,
                ),
            )

        print(
            "  Reliability status:",
            reliability[
                "reliability_status"
            ],
        )

    # ---------------------------------------------------------
    # SAVE MODEL METRICS
    # ---------------------------------------------------------

    model_rows = []

    for model_name, result in (
        model_results.items()
    ):

        model_rows.append(
            {
                "model_name": model_name,
                "period": "reference",
                **result["reference"],
            }
        )

        model_rows.append(
            {
                "model_name": model_name,
                "period": "drifted",
                **result["drifted"],
            }
        )

    model_metrics = pd.DataFrame(
        model_rows
    )

    save_dataframe(
        model_metrics,
        RESULTS_DIR
        / "model_metrics.csv",
    )

    # ---------------------------------------------------------
    # SAVE RELIABILITY METRICS
    # ---------------------------------------------------------

    reliability_rows = []

    for model_name, result in (
        model_results.items()
    ):

        reliability_rows.append(
            {
                "model_name": model_name,
                **result["reliability"],
            }
        )

    reliability_metrics = pd.DataFrame(
        reliability_rows
    )

    save_dataframe(
        reliability_metrics,
        RESULTS_DIR
        / "reliability_metrics.csv",
    )

    # ---------------------------------------------------------
    # SAVE DRIFT METRICS
    # ---------------------------------------------------------

    save_dataframe(
        drift_metrics,
        RESULTS_DIR
        / "drift_metrics.csv",
    )

    # ---------------------------------------------------------
    # SAVE METADATA
    # ---------------------------------------------------------

    metadata = {
        "seed": SEED,
        "features": features,
        "drift": drift_metadata.to_dict(),
        "dsi": drift_dsi,
        "severity": drift_severity,
        "models": [
            "logistic_regression",
            "svm",
            "naive_bayes",
        ],
        "task_type": "classification",
        "target_definition": (
            "binary target indicating whether "
            "the next timestamp mean load is "
            "greater than the current timestamp "
            "mean load"
        ),
        "primary_reliability_metric": "f1",
    }

    save_json(
        metadata,
        RESULTS_DIR
        / "person2_metadata.json",
    )

    # ---------------------------------------------------------
    # SAVE DRIFTED DATA
    # ---------------------------------------------------------

    save_dataframe(
        drifted_monitoring,
        RESULTS_DIR
        / "drifted_monitoring.csv",
    )

    # ---------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------

    best_model = min(
        model_results,
        key=lambda name: (
            model_results[name][
                "reliability"
            ]["degradation"]
        ),
    )

    summary = {
        "drift_dsi": float(
            drift_dsi
        ),
        "drift_severity": (
            drift_severity
        ),
        "models_evaluated": list(
            model_results.keys()
        ),
        "best_model_by_f1_retention": (
            best_model
        ),
        "reliability": {
            name: result[
                "reliability"
            ]
            for name, result
            in model_results.items()
        },
    }

    save_json(
        summary,
        RESULTS_DIR
        / "person2_summary.json",
    )

    print()

    print("=" * 80)

    print(
        "PERSON 2 EXPERIMENT COMPLETE"
    )

    print("=" * 80)

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()