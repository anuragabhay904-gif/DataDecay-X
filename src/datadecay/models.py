from dataclasses import dataclass

import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelResult:
    """
    Container for model evaluation results.
    """

    model_name: str
    task_type: str
    metrics: dict

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            **self.metrics,
        }


class BaselineModel:
    """
    Simple baseline model wrapper.

    Classification:
        Logistic Regression

    Regression:
        Mean prediction baseline
    """

    def __init__(
        self,
        task_type="classification",
        random_state=42,
    ):
        self.task_type = (
            task_type.lower().strip()
        )

        self.random_state = int(
            random_state
        )

        self.model = None

        self.feature_columns = None

        self.target_column = None

        self.mean_value = None

    def _validate_task_type(self):
        valid_tasks = {
            "classification",
            "regression",
        }

        if self.task_type not in valid_tasks:
            raise ValueError(
                "task_type must be either "
                "'classification' or 'regression'."
            )

    def fit(
        self,
        X,
        y,
    ):
        """
        Fit the baseline model.
        """

        self._validate_task_type()

        if X is None:
            raise ValueError(
                "X cannot be None."
            )

        if y is None:
            raise ValueError(
                "y cannot be None."
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same "
                "number of rows."
            )

        self.feature_columns = list(
            X.columns
        )

        if self.task_type == "classification":

            self.model = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=(
                                self.random_state
                            ),
                        ),
                    ),
                ]
            )

            self.model.fit(
                X,
                y,
            )

        else:

            values = pd.to_numeric(
                y,
                errors="coerce",
            )

            if values.isna().any():
                raise ValueError(
                    "Regression target contains "
                    "non-numeric values."
                )

            self.mean_value = float(
                values.mean()
            )

        return self

    def predict(
        self,
        X,
    ):
        """
        Generate predictions.
        """

        if self.feature_columns is None:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        missing = [
            column
            for column in self.feature_columns
            if column not in X.columns
        ]

        if missing:
            raise ValueError(
                "Missing model features: "
                f"{missing}"
            )

        X_model = X[
            self.feature_columns
        ]

        if self.task_type == "classification":

            return self.model.predict(
                X_model
            )

        return np.full(
            len(X_model),
            self.mean_value,
        )

    def predict_proba(
        self,
        X,
    ):
        """
        Generate classification probabilities.
        """

        if self.task_type != "classification":
            raise ValueError(
                "predict_proba is only available "
                "for classification."
            )

        if self.model is None:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        X_model = X[
            self.feature_columns
        ]

        return self.model.predict_proba(
            X_model
        )

class ModelManager:
    """
    Manage the core classification models used by the
    reliability tests.

    Models:
        - logistic_regression
        - svm
        - naive_bayes
    """

    def __init__(self, random_state=42):
        self.random_state = int(random_state)

        self.models = {
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
                            random_state=self.random_state,
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

        self.trained_models = {}

    def train_all(self, X, y):
        """
        Train all registered models.
        """

        if X is None:
            raise ValueError("X cannot be None.")

        if y is None:
            raise ValueError("y cannot be None.")

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same "
                "number of rows."
            )

        for name, model in self.models.items():
            model.fit(X, y)
            self.trained_models[name] = model

        return self

    def predict_all(self, X):
        """
        Generate predictions from all trained models.
        """

        if not self.trained_models:
            raise RuntimeError(
                "Models must be trained before prediction."
            )

        return {
            name: model.predict(X)
            for name, model in self.trained_models.items()
        }

def classification_metrics(
    y_true,
    y_pred,
):
    """
    Calculate standard classification metrics.
    """

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def regression_metrics(
    y_true,
    y_pred,
):
    """
    Calculate standard regression metrics.
    """

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            rmse
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }


def evaluate_model(
    model,
    X,
    y,
    task_type,
    model_name="model",
):
    """
    Fit and evaluate a model.
    """

    task_type = (
        task_type.lower()
        .strip()
    )

    model.fit(
        X,
        y,
    )

    predictions = model.predict(
        X
    )

    if task_type == "classification":

        metrics = classification_metrics(
            y,
            predictions,
        )

    elif task_type == "regression":

        metrics = regression_metrics(
            y,
            predictions,
        )

    else:

        raise ValueError(
            "task_type must be either "
            "'classification' or 'regression'."
        )

    return ModelResult(
        model_name=model_name,
        task_type=task_type,
        metrics=metrics,
    )


def evaluate_predictions(
    y_true,
    y_pred,
    task_type,
    model_name="model",
):
    """
    Evaluate already-generated predictions.
    """

    task_type = (
        task_type.lower()
        .strip()
    )

    if task_type == "classification":

        metrics = classification_metrics(
            y_true,
            y_pred,
        )

    elif task_type == "regression":

        metrics = regression_metrics(
            y_true,
            y_pred,
        )

    else:

        raise ValueError(
            "task_type must be either "
            "'classification' or 'regression'."
        )

    return ModelResult(
        model_name=model_name,
        task_type=task_type,
        metrics=metrics,
    )


def compare_model_results(
    results,
):
    """
    Convert ModelResult objects into a dataframe.
    """

    if not results:
        return pd.DataFrame()

    rows = [
        result.to_dict()
        for result in results
    ]

    return pd.DataFrame(
        rows
    )


def temporal_model_evaluation(
    model,
    reference_X,
    reference_y,
    evaluation_X,
    evaluation_y,
    task_type,
    model_name="model",
):
    """
    Train only on the historical/reference period
    and evaluate on a later chronological period.

    This deliberately avoids random train/test splitting.
    """

    model.fit(
        reference_X,
        reference_y,
    )

    predictions = model.predict(
        evaluation_X
    )

    return evaluate_predictions(
        y_true=evaluation_y,
        y_pred=predictions,
        task_type=task_type,
        model_name=model_name,
    )