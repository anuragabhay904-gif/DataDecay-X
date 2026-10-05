"""
Baseline models for DataDecay-X.

This module provides simple, deterministic baseline models that
can be used as reference points before evaluating more advanced
candidate models.

The project works with both regression and classification tasks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselinePrediction:
    """Container for baseline model predictions."""

    predictions: np.ndarray
    model_type: str
    task: str


class MeanRegressor:
    """
    Simple regression baseline.

    The model predicts the mean target value observed during
    training. It provides a stable reference for evaluating
    more sophisticated regression models.
    """

    def __init__(self):
        self.mean_value = None
        self.is_fitted = False

    def fit(self, X, y):
        """Fit the model using the mean training target."""

        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            y = y.reshape(-1)

        if len(y) == 0:
            raise ValueError(
                "Cannot fit MeanRegressor with empty target data."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "Target contains NaN or infinite values."
            )

        self.mean_value = float(np.mean(y))
        self.is_fitted = True

        return self

    def predict(self, X):
        """Predict the training mean for every input row."""

        if not self.is_fitted:
            raise RuntimeError(
                "MeanRegressor must be fitted before prediction."
            )

        if hasattr(X, "__len__"):
            n_rows = len(X)
        else:
            n_rows = 1

        return np.full(
            n_rows,
            self.mean_value,
            dtype=float,
        )


class ClassificationBaseline:
    """
    Logistic-regression classification baseline.

    Standardization is included in the pipeline so that the
    baseline behaves consistently across features with different
    scales.
    """

    def __init__(
        self,
        random_state=42,
        max_iter=1000,
    ):
        self.random_state = random_state
        self.max_iter = max_iter

        self.model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=random_state,
                        max_iter=max_iter,
                    ),
                ),
            ]
        )

        self.is_fitted = False

    def fit(self, X, y):
        """Fit the classification baseline."""

        X = self._validate_features(X)
        y = self._validate_target(y)

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Classification requires at least two classes."
            )

        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, X):
        """Generate class predictions."""

        if not self.is_fitted:
            raise RuntimeError(
                "ClassificationBaseline must be fitted "
                "before prediction."
            )

        X = self._validate_features(X)

        return self.model.predict(X)

    def predict_proba(self, X):
        """Generate class probabilities."""

        if not self.is_fitted:
            raise RuntimeError(
                "ClassificationBaseline must be fitted "
                "before prediction."
            )

        X = self._validate_features(X)

        return self.model.predict_proba(X)

    @staticmethod
    def _validate_features(X):
        """Convert feature data to a numeric array."""

        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(
                include=np.number
            )

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(
                "Features must be a 2-dimensional array."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "No numeric features were provided."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Features contain NaN or infinite values."
            )

        return X

    @staticmethod
    def _validate_target(y):
        """Validate a classification target."""

        y = np.asarray(y)

        if y.ndim != 1:
            y = y.reshape(-1)

        if len(y) == 0:
            raise ValueError(
                "Target cannot be empty."
            )

        if pd.isna(y).any():
            raise ValueError(
                "Target contains missing values."
            )

        return y


def train_baseline(
    X,
    y,
    task="regression",
    random_state=42,
):
    """
    Train the appropriate baseline model.

    Parameters
    ----------
    X : array-like or DataFrame
        Training features.

    y : array-like
        Training target.

    task : str
        Either ``regression`` or ``classification``.

    random_state : int
        Random seed used by the classification model.

    Returns
    -------
    model
        Fitted baseline model.
    """

    task = task.lower()

    if task == "regression":
        model = MeanRegressor()

    elif task == "classification":
        model = ClassificationBaseline(
            random_state=random_state
        )

    else:
        raise ValueError(
            "Unsupported task. "
            "Use 'regression' or 'classification'."
        )

    model.fit(X, y)

    return model


def predict_baseline(
    model,
    X,
    task="regression",
):
    """
    Generate predictions from a fitted baseline model.
    """

    task = task.lower()

    if task not in {
        "regression",
        "classification",
    }:
        raise ValueError(
            "Unsupported task. "
            "Use 'regression' or 'classification'."
        )

    if not hasattr(model, "predict"):
        raise TypeError(
            "model must provide a predict() method."
        )

    predictions = model.predict(X)

    return BaselinePrediction(
        predictions=np.asarray(predictions),
        model_type=type(model).__name__,
        task=task,
    )


__all__ = [
    "BaselinePrediction",
    "MeanRegressor",
    "ClassificationBaseline",
    "train_baseline",
    "predict_baseline",
]