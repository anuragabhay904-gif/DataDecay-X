from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    """
    Container for a forecast result.
    """
    method: str
    horizon: int
    predictions: np.ndarray
    metadata: dict

    def to_dict(self):
        return {
            "method": self.method,
            "horizon": self.horizon,
            "metadata": self.metadata,
        }


class PersistenceForecaster:
    """
    Persistence forecasting baseline.

    The forecast assumes that the next value
    remains equal to the latest observed value.
    """

    def __init__(self):
        self.last_value = None

    def fit(self, values):
        values = np.asarray(values, dtype=float)

        if values.size == 0:
            raise ValueError(
                "Cannot fit persistence forecaster on empty data."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Persistence training data contains non-finite values."
            )

        self.last_value = float(values[-1])
        return self

    def predict(self, horizon):
        if self.last_value is None:
            raise RuntimeError(
                "Forecaster must be fitted before prediction."
            )

        horizon = int(horizon)

        if horizon <= 0:
            raise ValueError(
                "horizon must be greater than zero."
            )

        return np.full(
            horizon,
            self.last_value,
            dtype=float,
        )


class LinearTrendForecaster:
    """
    Simple linear-trend forecasting model.

    Fits:

        y = slope * x + intercept

    using the most recent observations.
    """

    def __init__(self, lookback=None):
        self.lookback = lookback
        self.slope = None
        self.intercept = None
        self.training_size = None

    def fit(self, values):
        values = np.asarray(values, dtype=float)

        if values.size < 2:
            raise ValueError(
                "At least two observations are required "
                "for linear trend forecasting."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Linear trend training data contains "
                "non-finite values."
            )

        if self.lookback is not None and self.lookback <= 1:
            raise ValueError(
                "lookback must be greater than one."
            )

        if (
            self.lookback is not None
            and self.lookback < len(values)
        ):
            values = values[-self.lookback:]

        self.training_size = len(values)

        x = np.arange(
            self.training_size,
            dtype=float,
        )

        self.slope, self.intercept = np.polyfit(
            x,
            values,
            1,
        )

        return self

    def predict(self, horizon):
        if self.slope is None:
            raise RuntimeError(
                "Forecaster must be fitted before prediction."
            )

        horizon = int(horizon)

        if horizon <= 0:
            raise ValueError(
                "horizon must be greater than zero."
            )

        future_x = np.arange(
            self.training_size,
            self.training_size + horizon,
            dtype=float,
        )

        predictions = (
            self.slope * future_x
            + self.intercept
        )

        return predictions.astype(float)


@dataclass
class ReliabilityForecastResult:
    """
    Container for reliability forecast values.

    Reliability is constrained to the interval [0, 1].
    """

    method: str
    values: np.ndarray
    horizon: int
    metadata: dict
    slope: float = None
    intercept: float = None

    def to_dict(self):
        return {
            "method": self.method,
            "values": self.values,
            "horizon": self.horizon,
            "metadata": self.metadata,
            "slope": self.slope,
            "intercept": self.intercept,
        }


class ReliabilityForecaster:
    """
    Forecast model reliability using persistence
    and linear-trend forecasting.
    """

    @staticmethod
    def _bound_values(values):
        """
        Keep reliability forecasts inside [0, 1].
        """
        return np.clip(
            np.asarray(values, dtype=float),
            0.0,
            1.0,
        )

    def persistence(self, history, horizon):
        """
        Persistence reliability forecast.
        """
        forecaster = PersistenceForecaster()

        forecaster.fit(history)

        values = forecaster.predict(horizon)

        values = self._bound_values(values)

        return ReliabilityForecastResult(
            method="persistence",
            values=values,
            horizon=int(horizon),
            metadata={},
            slope=0.0,
            intercept=float(values[0]),
        )

    def linear_trend(
        self,
        history,
        horizon,
        lookback=None,
    ):
        """
        Linear-trend reliability forecast.

        Forecast values are clipped to [0, 1].
        The fitted slope and intercept are exposed
        on the result object.
        """
        forecaster = LinearTrendForecaster(
            lookback=lookback
        )

        forecaster.fit(history)

        values = forecaster.predict(horizon)

        values = self._bound_values(values)

        return ReliabilityForecastResult(
            method="linear_trend",
            values=values,
            horizon=int(horizon),
            metadata={
                "lookback": lookback,
            },
            slope=float(forecaster.slope),
            intercept=float(forecaster.intercept),
        )

    def forecast_table(
        self,
        history,
        horizon,
        lookback=None,
    ):
        """
        Return forecasts from both core methods
        in tabular form.
        """
        persistence_result = self.persistence(
            history,
            horizon,
        )

        linear_result = self.linear_trend(
            history,
            horizon,
            lookback=lookback,
        )

        return pd.DataFrame(
            {
                "forecast_step": np.arange(
                    1,
                    int(horizon) + 1,
                ),
                "persistence": persistence_result.values,
                "linear_trend": linear_result.values,
            }
        )


def forecast_series(
    values,
    horizon,
    method="persistence",
    lookback=None,
):
    """
    Forecast a one-dimensional series.
    """
    values = np.asarray(values, dtype=float)

    if method == "persistence":
        forecaster = PersistenceForecaster()

    elif method == "linear_trend":
        forecaster = LinearTrendForecaster(
            lookback=lookback
        )

    else:
        raise ValueError(
            f"Unknown forecasting method: {method}"
        )

    forecaster.fit(values)

    predictions = forecaster.predict(horizon)

    return ForecastResult(
        method=method,
        horizon=int(horizon),
        predictions=predictions,
        metadata={
            "lookback": lookback,
        },
    )


def rolling_forecast(
    values,
    horizon,
    method="persistence",
    min_history=5,
):
    """
    Generate rolling forecasts across a series.
    """
    values = np.asarray(values, dtype=float)

    if len(values) <= min_history:
        raise ValueError(
            "Not enough observations for rolling forecast."
        )

    forecasts = []

    for end in range(
        min_history,
        len(values),
    ):
        history = values[:end]

        result = forecast_series(
            history,
            horizon=horizon,
            method=method,
        )

        forecasts.append(
            {
                "origin": end,
                "forecast": result.predictions,
            }
        )

    return forecasts


def forecast_dataframe(
    dataframe,
    column,
    horizon,
    method="persistence",
    lookback=None,
):
    """
    Forecast a column from a dataframe.
    """
    if column not in dataframe.columns:
        raise KeyError(
            f"Column '{column}' not found in dataframe."
        )

    values = dataframe[column].dropna().to_numpy(
        dtype=float
    )

    return forecast_series(
        values,
        horizon=horizon,
        method=method,
        lookback=lookback,
    )


def forecast_multiple_features(
    dataframe,
    columns,
    horizon,
    method="persistence",
    lookback=None,
):
    """
    Forecast multiple dataframe columns.
    """
    results = {}

    for column in columns:
        results[column] = forecast_dataframe(
            dataframe,
            column,
            horizon=horizon,
            method=method,
            lookback=lookback,
        )

    return results


def calculate_forecast_metrics(
    actual,
    predicted,
):
    """
    Calculate basic forecast error metrics.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError(
            "actual and predicted must have the same length."
        )

    error = actual - predicted

    mae = float(
        np.mean(np.abs(error))
    )

    mse = float(
        np.mean(error ** 2)
    )

    rmse = float(
        np.sqrt(mse)
    )

    non_zero = np.abs(actual) > 1e-12

    if np.any(non_zero):
        mape = float(
            np.mean(
                np.abs(
                    error[non_zero]
                    / actual[non_zero]
                )
            )
            * 100.0
        )
    else:
        mape = float("nan")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
    }


def evaluate_forecast(
    actual,
    predicted,
):
    """
    Evaluate forecast predictions.
    """
    return calculate_forecast_metrics(
        actual,
        predicted,
    )


def compare_forecasters(
    values,
    horizon,
    methods=None,
):
    """
    Compare multiple forecasting methods.
    """
    if methods is None:
        methods = [
            "persistence",
            "linear_trend",
        ]

    results = {}

    for method in methods:
        results[method] = forecast_series(
            values,
            horizon=horizon,
            method=method,
        )

    return results