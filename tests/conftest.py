from pathlib import Path

import pytest


from src.datadecay import (
    load_electricity_dataset,
    numeric_features,
    temporal_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)


def _select_features(
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


def _build_supervised_data(
    df,
    features,
):
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


@pytest.fixture
def df():
    return load_electricity_dataset(
        DATASET_PATH
    )


@pytest.fixture
def temporal_data(df):
    return temporal_split(
        df,
        train_fraction=0.60,
        monitor_fraction=0.20,
    )


@pytest.fixture
def reference(temporal_data):
    return temporal_data[0]


@pytest.fixture
def monitoring(temporal_data):
    return temporal_data[1]


@pytest.fixture
def evaluation(temporal_data):
    return temporal_data[2]


@pytest.fixture
def features(reference):
    return _select_features(
        reference,
        count=5,
    )


@pytest.fixture
def supervised_data(reference, features):
    return _build_supervised_data(
        reference,
        features,
    )


@pytest.fixture
def X(supervised_data):
    return supervised_data[0]


@pytest.fixture
def y(supervised_data):
    return supervised_data[1]


@pytest.fixture
def predictions(X, y):
    from src.datadecay.models import BaselineModel

    model = BaselineModel(
        task_type="regression",
        random_state=42,
    )

    model.fit(
        X,
        y,
    )

    return model.predict(
        X
    )