from pathlib import Path
import json

import numpy as np
import pandas as pd


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    """Save a Python object as formatted JSON."""
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            default=str,
        )


def load_electricity_dataset(path):
    """
    Load the LD2011_2014 electricity dataset.

    Dataset format:
    - semicolon separated
    - decimal comma
    - first column is timestamp
    - remaining columns are electricity series
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
    )

    if df.empty:
        raise ValueError(
            "Dataset was loaded but contains no rows."
        )

    first_column = df.columns[0]

    df = df.rename(
        columns={
            first_column: "timestamp"
        }
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps were found in the dataset."
        )

    feature_columns = [
        column
        for column in df.columns
        if column != "timestamp"
    ]

    for column in feature_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


def numeric_features(df, excluded=None):
    """
    Return numeric feature columns.

    Parameters
    ----------
    df : pandas.DataFrame
    excluded : iterable, optional
        Columns that should not be returned.
    """
    excluded = set(excluded or [])

    return [
        column
        for column in df.select_dtypes(
            include=np.number
        ).columns
        if column not in excluded
    ]


def validate_temporal_order(df):
    """Return True when timestamps are chronologically ordered."""
    if "timestamp" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'timestamp' column."
        )

    return df["timestamp"].is_monotonic_increasing


def temporal_split(
    df,
    train_fraction=0.60,
    monitor_fraction=0.20,
):
    """
    Split a dataframe chronologically.

    Returns
    -------
    reference
        Historical baseline period.

    monitoring
        Period used for drift monitoring.

    evaluation
        Final chronological period.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(
            "train_fraction must be between 0 and 1."
        )

    if not 0 < monitor_fraction < 1:
        raise ValueError(
            "monitor_fraction must be between 0 and 1."
        )

    if train_fraction + monitor_fraction >= 1:
        raise ValueError(
            "train_fraction + monitor_fraction "
            "must be less than 1."
        )

    if not validate_temporal_order(df):
        raise ValueError(
            "Input dataframe is not chronologically ordered."
        )

    n = len(df)

    reference_end = int(
        n * train_fraction
    )

    monitoring_end = int(
        n * (
            train_fraction
            + monitor_fraction
        )
    )

    reference = (
        df.iloc[:reference_end]
        .copy()
        .reset_index(drop=True)
    )

    monitoring = (
        df.iloc[
            reference_end:monitoring_end
        ]
        .copy()
        .reset_index(drop=True)
    )

    evaluation = (
        df.iloc[monitoring_end:]
        .copy()
        .reset_index(drop=True)
    )

    return (
        reference,
        monitoring,
        evaluation,
    )


def make_windows(
    df,
    window_size,
    step_size=None,
):
    """
    Create chronological windows.

    Parameters
    ----------
    window_size : int
        Number of rows per window.

    step_size : int, optional
        Number of rows between window starts.
        Defaults to window_size.
    """
    if window_size <= 0:
        raise ValueError(
            "window_size must be > 0."
        )

    if step_size is None:
        step_size = window_size

    if step_size <= 0:
        raise ValueError(
            "step_size must be > 0."
        )

    windows = []

    for window_id, start in enumerate(
        range(
            0,
            len(df) - window_size + 1,
            step_size,
        )
    ):
        end = start + window_size

        window = (
            df.iloc[start:end]
            .copy()
            .reset_index(drop=True)
        )

        window["window_id"] = window_id

        windows.append(window)

    return windows