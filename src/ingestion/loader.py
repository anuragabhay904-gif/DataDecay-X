"""
Dataset loading utilities for DataDecay-X.

This module provides the legacy ingestion interface while
keeping the actual electricity-dataset parsing consistent with
the main ``src.datadecay.utils`` implementation.
"""

from pathlib import Path

import pandas as pd

from src.datadecay.utils import (
    load_electricity_dataset,
)


def load_dataset(path):
    """
    Load a DataDecay-X dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded and chronologically ordered dataset.
    """

    return load_electricity_dataset(
        path
    )


def load_csv(
    path,
    delimiter=",",
    decimal=".",
    parse_dates=False,
):
    """
    Load a generic CSV file.

    This helper is intended for processed experiment outputs
    rather than the raw LD2011_2014 dataset.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {path}"
        )

    df = pd.read_csv(
        path,
        sep=delimiter,
        decimal=decimal,
        parse_dates=parse_dates,
    )

    if df.empty:
        raise ValueError(
            f"Dataset contains no rows: {path}"
        )

    return df


def load_processed_data(
    path,
):
    """
    Load a processed DataDecay-X CSV file.

    Processed files use the normal comma-separated CSV format.
    """

    return load_csv(
        path,
        delimiter=",",
        decimal=".",
        parse_dates=False,
    )


def load_drifted_data(
    path,
):
    """
    Load a drifted experiment dataset.

    The function keeps the file contents unchanged except for
    parsing the CSV structure.
    """

    return load_csv(
        path,
        delimiter=",",
        decimal=".",
        parse_dates=False,
    )


def load_synthetic_data(
    path,
):
    """
    Load an accepted or generated synthetic dataset.
    """

    return load_csv(
        path,
        delimiter=",",
        decimal=".",
        parse_dates=False,
    )


def load_dataset_with_timestamp(
    path,
    timestamp_column="timestamp",
):
    """
    Load a CSV dataset and parse its timestamp column.

    This helper is useful for processed, drifted, and synthetic
    datasets that have already been converted to standard CSV.
    """

    df = load_csv(
        path,
        delimiter=",",
        decimal=".",
        parse_dates=False,
    )

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_column}' "
            "was not found."
        )

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    if df[timestamp_column].isna().any():
        raise ValueError(
            "Invalid timestamps were found."
        )

    return (
        df.sort_values(
            timestamp_column
        )
        .reset_index(drop=True)
    )


__all__ = [
    "load_dataset",
    "load_csv",
    "load_processed_data",
    "load_drifted_data",
    "load_synthetic_data",
    "load_dataset_with_timestamp",
]