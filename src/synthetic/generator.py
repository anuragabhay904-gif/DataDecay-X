"""
Synthetic data generation for DataDecay-X.

This module provides lightweight synthetic-data generators for
time-series/tabular electricity data while preserving the
project's chronological structure.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticGenerationResult:
    """Container for generated synthetic data and metadata."""

    data: pd.DataFrame
    method: str
    seed: int
    source_rows: int
    generated_rows: int
    features: list[str]

    def to_dict(self):
        """Return generation metadata as a dictionary."""

        return {
            "method": self.method,
            "seed": self.seed,
            "source_rows": self.source_rows,
            "generated_rows": self.generated_rows,
            "features": self.features,
        }


def _validate_reference(reference_df):
    """Validate the reference dataframe."""

    if not isinstance(
        reference_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "reference_df must be a pandas DataFrame."
        )

    if reference_df.empty:
        raise ValueError(
            "reference_df cannot be empty."
        )

    return reference_df


def _select_features(
    reference_df,
    features=None,
):
    """Select and validate numeric features."""

    if features is None:
        features = [
            column
            for column in reference_df.columns
            if column != "timestamp"
            and pd.api.types.is_numeric_dtype(
                reference_df[column]
            )
        ]

    features = list(features)

    if not features:
        raise ValueError(
            "At least one numeric feature is required."
        )

    missing = [
        feature
        for feature in features
        if feature not in reference_df.columns
    ]

    if missing:
        raise ValueError(
            "Features not found in reference dataframe: "
            f"{missing}"
        )

    non_numeric = [
        feature
        for feature in features
        if not pd.api.types.is_numeric_dtype(
            reference_df[feature]
        )
    ]

    if non_numeric:
        raise ValueError(
            "Synthetic generation requires numeric features: "
            f"{non_numeric}"
        )

    return features


def _infer_time_frequency(
    timestamps,
):
    """
    Infer the most common timestamp interval.

    Returns
    -------
    pandas.Timedelta or None
    """

    timestamps = pd.to_datetime(
        timestamps,
        errors="coerce",
    )

    timestamps = (
        timestamps
        .dropna()
        .sort_values()
        .drop_duplicates()
    )

    if len(timestamps) < 2:
        return None

    differences = timestamps.diff().dropna()

    if differences.empty:
        return None

    return differences.mode().iloc[0]


def _generate_timestamps(
    reference_df,
    n_rows,
):
    """
    Generate chronological timestamps following the reference
    dataset's observed frequency.
    """

    if "timestamp" not in reference_df.columns:
        return None

    timestamps = pd.to_datetime(
        reference_df["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise ValueError(
            "Reference timestamps contain invalid values."
        )

    frequency = _infer_time_frequency(
        timestamps
    )

    if frequency is None:
        frequency = pd.Timedelta(
            minutes=15
        )

    start = timestamps.iloc[-1] + frequency

    return pd.date_range(
        start=start,
        periods=n_rows,
        freq=frequency,
    )


def _bootstrap_values(
    reference_df,
    features,
    n_rows,
    rng,
):
    """
    Generate values by independently bootstrapping each feature.
    """

    generated = {}

    for feature in features:

        values = pd.to_numeric(
            reference_df[feature],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        if len(values) == 0:
            raise ValueError(
                f"Feature '{feature}' contains no valid values."
            )

        indices = rng.integers(
            0,
            len(values),
            size=n_rows,
        )

        generated[feature] = values[
            indices
        ]

    return pd.DataFrame(
        generated
    )


def _perturb_values(
    reference_df,
    features,
    n_rows,
    rng,
    noise_scale=0.01,
):
    """
    Generate values by bootstrapping observations and adding
    controlled Gaussian perturbation.
    """

    if noise_scale < 0:
        raise ValueError(
            "noise_scale cannot be negative."
        )

    generated = {}

    for feature in features:

        values = pd.to_numeric(
            reference_df[feature],
            errors="coerce",
        ).dropna().to_numpy(
            dtype=float
        )

        if len(values) == 0:
            raise ValueError(
                f"Feature '{feature}' contains no valid values."
            )

        indices = rng.integers(
            0,
            len(values),
            size=n_rows,
        )

        sampled = values[
            indices
        ].copy()

        std = float(
            np.std(values)
        )

        noise = rng.normal(
            loc=0.0,
            scale=std * noise_scale,
            size=n_rows,
        )

        generated[feature] = (
            sampled + noise
        )

    return pd.DataFrame(
        generated
    )


def _block_bootstrap_values(
    reference_df,
    features,
    n_rows,
    rng,
    block_size=96,
):
    """
    Generate synthetic observations using sampled chronological
    blocks.

    Block bootstrap preserves more local temporal structure than
    independently sampling every feature.
    """

    if block_size <= 0:
        raise ValueError(
            "block_size must be greater than zero."
        )

    values = (
        reference_df[features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .copy()
    )

    values = values.dropna(
        how="any"
    )

    if values.empty:
        raise ValueError(
            "No complete numeric rows are available "
            "for block bootstrap."
        )

    if len(values) < block_size:
        block_size = len(values)

    blocks = []

    remaining = n_rows

    maximum_start = (
        len(values) - block_size
    )

    while remaining > 0:

        if maximum_start > 0:
            start = int(
                rng.integers(
                    0,
                    maximum_start + 1,
                )
            )
        else:
            start = 0

        block = values.iloc[
            start:start + block_size
        ]

        take = min(
            remaining,
            len(block),
        )

        blocks.append(
            block.iloc[:take]
        )

        remaining -= take

    return pd.concat(
        blocks,
        ignore_index=True,
    )


def generate_synthetic_data(
    df=None,
    features=None,
    n_samples=None,
    method="bootstrap",
    random_state=42,
    noise_scale=0.01,
    reference=None,
    reference_df=None,
    n_rows=None,
    seed=None,
    block_size=96,
):
    """
    Generate synthetic data from a reference dataset.

    Supports both the original project API and the Person 3
    pipeline API.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Reference dataframe.

    reference : pandas.DataFrame, optional
        Alias for ``df``.

    reference_df : pandas.DataFrame, optional
        Backwards-compatible alias for ``df``.

    features : iterable, optional
        Numeric columns to generate.

    n_samples : int, optional
        Number of synthetic rows.

    n_rows : int, optional
        Backwards-compatible alias for ``n_samples``.

    method : str
        One of:

        - ``bootstrap``
        - ``perturbation``
        - ``block_bootstrap``

    random_state : int, optional
        Random seed.

    seed : int, optional
        Backwards-compatible alias for ``random_state``.

    noise_scale : float
        Noise multiplier used by ``perturbation``.

    block_size : int
        Number of chronological rows per block for
        ``block_bootstrap``.

    Returns
    -------
    SyntheticGenerationResult
    """

    # ----------------------------------------------------------
    # Resolve reference dataframe aliases
    # ----------------------------------------------------------

    if df is None:
        if reference is not None:
            df = reference
        elif reference_df is not None:
            df = reference_df

    if df is None:
        raise ValueError(
            "A reference dataframe must be provided using "
            "df, reference, or reference_df."
        )

    reference_df = _validate_reference(df)

    if n_rows is not None:
        n_samples = n_rows

    if n_samples is None:
        n_samples = len(reference_df)

    if not isinstance(
        n_samples,
        (int, np.integer),
    ):
        raise TypeError(
            "n_samples must be an integer."
        )

    n_samples = int(n_samples)

    if n_samples <= 0:
        raise ValueError(
            "n_samples must be greater than zero."
        )

    # ----------------------------------------------------------
    # Resolve seed aliases
    # ----------------------------------------------------------

    if seed is not None:
        random_state = seed

    random_state = int(random_state)

    # ----------------------------------------------------------
    # Validate/select features
    # ----------------------------------------------------------

    features = _select_features(
        reference_df,
        features,
    )

    # ----------------------------------------------------------
    # Validate method
    # ----------------------------------------------------------

    if not isinstance(method, str):
        raise TypeError(
            "method must be a string."
        )

    method = method.strip().lower()

    # ----------------------------------------------------------
    # Random generator
    # ----------------------------------------------------------

    rng = np.random.default_rng(
        random_state
    )

    # ----------------------------------------------------------
    # Generate synthetic values
    # ----------------------------------------------------------

    if method == "bootstrap":

        generated = _bootstrap_values(
            reference_df,
            features,
            n_samples,
            rng,
        )

    elif method in {
        "perturbation",
        "noise",
        "bootstrap_noise",
    }:

        generated = _perturb_values(
            reference_df,
            features,
            n_samples,
            rng,
            noise_scale=noise_scale,
        )

    elif method in {
        "block_bootstrap",
        "block",
    }:

        generated = _block_bootstrap_values(
            reference_df,
            features,
            n_samples,
            rng,
            block_size=block_size,
        )

    else:
        raise ValueError(
            f"Unknown synthetic generation method: {method}. "
            "Supported methods are: bootstrap, perturbation, "
            "block_bootstrap."
        )

    # ----------------------------------------------------------
    # Add future timestamps
    # ----------------------------------------------------------

    if "timestamp" in reference_df.columns:

        timestamps = _generate_timestamps(
            reference_df,
            n_samples,
        )

        generated.insert(
            0,
            "timestamp",
            timestamps,
        )

    # ----------------------------------------------------------
    # Return result
    # ----------------------------------------------------------

    return SyntheticGenerationResult(
        data=generated,
        method=method,
        seed=random_state,
        source_rows=len(reference_df),
        generated_rows=len(generated),
        features=features,
    )


def generate_bootstrap(
    reference_df,
    n_rows=None,
    features=None,
    seed=42,
):
    """Generate synthetic data using bootstrap sampling."""

    return generate_synthetic_data(
        reference_df=reference_df,
        n_rows=n_rows,
        features=features,
        method="bootstrap",
        seed=seed,
    )


def generate_perturbed(
    reference_df,
    n_rows=None,
    features=None,
    seed=42,
    noise_scale=0.01,
):
    """Generate synthetic data using bootstrap plus noise."""

    return generate_synthetic_data(
        reference_df=reference_df,
        n_rows=n_rows,
        features=features,
        method="perturbation",
        seed=seed,
        noise_scale=noise_scale,
    )


def generate_block_bootstrap(
    reference_df,
    n_rows=None,
    features=None,
    seed=42,
    block_size=96,
):
    """Generate synthetic data using chronological blocks."""

    return generate_synthetic_data(
        reference_df=reference_df,
        n_rows=n_rows,
        features=features,
        method="block_bootstrap",
        seed=seed,
        block_size=block_size,
    )


__all__ = [
    "SyntheticGenerationResult",
    "generate_synthetic_data",
    "generate_bootstrap",
    "generate_perturbed",
    "generate_block_bootstrap",
]