from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticGenerationResult:
    """
    Container for generated synthetic samples and generation metadata.
    """

    samples: pd.DataFrame
    method: str
    source_features: List[str]
    n_samples: int
    random_state: int
    metadata: Dict

    def to_dict(self) -> Dict:
        """
        Convert result metadata to a serializable dictionary.
        """
        result = asdict(self)

        result["samples"] = {
            "rows": int(self.samples.shape[0]),
            "columns": int(self.samples.shape[1]),
        }

        return result


class SyntheticGenerator:
    """
    Synthetic data generator for tabular numerical data.

    The generator is deliberately conservative:
    - source distributions are learned from reference data
    - synthetic values remain within configurable bounds
    - generated samples preserve feature names
    - random state is explicit for reproducibility
    """

    def __init__(
        self,
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.rng = np.random.default_rng(
            random_state
        )

    @staticmethod
    def _validate_input(
        df: pd.DataFrame,
        features: Sequence[str],
    ) -> None:
        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Input dataframe is empty."
            )

        if not features:
            raise ValueError(
                "At least one feature is required."
            )

        missing = [
            feature
            for feature in features
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing features: {missing}"
            )

        non_numeric = [
            feature
            for feature in features
            if not pd.api.types.is_numeric_dtype(
                df[feature]
            )
        ]

        if non_numeric:
            raise TypeError(
                "Synthetic generation requires "
                f"numeric features: {non_numeric}"
            )

    @staticmethod
    def _clean_source(
        df: pd.DataFrame,
        features: Sequence[str],
    ) -> pd.DataFrame:
        clean = (
            df[list(features)]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
            .copy()
        )

        if clean.empty:
            raise ValueError(
                "No valid numeric rows remain "
                "after cleaning."
            )

        return clean

    def gaussian(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        n_samples: int,
        clip_quantiles: Tuple[
            float,
            float
        ] = (0.01, 0.99),
    ) -> SyntheticGenerationResult:
        """
        Generate synthetic observations using a multivariate
        Gaussian distribution estimated from the source data.
        """

        self._validate_input(
            df,
            features,
        )

        if n_samples <= 0:
            raise ValueError(
                "n_samples must be positive."
            )

        lower_q, upper_q = (
            clip_quantiles
        )

        if not (
            0.0
            <= lower_q
            < upper_q
            <= 1.0
        ):
            raise ValueError(
                "clip_quantiles must satisfy "
                "0 <= lower < upper <= 1."
            )

        source = self._clean_source(
            df,
            features,
        )

        values = source.to_numpy(
            dtype=float
        )

        mean = np.mean(
            values,
            axis=0,
        )

        covariance = np.cov(
            values,
            rowvar=False,
        )

        if len(features) == 1:
            covariance = np.asarray(
                [[float(covariance)]]
            )

        covariance = np.atleast_2d(
            covariance
        )

        # Numerical regularization prevents singular covariance
        # matrices from making generation unstable.
        covariance += (
            np.eye(
                covariance.shape[0]
            )
            * 1e-8
        )

        generated = (
            self.rng.multivariate_normal(
                mean=mean,
                cov=covariance,
                size=n_samples,
            )
        )

        lower = source.quantile(
            lower_q
        ).to_numpy(
            dtype=float
        )

        upper = source.quantile(
            upper_q
        ).to_numpy(
            dtype=float
        )

        generated = np.clip(
            generated,
            lower,
            upper,
        )

        synthetic = pd.DataFrame(
            generated,
            columns=list(features),
        )

        metadata = {
            "method": "gaussian",
            "source_rows": int(
                len(source)
            ),
            "clip_quantiles": [
                float(lower_q),
                float(upper_q),
            ],
            "source_means": {
                feature: float(
                    source[feature].mean()
                )
                for feature in features
            },
            "source_stds": {
                feature: float(
                    source[feature].std()
                )
                for feature in features
            },
        }

        return SyntheticGenerationResult(
            samples=synthetic,
            method="gaussian",
            source_features=list(
                features
            ),
            n_samples=int(n_samples),
            random_state=self.random_state,
            metadata=metadata,
        )

    def bootstrap(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        n_samples: int,
        noise_scale: float = 0.01,
    ) -> SyntheticGenerationResult:
        """
        Generate synthetic observations by bootstrap resampling
        followed by small distribution-aware Gaussian perturbations.
        """

        self._validate_input(
            df,
            features,
        )

        if n_samples <= 0:
            raise ValueError(
                "n_samples must be positive."
            )

        if noise_scale < 0:
            raise ValueError(
                "noise_scale must be non-negative."
            )

        source = self._clean_source(
            df,
            features,
        )

        source_values = source.to_numpy(
            dtype=float
        )

        indices = self.rng.integers(
            low=0,
            high=len(source_values),
            size=n_samples,
        )

        generated = (
            source_values[indices]
            .copy()
        )

        if noise_scale > 0:
            std = np.std(
                source_values,
                axis=0,
                ddof=1,
            )

            std = np.where(
                np.isfinite(std),
                std,
                0.0,
            )

            noise = (
                self.rng.normal(
                    loc=0.0,
                    scale=std * noise_scale,
                    size=generated.shape,
                )
            )

            generated += noise

        synthetic = pd.DataFrame(
            generated,
            columns=list(features),
        )

        metadata = {
            "method": "bootstrap",
            "source_rows": int(
                len(source)
            ),
            "noise_scale": float(
                noise_scale
            ),
        }

        return SyntheticGenerationResult(
            samples=synthetic,
            method="bootstrap",
            source_features=list(
                features
            ),
            n_samples=int(n_samples),
            random_state=self.random_state,
            metadata=metadata,
        )

    def empirical(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        n_samples: int,
    ) -> SyntheticGenerationResult:
        """
        Generate synthetic observations through independent
        empirical sampling from each feature distribution.
        """

        self._validate_input(
            df,
            features,
        )

        if n_samples <= 0:
            raise ValueError(
                "n_samples must be positive."
            )

        source = self._clean_source(
            df,
            features,
        )

        generated = {}

        for feature in features:
            values = (
                source[feature]
                .to_numpy(
                    dtype=float
                )
            )

            indices = (
                self.rng.integers(
                    0,
                    len(values),
                    size=n_samples,
                )
            )

            generated[feature] = (
                values[indices]
            )

        synthetic = pd.DataFrame(
            generated
        )

        metadata = {
            "method": "empirical",
            "source_rows": int(
                len(source)
            ),
        }

        return SyntheticGenerationResult(
            samples=synthetic,
            method="empirical",
            source_features=list(
                features
            ),
            n_samples=int(n_samples),
            random_state=self.random_state,
            metadata=metadata,
        )

    def generate(
        self,
        df: pd.DataFrame,
        features: Sequence[str],
        n_samples: int,
        method: str = "bootstrap",
        **kwargs,
    ) -> SyntheticGenerationResult:
        """
        Dispatch synthetic generation to the selected method.
        """

        method = method.lower().strip()

        if method == "bootstrap":
            return self.bootstrap(
                df=df,
                features=features,
                n_samples=n_samples,
                **kwargs,
            )

        if method == "gaussian":
            return self.gaussian(
                df=df,
                features=features,
                n_samples=n_samples,
                **kwargs,
            )

        if method == "empirical":
            return self.empirical(
                df=df,
                features=features,
                n_samples=n_samples,
                **kwargs,
            )

        raise ValueError(
            "Unknown synthetic generation "
            f"method: {method}"
        )


def generate_synthetic_data(
    df: Optional[pd.DataFrame] = None,
    features: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = None,
    method: str = "bootstrap",
    random_state: int = 42,
    reference: Optional[pd.DataFrame] = None,
    reference_df: Optional[pd.DataFrame] = None,
    n_rows: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> SyntheticGenerationResult:
    """
    Convenience wrapper for synthetic data generation.

    Supports both the original ``df`` API and the compatibility
    ``reference`` / ``reference_df`` API.
    """

    if df is None:
        if reference is not None:
            df = reference
        elif reference_df is not None:
            df = reference_df

    if df is None:
        raise ValueError(
            "A reference DataFrame must be provided via "
            "'df', 'reference', or 'reference_df'."
        )

    if n_samples is None:
        if n_rows is not None:
            n_samples = n_rows
        else:
            n_samples = len(df)

    if seed is not None:
        random_state = seed

    generator = SyntheticGenerator(
        random_state=random_state
    )

    return generator.generate(
        df=df,
        features=features,
        n_samples=n_samples,
        method=method,
        **kwargs,
    )


def distribution_summary(
    df: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
) -> pd.DataFrame:
    """
    Return a compact statistical summary for quality comparisons.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if features is None:
        features = [
            column
            for column in df.columns
            if pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

    if not features:
        raise ValueError(
            "No numeric features available."
        )

    summary = []

    for feature in features:
        if feature not in df.columns:
            raise ValueError(
                f"Missing feature: {feature}"
            )

        series = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).dropna()

        if series.empty:
            continue

        summary.append(
            {
                "feature": feature,
                "count": int(
                    series.count()
                ),
                "mean": float(
                    series.mean()
                ),
                "std": float(
                    series.std()
                ),
                "min": float(
                    series.min()
                ),
                "q25": float(
                    series.quantile(0.25)
                ),
                "median": float(
                    series.median()
                ),
                "q75": float(
                    series.quantile(0.75)
                ),
                "max": float(
                    series.max()
                ),
            }
        )

    return pd.DataFrame(
        summary
    )


def compare_distributions(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
) -> pd.DataFrame:
    """
    Compare reference and synthetic distributions using
    interpretable summary statistics.
    """

    if features is None:
        features = [
            column
            for column in reference.columns
            if column in synthetic.columns
            and pd.api.types.is_numeric_dtype(
                reference[column]
            )
        ]

    if not features:
        raise ValueError(
            "No common numeric features found."
        )

    rows = []

    for feature in features:
        ref = pd.to_numeric(
            reference[feature],
            errors="coerce",
        ).dropna()

        syn = pd.to_numeric(
            synthetic[feature],
            errors="coerce",
        ).dropna()

        if ref.empty or syn.empty:
            continue

        reference_mean = float(
            ref.mean()
        )

        synthetic_mean = float(
            syn.mean()
        )

        reference_std = float(
            ref.std()
        )

        synthetic_std = float(
            syn.std()
        )

        mean_difference = (
            synthetic_mean
            - reference_mean
        )

        std_difference = (
            synthetic_std
            - reference_std
        )

        mean_relative_error = (
            abs(mean_difference)
            / max(
                abs(reference_mean),
                1e-12,
            )
        )

        std_relative_error = (
            abs(std_difference)
            / max(
                abs(reference_std),
                1e-12,
            )
        )

        rows.append(
            {
                "feature": feature,
                "reference_mean": (
                    reference_mean
                ),
                "synthetic_mean": (
                    synthetic_mean
                ),
                "mean_difference": (
                    mean_difference
                ),
                "mean_relative_error": (
                    mean_relative_error
                ),
                "reference_std": (
                    reference_std
                ),
                "synthetic_std": (
                    synthetic_std
                ),
                "std_difference": (
                    std_difference
                ),
                "std_relative_error": (
                    std_relative_error
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def synthetic_quality_score(
    reference: pd.DataFrame,
    synthetic: pd.DataFrame,
    features: Optional[
        Sequence[str]
    ] = None,
) -> float:
    """
    Calculate a simple distribution-preservation score.

    Score:
        1.0 = identical summary statistics
        0.0 = poor agreement

    The score is based on normalized differences in means
    and standard deviations.
    """

    comparison = compare_distributions(
        reference=reference,
        synthetic=synthetic,
        features=features,
    )

    if comparison.empty:
        return 0.0

    mean_error = float(
        comparison[
            "mean_relative_error"
        ].mean()
    )

    std_error = float(
        comparison[
            "std_relative_error"
        ].mean()
    )

    error = (
        0.5 * mean_error
        + 0.5 * std_error
    )

    score = 1.0 / (
        1.0 + error
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )