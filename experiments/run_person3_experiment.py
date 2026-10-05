from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import numpy as np
import pandas as pd

from src.datadecay.quality_gate import (
    quality_gate_summary,
    run_quality_gate,
)

from src.datadecay.recovery import (
    compare_recovery_methods,
    create_missingness,
    recovery_quality,
    recover_missing_values,
    recovery_summary,
)

from src.datadecay.synthetic import (
    compare_distributions,
    generate_synthetic_data,
    synthetic_quality_score,
)

from src.datadecay.utils import (
    ensure_dir,
    load_electricity_dataset,
    save_json,
    temporal_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LD2011_2014.txt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "person3"
)

SYNTHETIC_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
)

ACCEPTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "accepted"
)


def select_features(
    reference: pd.DataFrame,
    n_features: int = 5,
):
    """
    Select high-variance numerical electricity series.
    """

    numeric_columns = [
        column
        for column in reference.columns
        if column != "timestamp"
        and pd.api.types.is_numeric_dtype(
            reference[column]
        )
    ]

    variances = (
        reference[numeric_columns]
        .var()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .sort_values(
            ascending=False
        )
    )

    features = list(
        variances.head(
            n_features
        ).index
    )

    if not features:
        raise ValueError(
            "No numeric features were available."
        )

    return features


def main():
    print("=" * 70)
    print("PERSON 3 - SYNTHETIC + RECOVERY EXPERIMENT")
    print("=" * 70)

    ensure_dir(RESULTS_DIR)
    ensure_dir(SYNTHETIC_DIR)
    ensure_dir(ACCEPTED_DIR)

    print("\nLoading dataset...")

    df = load_electricity_dataset(
        DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    reference, monitoring, evaluation = (
        temporal_split(
            df,
            train_fraction=0.60,
            monitor_fraction=0.20,
        )
    )

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

    features = select_features(
        reference,
        n_features=5,
    )

    print(
        "\nSelected features:"
    )

    for feature in features:
        print(
            f"  - {feature}"
        )

    # ----------------------------------------------------------
    # 1. Synthetic data generation
    # ----------------------------------------------------------

    print(
        "\nGenerating synthetic data..."
    )

    synthetic_result = (
        generate_synthetic_data(
            df=reference,
            features=features,
            n_samples=len(monitoring),
            method="bootstrap",
            random_state=42,
            noise_scale=0.01,
        )
    )

    synthetic = (
        synthetic_result.samples
        .copy()
    )

    synthetic_path = (
        SYNTHETIC_DIR
        / "person3_bootstrap.csv"
    )

    synthetic.to_csv(
        synthetic_path,
        index=False,
    )

    print(
        f"Synthetic data saved to: "
        f"{synthetic_path}"
    )

    # ----------------------------------------------------------
    # 2. Synthetic quality gate
    # ----------------------------------------------------------

    print(
        "\nRunning synthetic quality gate..."
    )

    quality_result = run_quality_gate(
        reference=reference,
        candidate=synthetic,
        features=features,
        threshold=0.70,
        max_missing_fraction=0.01,
        psi_threshold=0.25,
        wasserstein_threshold=1.0,
        correlation_difference=0.20,
    )

    print(
        "Quality accepted:",
        quality_result.accepted,
    )

    print(
        "Quality score:",
        round(
            quality_result.quality_score,
            4,
        ),
    )

    quality_summary = (
        quality_gate_summary(
            quality_result
        )
    )

    quality_json_path = (
        RESULTS_DIR
        / "quality_gate.json"
    )

    save_json(
        quality_summary,
        quality_json_path,
    )

    quality_metrics_path = (
        RESULTS_DIR
        / "synthetic_quality_metrics.csv"
    )

    quality_result.feature_metrics.to_csv(
        quality_metrics_path,
        index=False,
    )

    if quality_result.accepted:
        accepted_path = (
            ACCEPTED_DIR
            / "person3_accepted.csv"
        )

        synthetic.to_csv(
            accepted_path,
            index=False,
        )

        print(
            f"Accepted synthetic data saved to: "
            f"{accepted_path}"
        )

    # ----------------------------------------------------------
    # 3. Distribution comparison
    # ----------------------------------------------------------

    print(
        "\nComparing distributions..."
    )

    distribution_comparison = (
        compare_distributions(
            reference=reference,
            synthetic=synthetic,
            features=features,
        )
    )

    distribution_path = (
        RESULTS_DIR
        / "distribution_comparison.csv"
    )

    distribution_comparison.to_csv(
        distribution_path,
        index=False,
    )

    synthetic_score = (
        synthetic_quality_score(
            reference=reference,
            synthetic=synthetic,
            features=features,
        )
    )

    print(
        "Synthetic distribution score:",
        round(
            synthetic_score,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 4. Controlled missingness
    # ----------------------------------------------------------

    print(
        "\nCreating controlled missingness..."
    )

    clean_monitoring = (
        monitoring[
            ["timestamp"]
            + features
        ]
        .copy()
        .reset_index(drop=True)
    )

    corrupted_monitoring = (
        create_missingness(
            clean_monitoring,
            features=features,
            missing_fraction=0.05,
            random_state=42,
        )
    )

    corrupted_path = (
        RESULTS_DIR
        / "corrupted_monitoring.csv"
    )

    corrupted_monitoring.to_csv(
        corrupted_path,
        index=False,
    )

    original_missing = int(
        clean_monitoring[
            features
        ]
        .isna()
        .sum()
        .sum()
    )

    corrupted_missing = int(
        corrupted_monitoring[
            features
        ]
        .isna()
        .sum()
        .sum()
    )

    print(
        "Original missing cells:",
        original_missing,
    )

    print(
        "Corrupted missing cells:",
        corrupted_missing,
    )

    # ----------------------------------------------------------
    # 5. Recovery methods
    # ----------------------------------------------------------

    print(
        "\nRunning recovery methods..."
    )

    recovery_methods = [
        "forward_fill",
        "backward_fill",
        "interpolate",
        "median",
        "mean",
        "synthetic",
    ]

    recovery_comparison = (
        compare_recovery_methods(
            original=clean_monitoring,
            corrupted=corrupted_monitoring,
            methods=recovery_methods,
            features=features,
            synthetic_df=synthetic,
        )
    )

    recovery_comparison_path = (
        RESULTS_DIR
        / "recovery_comparison.csv"
    )

    recovery_comparison.to_csv(
        recovery_comparison_path,
        index=False,
    )

    print(
        recovery_comparison.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Select best recovery method
    # ----------------------------------------------------------

    valid_comparison = (
        recovery_comparison.dropna(
            subset=["rmse"]
        )
    )

    if valid_comparison.empty:
        best_method = "interpolate"
    else:
        best_method = str(
            valid_comparison.sort_values(
                "rmse"
            ).iloc[0]["method"]
        )

    print(
        "\nBest recovery method:",
        best_method,
    )

    best_recovery = (
        recover_missing_values(
            corrupted_monitoring,
            method=best_method,
            features=features,
            synthetic_df=synthetic,
        )
    )

    best_recovery_path = (
        RESULTS_DIR
        / "recovered_monitoring.csv"
    )

    best_recovery.recovered.to_csv(
        best_recovery_path,
        index=False,
    )

    recovery_quality_result = (
        recovery_quality(
            original=clean_monitoring,
            recovered=best_recovery.recovered,
            corrupted=corrupted_monitoring,
            features=features,
        )
    )

    # ----------------------------------------------------------
    # 7. Synthetic-assisted recovery
    # ----------------------------------------------------------

    synthetic_recovery = (
        recover_missing_values(
            corrupted_monitoring,
            method="synthetic",
            features=features,
            synthetic_df=synthetic,
        )
    )

    synthetic_recovery_quality = (
        recovery_quality(
            original=clean_monitoring,
            recovered=(
                synthetic_recovery.recovered
            ),
            corrupted=corrupted_monitoring,
            features=features,
        )
    )

    synthetic_recovery_path = (
        RESULTS_DIR
        / "synthetic_recovered_monitoring.csv"
    )

    synthetic_recovery.recovered.to_csv(
        synthetic_recovery_path,
        index=False,
    )

    # ----------------------------------------------------------
    # 8. Final metadata
    # ----------------------------------------------------------

    metadata = {
        "project": "DataDecay-X",
        "person": 3,
        "random_state": 42,
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "split": {
            "reference_rows": int(
                len(reference)
            ),
            "monitoring_rows": int(
                len(monitoring)
            ),
            "evaluation_rows": int(
                len(evaluation)
            ),
        },
        "features": features,
        "synthetic": {
            "method": (
                synthetic_result.method
            ),
            "rows": int(
                len(synthetic)
            ),
            "quality_score": float(
                synthetic_score
            ),
            "quality_gate": (
                quality_summary
            ),
        },
        "corruption": {
            "missing_fraction": 0.05,
            "missing_cells": int(
                corrupted_missing
            ),
        },
        "recovery": {
            "methods": recovery_methods,
            "best_method": best_method,
            "best_method_quality": (
                recovery_quality_result
            ),
            "synthetic_method_quality": (
                synthetic_recovery_quality
            ),
        },
    }

    metadata_path = (
        RESULTS_DIR
        / "person3_metadata.json"
    )

    save_json(
        metadata,
        metadata_path,
    )

    # ----------------------------------------------------------
    # 9. Compact summary
    # ----------------------------------------------------------

    summary = {
        "synthetic_quality_score": float(
            synthetic_score
        ),
        "quality_gate_accepted": bool(
            quality_result.accepted
        ),
        "quality_gate_score": float(
            quality_result.quality_score
        ),
        "corrupted_missing_cells": int(
            corrupted_missing
        ),
        "best_recovery_method": best_method,
        "best_recovery_mae": (
            recovery_quality_result["mae"]
        ),
        "best_recovery_rmse": (
            recovery_quality_result["rmse"]
        ),
        "synthetic_recovery_mae": (
            synthetic_recovery_quality[
                "mae"
            ]
        ),
        "synthetic_recovery_rmse": (
            synthetic_recovery_quality[
                "rmse"
            ]
        ),
    }

    summary_path = (
        RESULTS_DIR
        / "person3_summary.json"
    )

    save_json(
        summary,
        summary_path,
    )

    print(
        "\nPerson 3 experiment artifacts saved."
    )

    print(
        f"Results directory: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()