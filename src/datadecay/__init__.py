"""
DataDecay-X

Core package exports for data drift detection, model monitoring,
forecasting, synthetic data generation, quality gates, and recovery.
"""

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

from .utils import (
    ensure_dir,
    load_electricity_dataset,
    make_windows,
    numeric_features,
    save_json,
    temporal_split,
    validate_temporal_order,
)

# ---------------------------------------------------------
# Drift simulation
# ---------------------------------------------------------

from .simulator import (
    DriftMetadata,
    DriftSimulator,
)

# ---------------------------------------------------------
# Drift detection
# ---------------------------------------------------------

from .detection import (
    calculate_dsi,
    calculate_feature_metrics,
    calculate_ks,
    calculate_psi,
    calculate_wasserstein,
    classify_dsi,
)

# ---------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------

from .monitoring import (
    monitor_window,
    summarize_monitoring,
)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

from .models import (
    BaselineModel,
    ModelResult,
    classification_metrics,
    compare_model_results,
    evaluate_model,
    evaluate_predictions,
    regression_metrics,
    temporal_model_evaluation,
)

# ---------------------------------------------------------
# Forecasting
# ---------------------------------------------------------

from .forecasting import (
    ForecastResult,
    LinearTrendForecaster,
    PersistenceForecaster,
    calculate_forecast_metrics,
    compare_forecasters,
    evaluate_forecast,
    forecast_dataframe,
    forecast_multiple_features,
    forecast_series,
    rolling_forecast,
)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

from .evaluation import (
    EvaluationResult,
    classification_accuracy,
    classification_f1,
    evaluate_classification,
    evaluate_regression,
    evaluate_predictions as evaluate_prediction_metrics,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r_squared,
    results_to_dataframe,
    root_mean_squared_error,
    temporal_evaluation,
)

# ---------------------------------------------------------
# Reliability
# ---------------------------------------------------------

from .reliability import (
    ReliabilityResult,
    calculate_degradation,
    calculate_degradation_ratio,
    calculate_reliability_index,
    classify_reliability,
    compare_reliability_by_period,
    evaluate_multiple_models,
    evaluate_reliability,
    reliability_summary,
)

# ---------------------------------------------------------
# Model monitoring
# ---------------------------------------------------------

from .monitoring_models import (
    MonitoringResult,
    build_monitoring_result,
    compare_model_monitoring,
    monitor_model,
    monitor_model_performance,
    monitor_model_reliability,
    summarize_model_monitoring,
)

# ---------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------

from .synthetic import (
    SyntheticGenerationResult,
    SyntheticGenerator,
    compare_distributions,
    distribution_summary,
    generate_synthetic_data,
    synthetic_quality_score,
)

# ---------------------------------------------------------
# Synthetic-data quality gate
# ---------------------------------------------------------

from .quality_gate import (
    QualityGateResult,
    calculate_quality_score,
    check_correlation,
    check_distribution,
    check_missingness,
    check_schema,
    check_value_bounds,
    quality_gate_summary,
    run_quality_gate,
    save_quality_gate_report,
)

# ---------------------------------------------------------
# Recovery
# ---------------------------------------------------------

from .recovery import (
    RecoveryResult,
    backward_fill_recovery,
    compare_recovery_methods,
    create_missingness,
    forward_fill_recovery,
    interpolate_recovery,
    mean_recovery,
    median_recovery,
    recover_missing_values,
    recovery_quality,
    recovery_summary,
    synthetic_recovery,
)


__all__ = [
    # Utilities
    "ensure_dir",
    "load_electricity_dataset",
    "make_windows",
    "numeric_features",
    "save_json",
    "temporal_split",
    "validate_temporal_order",

    # Drift simulation
    "DriftMetadata",
    "DriftSimulator",

    # Drift detection
    "calculate_dsi",
    "calculate_feature_metrics",
    "calculate_ks",
    "calculate_psi",
    "calculate_wasserstein",
    "classify_dsi",

    # Drift monitoring
    "monitor_window",
    "summarize_monitoring",

    # Models
    "BaselineModel",
    "ModelResult",
    "classification_metrics",
    "compare_model_results",
    "evaluate_model",
    "evaluate_predictions",
    "regression_metrics",
    "temporal_model_evaluation",

    # Forecasting
    "ForecastResult",
    "LinearTrendForecaster",
    "PersistenceForecaster",
    "calculate_forecast_metrics",
    "compare_forecasters",
    "evaluate_forecast",
    "forecast_dataframe",
    "forecast_multiple_features",
    "forecast_series",
    "rolling_forecast",

    # Evaluation
    "EvaluationResult",
    "classification_accuracy",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_prediction_metrics",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "r_squared",
    "results_to_dataframe",
    "root_mean_squared_error",
    "temporal_evaluation",
    "weighted_f1",

    # Reliability
    "ReliabilityResult",
    "calculate_degradation",
    "calculate_degradation_ratio",
    "calculate_reliability_index",
    "classify_reliability",
    "compare_reliability_by_period",
    "evaluate_multiple_models",
    "evaluate_reliability",
    "reliability_summary",

    # Model monitoring
    "MonitoringResult",
    "build_monitoring_result",
    "compare_model_monitoring",
    "monitor_model",
    "monitor_model_performance",
    "monitor_model_reliability",
    "summarize_model_monitoring",

    # Synthetic data
    "SyntheticGenerationResult",
    "SyntheticGenerator",
    "compare_distributions",
    "distribution_summary",
    "generate_synthetic_data",
    "synthetic_quality_score",

    # Quality gate
    "QualityGateResult",
    "calculate_quality_score",
    "check_correlation",
    "check_distribution",
    "check_missingness",
    "check_schema",
    "check_value_bounds",
    "quality_gate_summary",
    "run_quality_gate",
    "save_quality_gate_report",

    # Recovery
    "RecoveryResult",
    "backward_fill_recovery",
    "compare_recovery_methods",
    "create_missingness",
    "forward_fill_recovery",
    "interpolate_recovery",
    "mean_recovery",
    "median_recovery",
    "recover_missing_values",
    "recovery_quality",
    "recovery_summary",
    "synthetic_recovery",
]