"""
Drift plotting utilities for DataDecay-X.

The functions in this module create simple matplotlib figures
from drift-metric results. They do not alter the underlying data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent_directory(path):
    """Create the parent directory for an output path."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def plot_drift_metrics(
    feature_metrics,
    output_path=None,
    title="Drift Metrics by Feature",
):
    """
    Plot PSI, KS, and Wasserstein metrics by feature.

    Parameters
    ----------
    feature_metrics : dict
        Feature-level drift metric dictionary.

    output_path : str or Path, optional
        If supplied, save the figure to this location.

    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Created figure.
    """

    if not isinstance(
        feature_metrics,
        dict,
    ):
        raise TypeError(
            "feature_metrics must be a dictionary."
        )

    features = list(
        feature_metrics.keys()
    )

    psi = [
        float(
            feature_metrics[feature].get(
                "psi",
                0.0,
            )
        )
        for feature in features
    ]

    ks = [
        float(
            feature_metrics[feature].get(
                "ks",
                0.0,
            )
        )
        for feature in features
    ]

    wasserstein = [
        float(
            feature_metrics[feature].get(
                "wasserstein",
                0.0,
            )
        )
        for feature in features
    ]

    x = np.arange(
        len(features)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        x - width,
        psi,
        width=width,
        label="PSI",
    )

    ax.bar(
        x,
        ks,
        width=width,
        label="KS",
    )

    ax.bar(
        x + width,
        wasserstein,
        width=width,
        label="Wasserstein",
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Feature"
    )

    ax.set_ylabel(
        "Metric value"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        features,
        rotation=45,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    if output_path is not None:
        output_path = _ensure_parent_directory(
            output_path
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    return fig


def plot_dsi(
    dsi,
    severity=None,
    output_path=None,
    title="Data Shift Index",
):
    """
    Plot a single DSI value.

    Parameters
    ----------
    dsi : float
        Data Shift Index.

    severity : str, optional
        Severity label displayed in the figure.

    output_path : str or Path, optional
        Output image path.

    Returns
    -------
    matplotlib.figure.Figure
    """

    dsi = float(dsi)

    if not np.isfinite(dsi):
        raise ValueError(
            "DSI must be finite."
        )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        ["DSI"],
        [dsi],
    )

    ax.set_title(title)

    ax.set_ylabel(
        "Data Shift Index"
    )

    ax.set_ylim(
        0,
        max(
            1.0,
            dsi * 1.20,
        ),
    )

    if severity is not None:
        ax.text(
            0,
            dsi,
            f"Severity: {severity}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    if output_path is not None:
        output_path = _ensure_parent_directory(
            output_path
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    return fig


def plot_scenario_comparison(
    scenario_results,
    output_path=None,
    title="Drift Scenario Comparison",
):
    """
    Compare DSI values across drift scenarios.

    Parameters
    ----------
    scenario_results : dict
        Mapping of scenario names to either:

        {
            "dsi": value
        }

        or directly:

        {
            "stable": value,
            "gradual": value,
            ...
        }

    output_path : str or Path, optional
        Output image path.

    Returns
    -------
    matplotlib.figure.Figure
    """

    if not isinstance(
        scenario_results,
        dict,
    ):
        raise TypeError(
            "scenario_results must be a dictionary."
        )

    scenario_names = []
    dsi_values = []

    for scenario, result in scenario_results.items():

        if isinstance(
            result,
            dict,
        ):
            value = result.get(
                "dsi",
                0.0,
            )
        else:
            value = result

        scenario_names.append(
            str(scenario)
        )

        dsi_values.append(
            float(value)
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        scenario_names,
        dsi_values,
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Scenario"
    )

    ax.set_ylabel(
        "DSI"
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = _ensure_parent_directory(
            output_path
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    return fig


def plot_metric_timeseries(
    metric_values,
    output_path=None,
    title="Drift Metric Over Time",
    ylabel="Metric",
):
    """
    Plot a metric across chronological monitoring windows.

    Parameters
    ----------
    metric_values : sequence
        Metric value for each monitoring window.

    output_path : str or Path, optional
        Output image path.

    Returns
    -------
    matplotlib.figure.Figure
    """

    values = np.asarray(
        metric_values,
        dtype=float,
    )

    if values.ndim != 1:
        values = values.reshape(-1)

    if len(values) == 0:
        raise ValueError(
            "metric_values cannot be empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "metric_values contains NaN or infinite values."
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        np.arange(
            len(values)
        ),
        values,
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Monitoring window"
    )

    ax.set_ylabel(ylabel)

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    if output_path is not None:
        output_path = _ensure_parent_directory(
            output_path
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    return fig


def close_figure(fig):
    """
    Close a matplotlib figure safely.
    """

    if fig is not None:
        plt.close(fig)


__all__ = [
    "plot_drift_metrics",
    "plot_dsi",
    "plot_scenario_comparison",
    "plot_metric_timeseries",
    "close_figure",
]