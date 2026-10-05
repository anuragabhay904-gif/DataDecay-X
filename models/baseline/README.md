# Baseline Models

This directory contains the baseline model definitions and related documentation for DataDecay-X.

The baseline models provide a simple reference point for evaluating model degradation under different drift scenarios.

## Purpose

Baseline models are used to:

- establish initial model performance;
- provide a reference for candidate models;
- compare performance before and after drift;
- separate data-drift effects from model-performance changes;
- support recovery and reliability experiments.

## Baseline Types

The project supports simple baseline approaches, including:

- Mean-based regression baseline
- Logistic regression classification baseline

## Evaluation

Baseline models should be evaluated using the metrics defined by the experiment configuration.

For regression:

- MAE
- MSE
- RMSE
- R²

For classification:

- Accuracy
- Weighted F1

## Reproducibility

Experiments should use the project-wide random seed where randomness is involved.

Baseline results should be saved under:

```text
experiments/results/