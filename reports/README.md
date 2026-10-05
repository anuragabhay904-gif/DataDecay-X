# DataDecay-X Reports

This directory contains report-generation scripts and generated
visualizations for the DataDecay-X experiments.

## Report Scripts

### Person 1

`plot_person1_results.py`

Generates visualizations for:

- drift scenarios
- PSI
- KS statistic
- Wasserstein distance
- Data Shift Index (DSI)
- drift severity

### Person 2

`plot_person2_results.py`

Generates visualizations for:

- baseline model performance
- drifted model performance
- MAE
- RMSE
- R²
- reliability degradation
- model monitoring results

### Person 3

`plot_person3_results.py`

Generates visualizations for:

- synthetic-data generation
- synthetic-data quality
- quality-gate results
- recovery performance
- recovery improvement

## Output

Generated report figures should be stored under the appropriate
report/output directory and should not modify the raw dataset.

Typical report outputs include:

- drift summaries
- model-performance comparisons
- reliability comparisons
- synthetic-data quality comparisons
- recovery comparisons

## Reproducibility

Report scripts should use experiment outputs from:

```text
experiments/results/