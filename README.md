# SHAPformer: Explainable time-series forecasting with sampling-free SHAP for Transformers

This repository contains the code accompanying the [Nature Communications publication](https://www.nature.com/articles/s41467-026-73243-5) on a new algorithm for estimating Shapley additive explanations (SHAP) for time-series forecasting.
It introduces an explainable model, called SHAPformer, that relies on attention manipulation to evaluate feature subsets without the need for sampling from background data.

> Matthias Hertel, Sebastian Pütz, Ralf Mikut, Veit Hagenmeyer, Benjamin Schäfer.
> *Explainable time-series forecasting with sampling-free SHAP for Transformers.*
> Nature Communications (2026). DOI: https://doi.org/10.1038/s41467-026-73243-5

A recording of a presentation of SHAPformer at the Helmholtz AI Conference 2025 is available via [YouTube](https://www.youtube.com/watch?v=Cfhcffm9DD0&list=PL_Qg4h55JDFui2bUKprmOiHtPNHLD_NuU&index=3) (duration: 11 minutes).

## Documentation

This repository demonstrates the usage of the synthetic data with ground truth explanations, the model training and how to create an explanation with SHAPformer.

### System requirements

The requirements are listed in the file `requirements.txt`.

To create a fresh virtual environment and install the dependencies in it, execute the following lines:
```
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

The expected installation duration is a few minutes.

Tested with Python 3.11 and 3.12.

### Datasets

You can download the datasets used in the paper via this link: [data.zip](https://bwsyncandshare.kit.edu/s/22JeaRwgGTqxNyS)

The zip file contains both datasets:
- The synthetic dataset with ground truth explanations.
- The real-world dataset containing the electrical load of TransnetBW (original source: [OPSD](https://data.open-power-system-data.org/time_series/2020-10-06)) and weather data (original source: [Copernicus](https://cds.climate.copernicus.eu/datasets/sis-energy-derived-reanalysis)).

For a demonstration how to use the synthetic dataset with ground truth, refer to this notebook: [ground_truth_explanations.ipynb](notebooks/ground_truth_explanations.ipynb)

### Demos

The notebook [training.ipynb](notebooks/training.ipynb) demonstrates how to train the SHAPformer model (expected runtime on GPU: <5 minutes).

The notebook [evaluate.ipynb](notebooks/evaluation.ipynb) demonstrates how to use the trained SHAPformer model to generate predictions and explanations (expected runtime on GPU: <5 minutes).

## Changes since version 0.1.0

### Dataframe interface

The new class `SHAPformer` ([src/models/shapformer.py](src/models/shapformer.py)) trains, forecasts and explains directly on pandas dataframes, without preparing JSON files with samples first.
A model is defined by its target column and its covariates, each with its number of categories (or `None` for continuous features):

```python
model = SHAPformer(
    target="load",
    feature_categories={"hour_of_day": 24, "day_of_week": 7, "month": 12, "holiday": 2,
                        "temperature": None, "precipitation": None},
    d_model=128, n_layers=1, n_heads=4
).to("cuda")

model = model.fit(df, validation_start="2019-01-01")
prediction = model.predict(past, future)
explanation = model.explain(past, future)
```

- `instance_norm` (constructor argument, default `"none"`) optionally normalizes the target per sample with the statistics of its past window: `"mean"` divides by the mean absolute value (as in Chronos), `"RevIN"` subtracts the mean, divides by the standard deviation and applies a learnable affine transformation (reversible instance normalization). The statistics are computed from the full past window for every mask, so explanations stay additive; the effect of the level and scale of the past target is then part of the base value and not attributed to the past days.
- `fit(data, validation_start, epochs=100, batch_size=128, learning_rate=0.0001, p_dropout=0.5, patience=10, min_improvement=0.00001)` trains on all rows before `validation_start` and validates on the rest:
  - fits a standard scaler for each continuous variable on the training part (except for the target with instance normalization, which is normalized per sample instead),
  - trains in batches, masking each sample with a random feature group subset,
  - prints the training and validation loss after each epoch,
  - stops early when the validation loss has not improved by more than `min_improvement` for `patience` epochs, restores the best model and returns it,
  - accepts incomplete data: rows with missing values (NaN) and missing time steps are allowed. Training and validation only use windows of 336 consecutive time steps (input and horizon) without missing values; the time step is the most frequent difference between consecutive timestamps. `fit` prints how many windows it skips, and raises an error if no window is left.
- `predict(past, future)` forecasts the target for the 168 hours in `future` (covariates only) from the 168 hours in `past` (target and covariates). The forecast is in the unit of the target.
- `explain(past, future)` returns the prediction, the base value and the SHAP (Owen) values of all feature groups (one per covariate, plus one per day of the past target), all in the unit of the target, together with the inputs. Base value plus all SHAP values equals the prediction.
- Categorical features must be integers starting at 0; `fit` raises an error otherwise.

### Two explanation methods

The argument `method` of `explain()` selects how the SHAP values are computed:

- `method="exact"` (default) evaluates the model with all feature group subsets and computes the exact Owen values.
  Its cost doubles with each additional feature group.
- `method="permutation"` approximates the Owen values by sampling `n_permutations` random orders of the feature groups (keeping the days of the past target together), similar to the permutation explainer of the shap package.
  Each permutation needs 2 × (number of feature groups) + 1 model evaluations, and `seed` makes the result reproducible.
  The values are estimates whose error decreases with more permutations; base value plus all SHAP values still equals the prediction exactly.

With `progress=False`, `explain()` shows no progress bar over its model evaluations, e.g. to show one progress bar over many explanations instead.

With 13 feature groups on an RTX 3070 Laptop GPU, an exact explanation takes about 2 s, and a permutation explanation with 10 permutations about 0.07 s (with a mean absolute error of about 30% of the mean absolute SHAP value; about 5% with 200 permutations, which take about 1.3 s).

### Faster exact explanations

`explain_sample` in [src/shap.py](src/shap.py) is about four times faster than in version 0.1.0 (about 2 s instead of 8 s per explanation in the setting above), with the same results up to floating-point precision:

- The masks of all feature group subsets are built once with vectorized tensor operations and cached, instead of being created anew for every explanation.
- The SHAP values are computed as one matrix product of the model outputs with a cached coefficient matrix, instead of a Python loop over all subsets and feature groups.

The masks for training are also built with vectorized tensor operations, for a whole batch at once.

### Plots

[src/plot.py](src/plot.py) contains two plot functions for explanations from `explain()`:

- `plot_explanation(explanation)` shows an explanation as a stacked area chart around the base value, optionally with the actual values. The past days can be summarized into one group (`mode="load"`), or all other features into one group (`mode="days"`).
- `plot_dependence(explanations, feature)` shows, for a collection of explanations, the SHAP values of a feature against its values at one forecasted hour (`horizon`, default: the 24th hour, `None` for all hours), optionally colored by a second feature (`color_by`).

### Notebooks

The notebook [training_dataframe.ipynb](notebooks/training_dataframe.ipynb) demonstrates the dataframe interface on the TransnetBW data: training (or loading a trained model with `TRAINING = False`), forecasting and explaining one week, and dependence plots of 100 and 1000 explanations that compare the two explanation methods and their runtimes.

The notebook [electricity_price_forecasting_example.ipynb](notebooks/electricity_price_forecasting_example.ipynb) applies the dataframe interface to electricity price forecasting on the EPF-DE dataset of the [fev-bench](https://github.com/autogluon/fev) benchmark, loaded with the `fev` package: it adds calendar features to the covariates of the dataset (load forecast, PV and wind generation forecast), trains on 2012–2016, validates on the first and tests on the second half of 2017, compares the forecasts with a weekly naive baseline, explains the forecast of the week of 1 May 2017 and shows dependence plots of 1000 explanations.

The notebook [solar_energy_forecasting_example.ipynb](notebooks/solar_energy_forecasting_example.ipynb) forecasts hourly solar generation on the solar_with_weather_1H dataset of fev-bench, loaded with the `fev` package: it uses all weather covariates of the dataset and calendar features, treating cloud cover and irradiance (past covariates in fev-bench) as future covariates, i.e. assuming perfect forecasts of them. It inspects the gaps in the data (missing days, which `fit` skips), trains on January 2017 – August 2020, validates and tests on one year each (September 2020 – August 2022), compares the forecasts with daily and weekly naive baselines, explains the forecast of the week of 13 June 2022 and shows dependence plots of 1000 explanations.

The notebook [bike_sharing_forecasting_example.ipynb](notebooks/bike_sharing_forecasting_example.ipynb) forecasts hourly bike rentals on the [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) dataset (Washington, D.C., 2011–2012), which it downloads once to `data/bike_sharing/`, with calendar and weather covariates. Since the number of rentals grew strongly from 2011 to 2012, it retrains a model with `instance_norm="RevIN"` at the beginning of every month of 2012 on all earlier data, validated on the previous month. It evaluates the forecasts of 2012 per month and for the validation (January–June) and test (July–December) period, explains the forecast of the week of 2 July 2012 (with Independence Day) and shows dependence plots of 1000 explanations.
