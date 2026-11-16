# SHAPformer: Explainable time-series forecasting with sampling-free SHAP for Transformers

This repository contains the code accompanying the [Nature Communications publication](https://www.nature.com/articles/s41467-026-73243-5) on a new algorithm for estimating Shapley additive explanations (SHAP) for time-series forecasting.
It introduces an explainable model, called SHAPformer, that relies on attention manipulation to evaluate feature subsets without the need for sampling from background data.

> Matthias Hertel, Sebastian Pütz, Ralf Mikut, Veit Hagenmeyer, Benjamin Schäfer.
> *Explainable time-series forecasting with sampling-free SHAP for Transformers.*
> Nature Communications (2026). DOI: https://doi.org/10.1038/s41467-026-73243-5

A recording of a presentation of SHAPformer at the Helmholtz AI Conference 2025 is available via [YouTube](https://www.youtube.com/watch?v=Cfhcffm9DD0&list=PL_Qg4h55JDFui2bUKprmOiHtPNHLD_NuU&index=3) (duration: 11 minutes).

## System requirements

The requirements are listed in the file `requirements.txt`.
To create a fresh virtual environment and install the dependencies in it, execute the following lines:

```
python -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

The expected installation duration is a few minutes. Tested with Python 3.11 and 3.12.

## Datasets

The datasets used in the paper can be downloaded via this link: [data.zip](https://bwsyncandshare.kit.edu/s/22JeaRwgGTqxNyS). It contains:

- the synthetic dataset with ground truth explanations,
- the real-world dataset with the electrical load of TransnetBW (original source: [OPSD](https://data.open-power-system-data.org/time_series/2020-10-06)) and weather data (original source: [Copernicus](https://cds.climate.copernicus.eu/datasets/sis-energy-derived-reanalysis)).

The example notebooks use further public datasets, which they download automatically:

- EPF-DE (German electricity prices) and solar_with_weather_1H (solar generation) from the [fev-bench](https://github.com/autogluon/fev) benchmark, loaded with the `fev` package,
- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (hourly bike rentals in Washington, D.C.).

## Demos

| Notebook | Content |
|---|---|
| [ground_truth_explanations.ipynb](notebooks/ground_truth_explanations.ipynb) | Usage of the synthetic dataset with ground truth explanations |
| [training.ipynb](notebooks/training.ipynb) | Training of SHAPformer on the TransnetBW data (runtime on GPU: <5 minutes) |
| [evaluation.ipynb](notebooks/evaluation.ipynb) | Predictions and explanations with the trained model (runtime on GPU: <5 minutes) |
| [training_dataframe.ipynb](notebooks/training_dataframe.ipynb) | Dataframe interface on the TransnetBW data: training, forecasting, explaining and comparing the two explanation methods |
| [adapt_context_and_horizon_example.ipynb](notebooks/adapt_context_and_horizon_example.ipynb) | TransnetBW load forecasting with a two-week input and a two-day horizon (`input_length`, `horizon`) |
| [uneven_periods_example.ipynb](notebooks/uneven_periods_example.ipynb) | TransnetBW load forecasting with four weeks of input in periods of different lengths (`period_length=(336, 168, 144, 24)`) |
| [quarter_hourly_resolution_example.ipynb](notebooks/quarter_hourly_resolution_example.ipynb) | TransnetBW load forecasting on quarter-hourly data, one week of input in days (`period_length=96`) |
| [no_periods_example.ipynb](notebooks/no_periods_example.ipynb) | TransnetBW load forecasting with the past load as a single feature (`period_length=input_length`) |
| [electricity_price_forecasting_example.ipynb](notebooks/electricity_price_forecasting_example.ipynb) | Electricity price forecasting on EPF-DE with load and PV/wind forecasts |
| [solar_energy_forecasting_example.ipynb](notebooks/solar_energy_forecasting_example.ipynb) | Solar generation forecasting with weather covariates and gaps in the data |
| [bike_sharing_forecasting_example.ipynb](notebooks/bike_sharing_forecasting_example.ipynb) | Bike rental forecasting with monthly retraining and instance normalization |
The example notebooks evaluate the forecasts against naive baselines, explain one forecast and show dependence plots of 1000 explanations.

## Changes since version 0.1.0

- **Dataframe interface:** the class `SHAPformer` ([src/models/shapformer.py](src/models/shapformer.py)) trains (`fit`), forecasts (`predict`) and explains (`explain`) directly on pandas dataframes, with early stopping on a validation set.
- **Incomplete data:** `fit` accepts missing values and missing time steps and only uses complete windows.
- **Window lengths:** `SHAPformer(input_length=..., horizon=...)` sets the number of past and forecasted time steps (default: 168 each).
- **Periods of the past target:** `SHAPformer(period_length=...)` sets the groups ("periods") in which the past target is explained (default: 24 time steps, one period per day of hourly data). It takes an int for periods of equal length, or a sequence of lengths from the oldest to the most recent period, e.g. `(336, 168, 144, 24)` for four weeks of hourly data. With a single period (`period_length=input_length`), the past target is masked as a whole, like a covariate, without masking time steps. The explanation columns are named `period 1` (the oldest) to `period n`, or after the target for a single period.
- **Training on fewer windows:** `fit(window_stride=...)` trains and validates on every n-th window only, e.g. on windows that start at full hours of quarter-hourly data.
- **Instance normalization:** optional per-sample normalization of the target (`instance_norm="mean"` or `"RevIN"`).
- **Approximate explanations:** `explain(method="permutation")` estimates the Owen values from sampled permutations, much faster than the exact method (`method="exact"`, default).
- **Plots:** [src/plot.py](src/plot.py) plots single explanations (`plot_explanation`) and dependence plots over many explanations (`plot_dependence`).
- **Example notebooks:** dataframe interface on TransnetBW, EPF-DE, solar and bike sharing data (see [Demos](#demos)).
