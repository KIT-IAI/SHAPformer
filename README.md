# SHAPformer: Explainable time-series forecasting with sampling-free SHAP for Transformers

This repository contains the code accompanying the [Nature Communications publication](https://www.nature.com/articles/s41467-026-73243-5) on a new algorithm for estimating Shapley additive explanations (SHAP) for time-series forecasting.
It introduces an explainable model, called SHAPformer, that relies on attention manipulation to evaluate feature subsets without the need for sampling from background data.

> Matthias Hertel, Sebastian Pütz, Ralf Mikut, Veit Hagenmeyer, Benjamin Schäfer.
> *Explainable time-series forecasting with sampling-free SHAP for Transformers.*
> Nature Communications (2026). DOI: https://doi.org/10.1038/s41467-026-73243-5

A recording of a presentation of SHAPformer at the Helmholtz AI Conference 2025 is available via [YouTube](https://www.youtube.com/watch?v=Cfhcffm9DD0&list=PL_Qg4h55JDFui2bUKprmOiHtPNHLD_NuU&index=3) (duration: 11 minutes).

## Installation

SHAPformer requires Python 3.11 or newer and can be installed with pip:

```bash
pip install shapformer
```

Optional dependencies are available as extras:

| Extra | Adds |
|---|---|
| `pip install "shapformer[plot]"` | plots of explanations (matplotlib) |
| `pip install "shapformer[examples]"` | plots and everything the example notebooks need (Jupyter, fev, pyarrow, workalendar) |
| `pip install "shapformer[test]"` | the test suite (pytest) |

The quotes are needed in some shells, such as zsh.

### GPU support

SHAPformer is based on PyTorch and runs on NVIDIA GPUs with CUDA.
On Linux, the PyTorch version installed from PyPI already supports CUDA.
On Windows, PyPI only has a CPU version of PyTorch, so install a CUDA version of PyTorch first ([other CUDA versions](https://pytorch.org/get-started/locally/)):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install shapformer
```

To use the GPU, move the model to it; training, forecasting and explaining then run on the GPU:

```python
model = SHAPformer(...).to("cuda")
```

## Usage

```python
from shapformer import SHAPformer, plot_explanation

# df: a dataframe with a time index, the target and the covariates;
# categorical covariates have the number of their categories (values 0 to n - 1), continuous ones None
model = SHAPformer(
    target="load",
    feature_categories={"hour_of_day": 24, "day_of_week": 7, "temperature": None},
    d_model=128,
    n_layers=1,
    n_heads=4,
)
model.fit(df, validation_start="2019-01-01")

# past: the last 168 time steps with the target and the covariates; future: the next 168 time steps with the covariates
prediction = model.predict(past, future)
explanation = model.explain(past, future)  # prediction = base + sum of the SHAP values
plot_explanation(explanation)  # needs shapformer[plot]
```

See the [example notebooks](#demos) for complete examples.

## Development

To work on the code, clone the repository and install it in editable mode with all extras:

```bash
git clone https://github.com/KIT-IAI/SHAPformer.git
cd SHAPformer
pip install -e ".[examples,test]"
pytest
```

The notebooks then use the code of the repository instead of a release; their first cell prints the version and the location of the package they use.

## Datasets

The datasets used in the paper can be downloaded via this link: [data.zip](https://bwsyncandshare.kit.edu/s/22JeaRwgGTqxNyS). It contains:

- the synthetic dataset with ground truth explanations,
- the real-world dataset with the electrical load of TransnetBW (original source: [OPSD](https://data.open-power-system-data.org/time_series/2020-10-06)) and weather data (original source: [Copernicus](https://cds.climate.copernicus.eu/datasets/sis-energy-derived-reanalysis)).

The notebooks expect these datasets in the folder `data/` of the repository (`data/synthetic/` and `data/transnet/`).
The example notebooks use further public datasets, which they download automatically:

- EPF-DE (German electricity prices) and solar_with_weather_1H (solar generation) from the [fev-bench](https://github.com/autogluon/fev) benchmark, loaded with the `fev` package,
- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (hourly bike rentals in Washington, D.C.).

## Demos

| Notebook | Content |
|---|---|
| [ground_truth_explanations.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/ground_truth_explanations.ipynb) | Usage of the synthetic dataset with ground truth explanations |
| [training_json.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/training_json.ipynb) | Training of `ExogenousTransformer` on the TransnetBW data in JSON format (runtime on GPU: <5 minutes) |
| [evaluation_json.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/evaluation_json.ipynb) | Predictions and explanations with the trained `ExogenousTransformer` (runtime on GPU: <5 minutes) |
| [training_dataframe.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/training_dataframe.ipynb) | Dataframe interface on the TransnetBW data: training, forecasting, explaining and comparing the two explanation methods |
| [adapt_context_and_horizon_example.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/adapt_context_and_horizon_example.ipynb) | TransnetBW load forecasting with a two-week input and a two-day horizon (`input_length`, `horizon`) |
| [uneven_periods_example.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/uneven_periods_example.ipynb) | TransnetBW load forecasting with four weeks of input in periods of different lengths (`period_length=(336, 168, 144, 24)`) |
| [quarter_hourly_resolution_example.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/quarter_hourly_resolution_example.ipynb) | TransnetBW load forecasting on quarter-hourly data, one week of input in days (`period_length=96`) |
| [no_periods_example.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/no_periods_example.ipynb) | TransnetBW load forecasting with the past load as a single feature (`period_length=input_length`) |
| [electricity_price_forecasting_example.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/electricity_price_forecasting_example.ipynb) | Electricity price forecasting on EPF-DE with load and PV/wind forecasts |
| [solar_energy_forecasting_example.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/solar_energy_forecasting_example.ipynb) | Solar generation forecasting with weather covariates and gaps in the data |
| [bike_sharing_forecasting_example.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/bike_sharing_forecasting_example.ipynb) | Bike rental forecasting with monthly retraining and instance normalization |

The example notebooks evaluate the forecasts against naive baselines, explain one forecast and show dependence plots of 1000 explanations.
They save trained models in the folder `models/` of the repository.

## Changes since version 0.1.0

- **Dataframe interface:** the new class `SHAPformer` ([shapformer.py](https://github.com/KIT-IAI/SHAPformer/blob/main/src/shapformer/models/shapformer.py)) is built on `ExogenousTransformer`, the model of version 0.1.0 ([exogenous_transformer.py](https://github.com/KIT-IAI/SHAPformer/blob/main/src/shapformer/models/exogenous_transformer.py)), and trains (`fit`), forecasts (`predict`) and explains (`explain`) directly on pandas dataframes, with early stopping on a validation set. `ExogenousTransformer` is still available for preprocessed samples in JSON format (see [training_json.ipynb](https://github.com/KIT-IAI/SHAPformer/blob/main/notebooks/training_json.ipynb)).
- **Incomplete data:** `fit` accepts missing values and missing time steps and only uses complete windows.
- **Window lengths:** `SHAPformer(input_length=..., horizon=...)` sets the number of past and forecasted time steps (default: 168 each).
- **Periods of the past target:** `SHAPformer(period_length=...)` sets the groups ("periods") in which the past target is explained (default: 24 time steps, one period per day of hourly data). It takes an int for periods of equal length, or a sequence of lengths from the oldest to the most recent period, e.g. `(336, 168, 144, 24)` for four weeks of hourly data. With a single period (`period_length=input_length`), the past target is masked as a whole, like a covariate, without masking time steps. The explanation columns are named `period 1` (the oldest) to `period n`, or after the target for a single period.
- **Training on fewer windows:** `fit(window_stride=...)` trains and validates on every n-th window only, e.g. on windows that start at full hours of quarter-hourly data.
- **Instance normalization:** optional per-sample normalization of the target (`instance_norm="mean"` or `"RevIN"`).
- **Approximate explanations:** `explain(method="permutation")` estimates the Owen values from sampled permutations, much faster than the exact method (`method="exact"`, default).
- **Plots:** [plot.py](https://github.com/KIT-IAI/SHAPformer/blob/main/src/shapformer/plot.py) plots single explanations (`plot_explanation`) and dependence plots over many explanations (`plot_dependence`).
- **Example notebooks:** dataframe interface on TransnetBW, EPF-DE, solar and bike sharing data (see [Demos](#demos)).
- **Python package:** SHAPformer can be installed with `pip install shapformer` (version 1.0.0). The plots are optional (`shapformer[plot]`), and a test suite checks the explanations, the masking and the training.
