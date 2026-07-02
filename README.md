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
