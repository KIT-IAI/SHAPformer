# SHAPformer: Explainable time-series forecasting with sampling-free SHAP for Transformers

This repository contains the code accompanying our publication on a new algorithm for estimating Shapley additive explanations (SHAP) for time-series forecasting.
It introduces an explainable model, called SHAPformer, that relies on attention manipulation to evaluate feature subsets without the need for sampling from background data.

Stay tuned for the upcoming publication:

> Matthias Hertel, Sebastian Pütz, Ralf Mikut, Veit Hagenmeyer, Benjamin Schäfer.
> *Explainable time-series forecasting with sampling-free SHAP for Transformers.*
> In Preparation (2025).

A talk about SHAPformer was given at the Helmholtz AI Conference 2025.
The recording is available via [YouTube](https://www.youtube.com/watch?v=Cfhcffm9DD0&list=PL_Qg4h55JDFui2bUKprmOiHtPNHLD_NuU&index=3) (duration: 11 minutes).

## Documentation

This repository demonstrates the usage of the synthetic data with ground truth explanations, the model training and how to create an explanation with SHAPformer.

### Datasets

You can download the datasets used in the paper via this link: [data.zip](https://bwsyncandshare.kit.edu/s/22JeaRwgGTqxNyS)

The zip file contains both datasets:
- The synthetic dataset with ground truth explanations.
- The real-world dataset containing the electrical load of TransnetBW (original source: [OPSD](https://data.open-power-system-data.org/time_series/2020-10-06)) and weather data (original source: [Copernicus](https://cds.climate.copernicus.eu/datasets/sis-energy-derived-reanalysis)).

For a demonstration how to use the synthetic dataset with ground truth, refer to this notebook: [ground_truth_explanations.ipynb](notebooks/ground_truth_explanations.ipynb)

### Model training

For example code on how to train a SHAPformer model, refer to this notebook: [training.ipynb](notebooks/training.ipynb)

### Explanation generation

For example code on how to generate explanations with a trained SHAPformer model, refer to this notebook: [evaluate.ipynb](notebooks/evaluation.ipynb).