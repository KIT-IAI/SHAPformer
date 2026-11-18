"""Shared helpers and fixtures: a tiny SHAPformer on CPU and synthetic data."""
try:
    import matplotlib
    matplotlib.use("Agg")  # no windows during tests, must be set before pyplot is imported
except ImportError:  # matplotlib is only needed for the plots; their tests are skipped without it
    matplotlib = None

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.shapformer import SHAPformer

TARGET = "load"
FEATURE_CATEGORIES = {"temperature": None, "hour_of_day": 24}

# input lengths, horizons and period lengths that the tests are run with
LENGTH_CONFIGS = {
    "3_equal_periods": dict(input_length=12, horizon=6, period_length=4),
    "single_period": dict(input_length=12, horizon=6, period_length=12),
    "uneven_periods": dict(input_length=None, horizon=6, period_length=(6, 3, 2, 1)),
    "4_periods_horizon_1": dict(input_length=8, horizon=1, period_length=2),
    "horizon_longer_than_input": dict(input_length=12, horizon=20, period_length=4),
}
INSTANCE_NORMS = ("none", "mean", "RevIN")


def make_data(n_rows: int = 300, freq: str = "h", seed: int = 0, start: str = "2024-01-01",
              tz: str = "UTC") -> pd.DataFrame:
    """A load series with a daily pattern that depends on a random-walk temperature."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_rows, freq=freq, tz=tz)
    hour = index.hour.to_numpy().astype(np.int64)
    temperature = 10 + np.cumsum(rng.normal(0, 0.5, n_rows))
    load = 100 + 20 * np.sin(2 * np.pi * hour / 24) - 1.5 * temperature + rng.normal(0, 1, n_rows)
    return pd.DataFrame({TARGET: load, "temperature": temperature, "hour_of_day": hour}, index=index)


def make_model(input_length=12, horizon=6, period_length=4, instance_norm="none", d_model=8, n_layers=1,
               n_heads=1, output_dimensions=1, seed=0) -> SHAPformer:
    """A tiny untrained model on CPU (without scalers; use fit_model for predict and explain)."""
    torch.manual_seed(seed)
    return SHAPformer(
        target=TARGET,
        feature_categories=FEATURE_CATEGORIES,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        output_dimensions=output_dimensions,
        instance_norm=instance_norm,
        input_length=input_length,
        horizon=horizon,
        period_length=period_length,
    )


def validation_start(data: pd.DataFrame):
    return data.index[int(len(data) * 0.7)]


def fit_model(model: SHAPformer, data: pd.DataFrame = None, epochs: int = 1, **kwargs) -> SHAPformer:
    if data is None:
        data = make_data()
    kwargs.setdefault("batch_size", 64)
    return model.fit(data, validation_start=validation_start(data), epochs=epochs, **kwargs)


def window(model: SHAPformer, data: pd.DataFrame, start: int = 0):
    """past (input_length rows with the target) and future (horizon rows without the target) from row start on."""
    input_length, horizon = model.get_input_length(), model.get_horizon()
    past = data.iloc[start:start + input_length]
    future = data.iloc[start + input_length:start + input_length + horizon].drop(columns=TARGET)
    return past, future


_FITTED_MODELS = {}


def get_fitted_model(config: str, instance_norm: str = "none") -> SHAPformer:
    """A model of a length configuration, trained for one epoch; cached for the whole test session."""
    key = (config, instance_norm)
    if key not in _FITTED_MODELS:
        model = make_model(instance_norm=instance_norm, **LENGTH_CONFIGS[config])
        _FITTED_MODELS[key] = fit_model(model)
    return _FITTED_MODELS[key]


@pytest.fixture
def data() -> pd.DataFrame:
    return make_data()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    if matplotlib is not None:
        import matplotlib.pyplot as plt
        plt.close("all")
