"""SHAPformer constructor, fit, predict, scaling and saving."""
import random

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import (LENGTH_CONFIGS, TARGET, fit_model, get_fitted_model, make_data, make_model, validation_start,
                      window)
from src.shap import get_full_masks


# --- constructor ---

@pytest.mark.parametrize("kwargs, message", [
    (dict(input_length=10, period_length=4), "multiple of period_length"),
    (dict(input_length=12, period_length=0), "multiple of period_length"),
    (dict(input_length=12, period_length=(6, 3)), "add up to"),
    (dict(period_length=(6, 0, 2)), "all positive"),
    (dict(period_length=()), "at least one length"),
    (dict(horizon=0), "horizon must be positive"),
    (dict(instance_norm="z-score"), "Unknown instance_norm"),
])
def test_invalid_arguments(kwargs, message):
    arguments = dict(input_length=12, horizon=6, period_length=4)
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        make_model(**arguments)


@pytest.mark.parametrize("kwargs, input_length, period_lengths", [
    (dict(input_length=None, period_length=24), 168, (24,) * 7),
    (dict(input_length=None, period_length=(6, 3, 2, 1)), 12, (6, 3, 2, 1)),
    (dict(input_length=12, period_length=12), 12, (12,)),
    (dict(input_length=8, period_length=2), 8, (2, 2, 2, 2)),
    (dict(input_length=12, period_length=np.int64(4)), 12, (4, 4, 4)),
])
def test_input_length_and_periods(kwargs, input_length, period_lengths):
    model = make_model(horizon=6, **kwargs)
    assert model.get_input_length() == input_length
    assert model.get_period_lengths() == period_lengths
    assert model.get_n_periods() == len(period_lengths)
    assert model.get_horizon() == 6


def test_default_lengths():
    model = make_model(input_length=None, horizon=168, period_length=24)
    assert (model.get_input_length(), model.get_horizon(), model.get_n_periods()) == (168, 168, 7)


# --- windows without missing values ---

def brute_force_window_starts(data: pd.DataFrame, window_length: int, step: pd.Timedelta):
    starts = []
    for start in range(len(data) - window_length + 1):
        rows = data.iloc[start:start + window_length]
        consecutive = (rows.index[1:] - rows.index[:-1] == step).all()
        if consecutive and not rows.isna().any().any():
            starts.append(start)
    return starts


def test_window_starts_skip_missing_values_and_gaps():
    model = make_model(input_length=4, horizon=2, period_length=2)  # windows of 6 rows
    data = make_data(n_rows=30)
    data.iloc[8, 1] = np.nan  # a missing temperature
    data = data.drop(data.index[20])  # a missing time step
    step = pd.Timedelta(hours=1)
    starts = model._window_starts(data, data.index, step)
    assert starts == brute_force_window_starts(data, 6, step)
    assert 2 in starts and 9 in starts  # the windows right before and after the missing value
    assert 3 not in starts and 8 not in starts
    # the gap is between the rows 19 and 20: windows 15 to 19 contain it, 14 ends right before it
    assert 14 in starts and 15 not in starts and 19 not in starts and 20 in starts


def test_window_starts_of_too_short_data():
    model = make_model(input_length=4, horizon=2, period_length=2)
    data = make_data(n_rows=5)
    assert model._window_starts(data, data.index, pd.Timedelta(hours=1)) == []


# --- fit ---

@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_fit_and_predict(config):
    model = make_model(**LENGTH_CONFIGS[config])
    data = make_data()
    assert fit_model(model, data, epochs=2) is model
    assert not model.training
    assert set(model.scalers) == {TARGET, "temperature"}
    past, future = window(model, data, start=10)
    prediction = model.predict(past, future)
    assert prediction.index.equals(future.index)
    assert prediction.name == TARGET
    assert np.isfinite(prediction).all()


def test_predict_returns_the_target_unit():
    model = get_fitted_model("3_equal_periods")
    data = make_data()
    past, future = window(model, data, start=7)
    x_enc, x_dec = model._prepare_inputs(past, future)
    ft_mask, attn_mask = get_full_masks(3, device="cpu", input_length=model.get_input_length())
    with torch.no_grad():
        scaled = model({k: v[None] for k, v in x_enc.items()}, {k: v[None] for k, v in x_dec.items()},
                       ft_mask[None], ft_mask[None, 1:], attn_mask[None])[0].numpy()
    scaler = model.scalers[TARGET]
    np.testing.assert_allclose(model.predict(past, future), scaled * scaler.scale_[0] + scaler.mean_[0], rtol=1e-5)


def test_scalers_ignore_missing_values_and_use_training_data_only():
    data = make_data()
    data.iloc[[3, 50, 100], 1] = np.nan
    model = fit_model(make_model(), data)
    train = data[data.index < validation_start(data)]
    np.testing.assert_allclose(model.scalers["temperature"].mean_[0], train["temperature"].mean())
    np.testing.assert_allclose(model.scalers["temperature"].scale_[0], train["temperature"].std(ddof=0))
    np.testing.assert_allclose(model.scalers[TARGET].mean_[0], train[TARGET].mean())


@pytest.mark.parametrize("instance_norm", ["mean", "RevIN"])
def test_target_is_not_scaled_globally_with_instance_norm(instance_norm):
    model = fit_model(make_model(instance_norm=instance_norm))
    scaler = model.scalers[TARGET]
    assert (scaler.mean_[0], scaler.scale_[0]) == (0, 1)


def test_fit_with_missing_values_and_gaps(capsys):
    data = make_data()
    data.iloc[[30, 120], 0] = np.nan
    data = data.drop(data.index[60:63])
    model = fit_model(make_model(), data)
    assert "Using" in capsys.readouterr().out  # reports the windows that are left out
    past, future = window(model, data, start=0)
    assert np.isfinite(model.predict(past, future)).all()


def test_fit_without_complete_windows():
    data = make_data()
    data.iloc[::10, 1] = np.nan  # no 18 consecutive complete rows
    with pytest.raises(ValueError, match="no window"):
        fit_model(make_model(), data)


def test_window_stride():
    model = make_model()
    recorded = []
    get_batch = model._get_batch

    def recording_get_batch(series, starts):
        recorded.append((id(series), list(starts)))
        return get_batch(series, starts)

    model._get_batch = recording_get_batch
    fit_model(model, epochs=1, window_stride=4)
    training_series = recorded[0][0]
    training_starts = [s for series, starts in recorded if series == training_series for s in starts]
    assert training_starts and all(start % 4 == 0 for start in training_starts)


def test_invalid_window_stride():
    with pytest.raises(ValueError, match="window_stride"):
        fit_model(make_model(), window_stride=0)


def test_early_stopping_restores_the_best_epoch():
    model = make_model()
    losses = iter([1.0, 0.5, 0.9, 0.8, 0.7, 0.6])
    states = []

    def fake_validation_loss(*args, **kwargs):
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        return next(losses)

    model._validation_loss = fake_validation_loss
    fit_model(model, epochs=6, patience=2, learning_rate=0.01)
    assert len(states) == 4  # stopped after two epochs without improvement
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, states[1][name])  # restored epoch 2
    assert not torch.equal(states[1]["projection.weight"], states[3]["projection.weight"])


@pytest.mark.parametrize("data_tz, start", [
    ("UTC", "2024-01-09"),  # naive validation start with an aware index
    ("UTC", pd.Timestamp("2024-01-09", tz="Europe/Berlin")),
    (None, "2024-01-09"),
])
def test_validation_start_time_zones(data_tz, start):
    data = make_data(tz=data_tz)
    model = make_model()
    model.fit(data, validation_start=start, epochs=1, batch_size=64)
    assert not model.training


# --- predict and explain inputs ---

@pytest.mark.parametrize("method", ["predict", "explain"])
def test_wrong_input_lengths(method):
    model = get_fitted_model("3_equal_periods")
    past, future = window(model, make_data())
    call = getattr(model, method)
    with pytest.raises(ValueError, match="past must have at least"):
        call(past.iloc[1:], future)
    with pytest.raises(ValueError, match="future must have"):
        call(past, future.iloc[1:])


def test_categorical_values_out_of_range():
    model = get_fitted_model("3_equal_periods")
    past, future = window(model, make_data())
    future = future.copy()
    future.iloc[0, future.columns.get_loc("hour_of_day")] = 24
    with pytest.raises(ValueError, match="categories"):
        model.predict(past, future)


# --- saving ---

@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_save_and_load(config, tmp_path):
    model = get_fitted_model(config, "RevIN")
    path = tmp_path / "model.pt"
    torch.save(model, path)
    loaded = torch.load(path, weights_only=False)
    past, future = window(model, make_data(), start=4)
    pd.testing.assert_series_equal(loaded.predict(past, future), model.predict(past, future))
    assert loaded.get_period_lengths() == model.get_period_lengths()
    assert set(loaded.scalers) == set(model.scalers)


# --- training ---

def forecast_error(model, data, subset: str) -> float:
    # mean squared error of the (unmasked) forecasts in the unit of the target, on the training windows or on the
    # validation windows (forecasts from the validation start on, with inputs from before it)
    input_length, horizon = model.get_input_length(), model.get_horizon()
    n_train = int(len(data) * 0.7)  # the row of validation_start
    if subset == "training":
        starts = range(0, n_train - input_length - horizon + 1, 5)
    else:
        starts = range(n_train - input_length, len(data) - input_length - horizon + 1, 5)
    errors = []
    for start in starts:
        past, future = window(model, data, start)
        actual = data[TARGET].iloc[start + input_length:start + input_length + horizon].to_numpy()
        errors.append(np.mean((model.predict(past, future).to_numpy() - actual) ** 2))
    return float(np.mean(errors))


@pytest.mark.parametrize("instance_norm", ["none", "mean", "RevIN"])
def test_training_reduces_the_error(instance_norm):
    # shows that the model learns, on the training and on the validation windows;
    # the forecast quality is evaluated in the notebooks on real data
    random.seed(0)
    torch.manual_seed(0)
    data = make_data(n_rows=600)
    model = make_model(instance_norm=instance_norm)
    fit_model(model, data, epochs=0)  # only fits the scalers
    initial_errors = {subset: forecast_error(model, data, subset) for subset in ("training", "validation")}
    fit_model(model, data, epochs=10, learning_rate=0.001, batch_size=16)
    for subset, initial_error in initial_errors.items():
        assert forecast_error(model, data, subset) < initial_error, subset


# --- default lengths ---

def test_default_lengths_end_to_end():
    data = make_data(n_rows=1200)
    model = fit_model(make_model(input_length=None, horizon=168, period_length=24), data, epochs=1)
    past, future = window(model, data, start=800)
    explanation = model.explain(past, future, progress=False)
    assert list(explanation["shap_values"].columns[2:]) == [f"period {i}" for i in range(1, 8)]
    total = explanation["base"] + explanation["shap_values"].sum(axis=1)
    np.testing.assert_allclose(total, explanation["prediction"], rtol=1e-6)
