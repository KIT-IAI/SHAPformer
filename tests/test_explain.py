"""SHAPformer.explain: additivity, consistency with predict and the permutation approximation, output format."""
import numpy as np
import pandas as pd
import pytest

from conftest import INSTANCE_NORMS, LENGTH_CONFIGS, TARGET, get_fitted_model, make_data, window

DATA = make_data()


def explain(config, instance_norm="none", method="exact", start=0, **kwargs):
    model = get_fitted_model(config, instance_norm)
    past, future = window(model, DATA, start)
    return model, past, future, model.explain(past, future, method=method, progress=False, **kwargs)


@pytest.mark.parametrize("method", ["exact", "permutation"])
@pytest.mark.parametrize("instance_norm", INSTANCE_NORMS)
@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_base_plus_shap_values_is_the_prediction(config, instance_norm, method):
    _, _, _, explanation = explain(config, instance_norm, method, n_permutations=3, seed=0)
    total = explanation["base"] + explanation["shap_values"].sum(axis=1)
    np.testing.assert_allclose(total, explanation["prediction"], rtol=1e-6)


@pytest.mark.parametrize("method", ["exact", "permutation"])
@pytest.mark.parametrize("instance_norm", INSTANCE_NORMS)
@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_explained_prediction_is_the_prediction(config, instance_norm, method):
    model, past, future, explanation = explain(config, instance_norm, method, start=5, n_permutations=2, seed=0)
    np.testing.assert_allclose(explanation["prediction"], model.predict(past, future), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("config", ["3_equal_periods", "uneven_periods", "single_period"])
def test_permutation_values_approximate_the_exact_values(config):
    _, _, _, exact = explain(config, method="exact")
    _, _, _, approx = explain(config, method="permutation", n_permutations=1000, seed=0)
    np.testing.assert_allclose(approx["base"], exact["base"], rtol=1e-5)
    scale = np.abs(exact["shap_values"].to_numpy()).max()
    difference = np.abs(approx["shap_values"] - exact["shap_values"]).to_numpy().max()
    assert difference < 0.05 * scale


def test_permutation_values_are_reproducible_with_a_seed():
    shap_values = [explain("uneven_periods", method="permutation", n_permutations=5, seed=seed)[3]["shap_values"]
                   for seed in (1, 1, 2)]
    pd.testing.assert_frame_equal(shap_values[0], shap_values[1])
    assert not np.allclose(shap_values[0], shap_values[2])


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        explain("3_equal_periods", method="kernel")


@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_output_format(config):
    model, past, future, explanation = explain(config, start=3)
    n_periods = model.get_n_periods()
    periods = [TARGET] if n_periods == 1 else [f"period {i + 1}" for i in range(n_periods)]
    assert list(explanation["shap_values"].columns) == ["temperature", "hour_of_day"] + periods
    for name in ("prediction", "base", "shap_values"):
        assert explanation[name].index.equals(future.index), name
    pd.testing.assert_frame_equal(explanation["past"], past[[TARGET, "temperature", "hour_of_day"]])
    pd.testing.assert_frame_equal(explanation["future"], future[["temperature", "hour_of_day"]])

    table = explanation["periods"]
    assert list(table.index) == periods
    assert list(table["length"]) == list(model.get_period_lengths())
    ends = np.cumsum(model.get_period_lengths())
    assert list(table["start"]) == list(past.index[ends - table["length"].to_numpy()])
    assert list(table["end"]) == list(past.index[ends - 1])


def test_longer_past_uses_the_last_input_length_rows():
    model = get_fitted_model("3_equal_periods")
    past, future = window(model, DATA, start=20)
    longer_past = DATA.iloc[:20 + model.get_input_length()]
    short = model.explain(past, future, progress=False)
    long = model.explain(longer_past, future, progress=False)
    pd.testing.assert_frame_equal(short["shap_values"], long["shap_values"])
    pd.testing.assert_frame_equal(short["past"], long["past"])
