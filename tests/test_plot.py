"""Plots and plot helpers (smoke tests for the figures, exact checks for the helpers)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")  # plotting is optional

import matplotlib.colors  # noqa: E402

from conftest import LENGTH_CONFIGS, TARGET, fit_model, get_fitted_model, make_data, make_model, window  # noqa: E402
from src.plot import (FEATURE_COLORS, format_duration, get_color, get_period_columns, get_target_lag,  # noqa: E402
                      plot_dependence, plot_explanation, summarize_load_impact, summarize_nonload_impact,
                      xkcd_color)

DATA = make_data()


def explanations(config, n=4):
    model = get_fitted_model(config)
    return [model.explain(*window(model, DATA, start), progress=False) for start in range(0, 5 * n, 5)]


# --- helpers ---

def test_xkcd_color():
    assert xkcd_color("sky blue") == "#75bbfd"
    with pytest.raises(KeyError):
        xkcd_color("not a color")


def test_feature_colors_are_valid():
    for feature, color in FEATURE_COLORS.items():
        assert matplotlib.colors.is_color_like(color), feature


def test_period_colors():
    assert get_color("period 1", 3) == "#15df1c"  # the oldest period is light green
    assert get_color("period 3", 3) == "#152015"  # the most recent one dark green
    assert get_color("period 2", 3) not in ("#15df1c", "#152015")
    assert get_color("day 1", 3) == "#15df1c"  # explanations from before the periods were configurable
    assert get_color("period 1", 1) == "#152015"
    assert get_color("temperature") == FEATURE_COLORS["temperature"]
    assert get_color("unknown feature") == "gray"


@pytest.mark.parametrize("duration, text", [
    (pd.Timedelta(hours=24), "24 h"),
    (pd.Timedelta(minutes=90), "1.5 h"),
    (pd.Timedelta(minutes=15), "15 min"),
])
def test_format_duration(duration, text):
    assert format_duration(duration) == text


@pytest.mark.parametrize("input_length, horizon, steps_per_week, lag", [
    (168, 24, 168, 168),
    (336, 168, 168, 168),
    (12, 6, 168, 12),
    (672, 96, 672, 672),
])
def test_target_lag(input_length, horizon, steps_per_week, lag):
    assert get_target_lag(input_length, horizon, steps_per_week) == lag


def test_target_lag_needs_horizon_up_to_the_input_length():
    with pytest.raises(ValueError, match="horizon <= input length"):
        get_target_lag(12, 20)


def test_summaries():
    shap_values = pd.DataFrame({"temperature": [1.0, 2.0], "period 1": [0.5, 0.0], "period 2": [-1.0, 3.0]})
    assert get_period_columns(shap_values) == ["period 1", "period 2"]
    pd.testing.assert_frame_equal(summarize_load_impact(shap_values),
                                  pd.DataFrame({"temperature": [1.0, 2.0], "load": [-0.5, 3.0]}))
    pd.testing.assert_frame_equal(summarize_nonload_impact(shap_values),
                                  pd.DataFrame({"other": [1.0, 2.0], "period 1": [0.5, 0.0],
                                                "period 2": [-1.0, 3.0]}))


def test_single_period_is_named_after_the_target():
    shap_values = pd.DataFrame({"temperature": [1.0], "load": [2.0]})
    assert get_period_columns(shap_values, "load") == ["load"]
    assert get_period_columns(shap_values) == []


# --- figures ---

@pytest.mark.parametrize("mode", ["all", "load", "periods", "days"])
@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_plot_explanation(config, mode):
    explanation = explanations(config, n=1)[0]
    actual = DATA[TARGET].iloc[:len(explanation["prediction"])]
    ax = plot_explanation(explanation, mode=mode, actual=actual, title="test")
    labels = ax.get_legend_handles_labels()[1]
    assert "prediction" in labels and "base" in labels and "actual" in labels


def test_plot_explanation_unknown_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        plot_explanation(explanations("3_equal_periods", n=1)[0], mode="weeks")


def test_plot_explanation_saves_the_figure(tmp_path):
    path = tmp_path / "explanation.png"
    plot_explanation(explanations("3_equal_periods", n=1)[0], path=str(path))
    assert path.stat().st_size > 0


def test_plot_explanation_quarter_hourly():
    data = make_data(n_rows=400, freq="15min")
    model = fit_model(make_model(input_length=16, horizon=8, period_length=4), data)
    explanation = model.explain(*window(model, data, start=0), progress=False)
    ax = plot_explanation(explanation)
    assert ax.get_xlabel() == "Time [h]"


@pytest.mark.parametrize("feature, color_by", [
    (TARGET, None),
    ("temperature", "hour_of_day"),
    ("hour_of_day", "prediction"),
])
@pytest.mark.parametrize("config", [c for c, lengths in LENGTH_CONFIGS.items()
                                    if lengths["horizon"] <= (lengths["input_length"] or 12)])
def test_plot_dependence(config, feature, color_by):
    ax = plot_dependence(explanations(config), feature, color_by=color_by, horizon=1)
    n_dots = len(ax.collections[0].get_offsets())
    assert n_dots == 4


def test_plot_dependence_all_horizons(tmp_path):
    path = tmp_path / "dependence.png"
    ax = plot_dependence(explanations("3_equal_periods"), "temperature", horizon=None, path=str(path))
    assert len(ax.collections[0].get_offsets()) == 4 * 6
    assert path.stat().st_size > 0


def test_plot_dependence_shows_the_summed_period_values():
    explanation_list = explanations("3_equal_periods")
    ax = plot_dependence(explanation_list, TARGET, horizon=2)
    y = ax.collections[0].get_offsets()[:, 1]
    periods = ["period 1", "period 2", "period 3"]
    expected = [e["shap_values"][periods].sum(axis=1).iloc[1] for e in explanation_list]
    np.testing.assert_allclose(y, expected)
