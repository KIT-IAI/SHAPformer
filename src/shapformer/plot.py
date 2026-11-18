from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import XKCD_COLORS
except ImportError as error:
    raise ImportError("Plotting needs matplotlib, which is an optional dependency of shapformer. "
                      "Install it with: pip install \"shapformer[plot]\"") from error


def xkcd_color(name: str) -> str:
    """Hex code of a color from the xkcd color survey, e.g. xkcd_color("sky blue")."""
    return XKCD_COLORS[f"xkcd:{name}"]


PREDICTION_COLOR = "black"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATURE_COLORS = {
    "hour_of_day": xkcd_color("pink"),
    "quarter_of_day": xkcd_color("pink"),
    "day_of_week": xkcd_color("magenta"),
    "month": xkcd_color("yellow"),
    "holiday": xkcd_color("brown"),
    "temperature": xkcd_color("orange"),
    "precipitation": xkcd_color("blue"),
    "irradiance": xkcd_color("goldenrod"),
    "load": xkcd_color("green"),
    "load_forecast": xkcd_color("red"),
    "pv_wind_forecast": xkcd_color("sky blue"),
    # weather covariates of the fev-bench solar dataset
    "day_length": xkcd_color("sunflower"),
    "temp": xkcd_color("orange"),
    "humidity": xkcd_color("teal"),
    "pressure": xkcd_color("mauve"),
    "wind_speed": xkcd_color("lavender"),
    "rain_1h": xkcd_color("blue"),
    "snow_1h": xkcd_color("light blue"),
    "clouds_all": xkcd_color("slate"),
    "global_horizontal_irradiance": xkcd_color("goldenrod"),
    # weather covariates of the UCI bike sharing dataset (besides temperature, humidity and wind_speed)
    "weather": xkcd_color("blue"),
    "felt_temperature": xkcd_color("orange"),
    "other": xkcd_color("grey")
}
DOT_COLOR = "tab:blue"
MEAN_COLOR = "red"
CYCLIC_FEATURES = {"hour_of_day", "quarter_of_day", "day_of_week", "month"}
CYCLIC_COLORMAP = "twilight"
COLORMAP = "viridis"
DISCRETE_TICK_LABELS = {
    "day_of_week": DAY_NAMES,
    "month": [str(month) for month in range(1, 13)],
    "holiday": ["no", "yes"],
    "weather": ["clear", "mist", "rain/snow"]
}
FEATURE_LABELS = {
    "hour_of_day": "Hour of day",
    "quarter_of_day": "Quarter of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "holiday": "Holiday",
    "temperature": "Temperature [°C]",
    "precipitation": "Precipitation",
    "irradiance": "Irradiance",
    "day_length": "Day length [min]",
    "temp": "Temperature [°C]",
    "humidity": "Humidity [%]",
    "pressure": "Pressure [hPa]",
    "wind_speed": "Wind speed",
    "rain_1h": "Rain (1 h)",
    "snow_1h": "Snow (1 h)",
    "clouds_all": "Cloud cover [%]",
    "global_horizontal_irradiance": "Irradiance (GHI)",
    "weather": "Weather situation",
    "felt_temperature": "Felt temperature [°C]",
    "prediction": "Prediction"
}


# the past periods are colored from light green (period 1, the oldest one) to dark green (the most recent one)
FIRST_PERIOD_COLOR = (0x15, 0xdf, 0x1c)
LAST_PERIOD_COLOR = (0x15, 0x20, 0x15)
WEEK = pd.Timedelta(days=1) * 7
MAX_DAY_TICKS = 15  # at most this many day ticks on the time axis of plot_explanation


def get_period_color(period: int, n_periods: int) -> str:
    # color of period `period` (counted from 1) out of n_periods past periods, evenly spaced from light to dark green
    t = (period - 1) / (n_periods - 1) if n_periods > 1 else 1.0
    rgb = [round(first + t * (last - first)) for first, last in zip(FIRST_PERIOD_COLOR, LAST_PERIOD_COLOR)]
    return "#" + "".join(f"{value:02x}" for value in rgb)


def _period_number(feature: str) -> Optional[int]:
    # 3 for "period 3" (or "day 3" in explanations created before periods were configurable), else None
    for prefix in ("period ", "day "):
        if feature.startswith(prefix) and feature[len(prefix):].isdigit():
            return int(feature[len(prefix):])
    return None


def get_color(feature: str, n_periods: int = 7) -> str:
    number = _period_number(feature)
    if number is not None:
        return get_period_color(number, n_periods)
    return FEATURE_COLORS.get(feature, "gray")


def get_period_columns(shap_values: pd.DataFrame, target: Optional[str] = None) -> list:
    # the columns of the periods of the past target; a single period is named after the target
    periods = [ft for ft in shap_values if _period_number(ft) is not None]
    if not periods and target is not None and target in shap_values:
        periods = [target]
    return periods


def summarize_load_impact(shap_values: pd.DataFrame, target: str = "load") -> pd.DataFrame:
    # sums the SHAP values of the past periods of the target into one group, named after the target
    periods = get_period_columns(shap_values, target)
    summarized = shap_values.drop(columns=periods)
    summarized[target] = shap_values[periods].sum(axis=1)
    return summarized


def summarize_nonload_impact(shap_values: pd.DataFrame, target: str = "load") -> pd.DataFrame:
    # sums the SHAP values of all features except the past periods into one "other" group
    periods = get_period_columns(shap_values, target)
    summarized = shap_values[periods].copy()
    summarized.insert(0, "other", shap_values.drop(columns=periods).sum(axis=1))
    return summarized


def _timestamps(index) -> list:
    # the timestamps of an index, each in its own time zone (or UTC offset)
    return [pd.Timestamp(timestamp) for timestamp in index]


def get_time_step(index) -> pd.Timedelta:
    # the most frequent difference between consecutive timestamps of an index
    timestamps = _timestamps(index)
    if len(timestamps) < 2:
        return pd.Timedelta(hours=1)
    return pd.Series([b - a for a, b in zip(timestamps[:-1], timestamps[1:])]).mode()[0]


def format_duration(duration: pd.Timedelta) -> str:
    # e.g. "24 h", "15 min" or "1.5 h"
    hours = duration / pd.Timedelta(hours=1)
    if hours >= 1:
        return f"{hours:g} h"
    return f"{duration / pd.Timedelta(minutes=1):g} min"


def plot_day_end_lines(ax, index):
    # dashed lines at the starts of the days and the names of the days, from the timestamps of the forecast
    timestamps = _timestamps(index)
    horizon = len(timestamps)
    steps_per_day = pd.Timedelta(days=1) / get_time_step(index)
    dates = [timestamp.date() for timestamp in timestamps]
    day_starts = [i for i in range(1, horizon) if dates[i] != dates[i - 1]]
    ymin, ymax = ax.get_ylim()
    ax.vlines(day_starts, ymin, ymax, color="gray", linestyles="dashed", linewidth=0.8)
    ax.set_ylim(ymin, ymax)
    bounds = [0] + day_starts + [horizon]
    for start, end in zip(bounds[:-1], bounds[1:]):
        # names only for days that are at least a third of a day long in the plot
        if end - start >= steps_per_day / 3:
            ax.text((start + end) / 2, ymin + (ymax - ymin) * 0.075, DAY_NAMES[timestamps[start].day_of_week],
                    fontsize=10, ha="center", va="top")


def set_time_ticks(ax, index):
    # ticks at multiples of whole days from the forecast start, labeled with the hours since the start
    step = get_time_step(index)
    horizon = len(index)
    steps_per_day = pd.Timedelta(days=1) / step
    if steps_per_day >= 1 and float(steps_per_day).is_integer():
        n_days = horizon / steps_per_day
        days_per_tick = max(1, int(np.ceil(n_days / MAX_DAY_TICKS)))
        ticks = list(range(0, horizon + 1, int(steps_per_day) * days_per_tick))
        ax.set_xticks(ticks, [f"{tick * step / pd.Timedelta(hours=1):g}" for tick in ticks])
    ax.set_xlabel("Time [h]")


def plot_explanation(
        explanation: Dict,
        mode: str = "all",
        actual: Optional[pd.Series] = None,
        title: Optional[str] = None,
        unit: str = "MW",
        alpha: float = 0.6,
        figsize=(12, 4),
        ax=None,
        path: Optional[str] = None
):
    """
    Plots an explanation from SHAPformer.explain as a stacked area chart around the base value.
    Positive SHAP values are stacked above the base, negative ones below it, so the prediction
    lies between the upper and lower edge of the stack.

    mode: "all" shows every feature group, "load" sums the past periods of the target into one group
    (named after the target), "periods" (or "days") sums all other features into one group.
    unit: the unit of the target, used in the axis label (e.g. "€/MWh" for prices).
    The time axis is derived from the timestamps of the forecast (any time step, e.g. hours or 15 minutes).
    """
    shap_values = explanation["shap_values"]
    target = _get_target(explanation)
    if mode == "load":
        shap_values = summarize_load_impact(shap_values, target)
    elif mode in ("periods", "days"):
        shap_values = summarize_nonload_impact(shap_values, target)
    elif mode != "all":
        raise ValueError(f"Unknown mode '{mode}', must be 'all', 'load' or 'periods'.")

    base = explanation["base"].to_numpy()
    prediction = explanation["prediction"].to_numpy()
    horizon = len(base)
    x = np.arange(horizon)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    upper = base.copy()
    lower = base.copy()
    n_periods = sum(_period_number(ft) is not None for ft in shap_values)
    for feature in shap_values:
        values = shap_values[feature].to_numpy()
        positive = np.clip(values, a_min=0, a_max=None)
        negative = np.clip(values, a_min=None, a_max=0)
        if feature == target:
            color = FEATURE_COLORS.get(target, FEATURE_COLORS["load"])
        else:
            color = get_color(feature, n_periods)
        ax.fill_between(x, upper, upper + positive, color=color, alpha=alpha, linewidth=0, label=feature)
        ax.fill_between(x, lower + negative, lower, color=color, alpha=alpha, linewidth=0)
        upper = upper + positive
        lower = lower + negative

    ax.plot(x, base, color="gray", linestyle=":", label="base")
    ax.plot(x, prediction, color=PREDICTION_COLOR, label="prediction")
    if actual is not None:
        ax.plot(x, np.asarray(actual), color=PREDICTION_COLOR, linestyle="--", label="actual")

    index = explanation["base"].index
    set_time_ticks(ax, index)
    ax.set_xlim(0, horizon)
    ax.set_ylabel(f"Prediction [{unit}]" if unit else "Prediction")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)

    plot_day_end_lines(ax, index)

    if path is not None:
        ax.figure.savefig(path, bbox_inches="tight")
    return ax


def _get_target(explanation: Dict) -> str:
    return explanation["past"].columns[0]


def get_target_lag(input_length: int, horizon: int, steps_per_week: int = 168) -> int:
    # number of time steps between a forecasted time step and the past target value shown for it in dependence
    # plots: one week if the past window covers one week before every forecasted time step, else the input length
    if horizon <= steps_per_week <= input_length:
        return steps_per_week
    if horizon <= input_length:
        return input_length
    raise ValueError(f"Dependence plots of the target need horizon <= input length, but the horizon is {horizon} "
                     f"and the input length {input_length}.")


def _steps_per_week(explanation: Dict) -> int:
    return int(round(WEEK / get_time_step(explanation["prediction"].index)))


def _get_target_lag(explanation: Dict) -> int:
    return get_target_lag(len(explanation["past"]), len(explanation["prediction"]), _steps_per_week(explanation))


def _get_feature_values(explanation: Dict, feature: str) -> np.ndarray:
    # values of a feature at the forecasted time steps;
    # for the target, the value get_target_lag time steps earlier (one week, if the past window is long enough)
    if feature == "prediction":
        return explanation["prediction"].to_numpy()
    if feature == _get_target(explanation):
        past = explanation["past"][feature].to_numpy()
        start = len(past) - _get_target_lag(explanation)
        return past[start:start + len(explanation["prediction"])]
    return explanation["future"][feature].to_numpy()


def _get_feature_shap_values(explanation: Dict, feature: str) -> np.ndarray:
    # for the target, the SHAP values of all past periods are summed
    shap_values = explanation["shap_values"]
    target = _get_target(explanation)
    if feature == target:
        return shap_values[get_period_columns(shap_values, target)].sum(axis=1).to_numpy()
    return shap_values[feature].to_numpy()


def _lag_text(lag: Optional[pd.Timedelta]) -> str:
    if lag is None or lag == WEEK:
        return "one week"
    return format_duration(lag)


def get_feature_label(
        feature: str, target: str, unit: Optional[str] = None, lag: Optional[pd.Timedelta] = None
) -> str:
    # lag: time between the forecasted time step and the shown target value (None: one week)
    if feature == target:
        return f"{feature.capitalize()} {_lag_text(lag)} earlier" + (f" [{unit}]" if unit else "")
    return FEATURE_LABELS.get(feature, feature)


def plot_dependence(
        explanations: Iterable[Dict],
        feature: str,
        color_by: Optional[str] = None,
        horizon: Optional[int] = 24,
        unit: str = "MW",
        jitter: float = 0.3,
        dot_size: float = 5,
        title: Optional[str] = None,
        figsize=(6.4, 4.8),
        ax=None,
        path: Optional[str] = None
):
    """
    Dependence plot of a feature over a collection of explanations from SHAPformer.explain:
    each dot is one forecasted time step, with the feature value on the x-axis and its SHAP value on the y-axis.

    For the target (e.g. load), the x-axis shows the target one week before the forecasted time step
    (or one input length before, if the past window is shorter than one week),
    and the y-axis the sum of the SHAP values of all past periods.
    Discrete (integer) features are jittered horizontally, and their mean SHAP values are drawn as a line.
    color_by: optional second feature (or "prediction") that colors the dots.
    horizon: forecast step to plot, counted from 1 (24 = the 24th predicted value); None plots all steps.
    """
    explanations = list(explanations)
    target = _get_target(explanations[0])
    step = get_time_step(explanations[0]["prediction"].index)
    lag = _get_target_lag(explanations[0]) * step
    steps = slice(None) if horizon is None else slice(horizon - 1, horizon)

    def collect(get_values, name):
        return np.concatenate([get_values(explanation, name)[steps] for explanation in explanations])

    x = collect(_get_feature_values, feature)
    y = collect(_get_feature_shap_values, feature)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    discrete = np.issubdtype(x.dtype, np.integer)
    x_plot = x + np.random.uniform(-jitter, jitter, size=len(x)) if discrete else x

    if color_by is None:
        ax.scatter(x_plot, y, s=dot_size, c=DOT_COLOR)
    else:
        colors = collect(_get_feature_values, color_by)
        cmap = CYCLIC_COLORMAP if color_by in CYCLIC_FEATURES else COLORMAP
        scatter = ax.scatter(x_plot, y, s=dot_size, c=colors, cmap=cmap)
        ax.figure.colorbar(scatter, ax=ax, label=get_feature_label(color_by, target, unit, lag))

    if discrete:
        values = np.unique(x)
        means = [y[x == value].mean() for value in values]
        ax.plot(values, means, color=MEAN_COLOR)
        if feature in DISCRETE_TICK_LABELS:
            labels = DISCRETE_TICK_LABELS[feature]
            ax.set_xticks(range(len(labels)), labels)
            ax.set_xlim(-0.5, len(labels) - 0.5)

    ax.set_xlabel(get_feature_label(feature, target, unit, lag))
    ax.set_ylabel(f"SHAP value [{unit}]" if unit else "SHAP value")
    if title is None and horizon is not None:
        title = f"{get_feature_label(feature, target, lag=lag)}, horizon {format_duration(horizon * step)}"
    if title is not None:
        ax.set_title(title)

    if path is not None:
        ax.figure.savefig(path, bbox_inches="tight")
    return ax
