from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


XKCD_COLORS_FILE = Path(__file__).parent.parent / "xkcd_colors.txt"


def get_xkcd_colors() -> Dict[str, str]:
    # maps color names to hex codes, from the most to the least common color
    with open(XKCD_COLORS_FILE) as f:
        lines = f.readlines()
    colors = [line.strip().split("\t") for line in lines[1:] if line.strip()]
    return {name: hex_code for name, hex_code in colors[::-1]}


XKCD_COLOR_NAMES = get_xkcd_colors()
XKCD_COLORS = list(XKCD_COLOR_NAMES.values())
PREDICTION_COLOR = "black"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATURE_COLORS = {
    "hour_of_day": XKCD_COLORS[3],
    "day_of_week": XKCD_COLORS[10],
    "month": XKCD_COLORS[11],
    "holiday": XKCD_COLORS[4],
    "temperature": XKCD_COLORS[8],
    "precipitation": XKCD_COLORS[2],
    "irradiance": XKCD_COLOR_NAMES["goldenrod"],
    "load": XKCD_COLORS[1],
    "day 1": "#15df1c",
    "day 2": "#15bf1b",
    "day 3": "#159f1a",
    "day 4": "#157f18",
    "day 5": "#156017",
    "day 6": "#154016",
    "day 7": "#152015",
    "other": XKCD_COLORS[13]
}
DOT_COLOR = "tab:blue"
MEAN_COLOR = "red"
CYCLIC_FEATURES = {"hour_of_day", "day_of_week", "month"}
CYCLIC_COLORMAP = "twilight"
COLORMAP = "viridis"
DISCRETE_TICK_LABELS = {
    "day_of_week": DAY_NAMES,
    "month": [str(month) for month in range(1, 13)],
    "holiday": ["no", "yes"]
}
FEATURE_LABELS = {
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
    "month": "Month",
    "holiday": "Holiday",
    "temperature": "Temperature [°C]",
    "precipitation": "Precipitation",
    "irradiance": "Irradiance",
    "prediction": "Prediction"
}


def get_color(feature: str) -> str:
    return FEATURE_COLORS.get(feature, "gray")


def summarize_load_impact(shap_values: pd.DataFrame) -> pd.DataFrame:
    # sums the SHAP values of the past days into one "load" group
    days = [ft for ft in shap_values if ft.startswith("day ")]
    summarized = shap_values.drop(columns=days)
    summarized["load"] = shap_values[days].sum(axis=1)
    return summarized


def summarize_nonload_impact(shap_values: pd.DataFrame) -> pd.DataFrame:
    # sums the SHAP values of all features except the past days into one "other" group
    days = [ft for ft in shap_values if ft.startswith("day ")]
    summarized = shap_values[days].copy()
    summarized.insert(0, "other", shap_values.drop(columns=days).sum(axis=1))
    return summarized


def plot_day_end_lines(ax, start_hour: int, first_day: int, horizon: int):
    ymin, ymax = ax.get_ylim()
    day_ends = np.arange(24 - start_hour, horizon, 24)
    ax.vlines(day_ends, ymin, ymax, color="gray", linestyles="dashed", linewidth=0.8)
    ax.set_ylim(ymin, ymax)
    for i in range(horizon // 24 + 2):
        pos = 12 - start_hour + i * 24
        if 4 <= pos <= horizon - 4:
            ax.text(pos, ymin + (ymax - ymin) * 0.075, DAY_NAMES[(first_day + i) % 7],
                    fontsize=10, ha="center", va="top")


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

    mode: "all" shows every feature group, "load" sums the past days into one group,
    "days" sums all other features into one group.
    """
    shap_values = explanation["shap_values"]
    if mode == "load":
        shap_values = summarize_load_impact(shap_values)
    elif mode == "days":
        shap_values = summarize_nonload_impact(shap_values)
    elif mode != "all":
        raise ValueError(f"Unknown mode '{mode}', must be 'all', 'load' or 'days'.")

    base = explanation["base"].to_numpy()
    prediction = explanation["prediction"].to_numpy()
    horizon = len(base)
    x = np.arange(horizon)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    upper = base.copy()
    lower = base.copy()
    for feature in shap_values:
        values = shap_values[feature].to_numpy()
        positive = np.clip(values, a_min=0, a_max=None)
        negative = np.clip(values, a_min=None, a_max=0)
        color = get_color(feature)
        ax.fill_between(x, upper, upper + positive, color=color, alpha=alpha, linewidth=0, label=feature)
        ax.fill_between(x, lower + negative, lower, color=color, alpha=alpha, linewidth=0)
        upper = upper + positive
        lower = lower + negative

    ax.plot(x, base, color="gray", linestyle=":", label="base")
    ax.plot(x, prediction, color=PREDICTION_COLOR, label="prediction")
    if actual is not None:
        ax.plot(x, np.asarray(actual), color=PREDICTION_COLOR, linestyle="--", label="actual")

    ax.set_xticks(list(range(0, horizon + 1, 24)))
    ax.set_xlim(0, horizon)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(f"Prediction [{unit}]" if unit else "Prediction")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)

    start = pd.Timestamp(explanation["base"].index[0])
    plot_day_end_lines(ax, start.hour, start.dayofweek, horizon)

    if path is not None:
        ax.figure.savefig(path, bbox_inches="tight")
    return ax


def _get_target(explanation: Dict) -> str:
    return explanation["past"].columns[0]


def _get_feature_values(explanation: Dict, feature: str) -> np.ndarray:
    # values of a feature at the forecasted time steps;
    # for the target, the value one week earlier (at the same position in the past)
    if feature == "prediction":
        return explanation["prediction"].to_numpy()
    if feature == _get_target(explanation):
        return explanation["past"][feature].to_numpy()
    return explanation["future"][feature].to_numpy()


def _get_feature_shap_values(explanation: Dict, feature: str) -> np.ndarray:
    # for the target, the SHAP values of all past days are summed
    shap_values = explanation["shap_values"]
    if feature == _get_target(explanation):
        days = [ft for ft in shap_values if ft.startswith("day ")]
        return shap_values[days].sum(axis=1).to_numpy()
    return shap_values[feature].to_numpy()


def get_feature_label(feature: str, target: str, unit: Optional[str] = None) -> str:
    if feature == target:
        return f"{feature.capitalize()} one week earlier" + (f" [{unit}]" if unit else "")
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

    For the target (e.g. load), the x-axis shows the target one week before the forecasted time step,
    and the y-axis the sum of the SHAP values of all past days.
    Discrete (integer) features are jittered horizontally, and their mean SHAP values are drawn as a line.
    color_by: optional second feature (or "prediction") that colors the dots.
    horizon: forecast step to plot, counted from 1 (24 = the 24th predicted value); None plots all steps.
    """
    explanations = list(explanations)
    target = _get_target(explanations[0])
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
        ax.figure.colorbar(scatter, ax=ax, label=get_feature_label(color_by, target, unit))

    if discrete:
        values = np.unique(x)
        means = [y[x == value].mean() for value in values]
        ax.plot(values, means, color=MEAN_COLOR)
        if feature in DISCRETE_TICK_LABELS:
            labels = DISCRETE_TICK_LABELS[feature]
            ax.set_xticks(range(len(labels)), labels)
            ax.set_xlim(-0.5, len(labels) - 0.5)

    ax.set_xlabel(get_feature_label(feature, target, unit))
    ax.set_ylabel(f"SHAP value [{unit}]" if unit else "SHAP value")
    if title is None and horizon is not None:
        title = f"{get_feature_label(feature, target)}, horizon {horizon} h"
    if title is not None:
        ax.set_title(title)

    if path is not None:
        ax.figure.savefig(path, bbox_inches="tight")
    return ax
