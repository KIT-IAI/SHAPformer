"""SHAPformer: explainable time-series forecasting with sampling-free SHAP for Transformers."""
from importlib.metadata import PackageNotFoundError, version

from .data import load_data
from .models.exogenous_transformer import ExogenousTransformer
from .models.shapformer import SHAPformer
from .shap import explain_sample

try:
    __version__ = version("shapformer")
except PackageNotFoundError:  # not installed, e.g. imported from a source checkout
    __version__ = "unknown"

__all__ = ["SHAPformer", "ExogenousTransformer", "explain_sample", "load_data", "plot_explanation",
           "plot_dependence", "__version__"]


def __getattr__(name):
    # the plots need matplotlib (pip install shapformer[plot]), so they are only imported when used
    if name in ("plot_explanation", "plot_dependence"):
        from . import plot
        return getattr(plot, name)
    raise AttributeError(f"module 'shapformer' has no attribute '{name}'")
