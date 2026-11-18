"""The installed package: version, top-level API and the optional plotting dependency."""
import subprocess
import sys
from importlib.metadata import version

import shapformer


def test_version():
    assert shapformer.__version__ == version("shapformer")


def test_top_level_api():
    from shapformer.data import load_data
    from shapformer.models.exogenous_transformer import ExogenousTransformer
    from shapformer.models.shapformer import SHAPformer
    from shapformer.shap import explain_sample

    assert shapformer.SHAPformer is SHAPformer
    assert shapformer.ExogenousTransformer is ExogenousTransformer
    assert shapformer.explain_sample is explain_sample
    assert shapformer.load_data is load_data
    for name in shapformer.__all__:
        assert getattr(shapformer, name) is not None, name


def test_plots_are_available_at_the_top_level():
    from shapformer.plot import plot_dependence, plot_explanation

    assert shapformer.plot_explanation is plot_explanation
    assert shapformer.plot_dependence is plot_dependence


def test_import_does_not_need_matplotlib():
    # a fresh interpreter, since matplotlib is already imported in this one
    code = "import sys, shapformer; print('matplotlib' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
