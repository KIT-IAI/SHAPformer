"""The masks hide feature groups from the model: hidden inputs must not change its output."""
import pytest
import torch

from conftest import FEATURE_CATEGORIES, INSTANCE_NORMS, LENGTH_CONFIGS, TARGET, make_model
from src.shap import expand_attention_masks, get_owen_mask_tensors

ARCHITECTURES = {"1_layer_1_head": dict(n_layers=1, n_heads=1), "2_layers_2_heads": dict(n_layers=2, n_heads=2)}


def random_inputs(model, generator):
    input_length, horizon = model.get_input_length(), model.get_horizon()

    def series(name, length):
        n_categories = FEATURE_CATEGORIES.get(name)
        if n_categories is None:
            return torch.randn(length, generator=generator)
        return torch.randint(0, n_categories, (length,), generator=generator)

    x_enc = {name: series(name, input_length) for name in model.feature_names}
    x_dec = {name: series(name, horizon) for name in model.feature_names[1:]}
    return x_enc, x_dec


def all_masks(model):
    return get_owen_mask_tensors(len(model.feature_names), model.get_period_lengths(), device="cpu")


def batch(inputs, n):
    return {name: values[None, :].repeat(n, 1) for name, values in inputs.items()}


def run(model, x_enc, x_dec, ft_masks, attn_rows):
    with torch.no_grad():
        return model(x_enc, x_dec, ft_masks, ft_masks[:, 1:], expand_attention_masks(attn_rows))


def perturb_hidden_groups(model, x_enc, x_dec, mask_keys, generator):
    """Copies of the inputs (one per mask) in which every hidden feature group has different values.

    Hidden exogenous features get new random values. Hidden time steps of the past target get new random values
    without instance normalization; with instance normalization, the statistics of the whole past window are used
    on purpose (they are part of the base value), so the hidden values are only shuffled among themselves, which
    keeps the statistics.
    """
    n_exogenous = len(model.feature_names) - 1
    period_of_step = torch.repeat_interleave(torch.arange(model.get_n_periods()),
                                             torch.tensor(model.get_period_lengths()))
    x_enc, x_dec = batch(x_enc, len(mask_keys)), batch(x_dec, len(mask_keys))
    fresh_enc, fresh_dec = random_inputs(model, generator)
    for row, key in enumerate(mask_keys):
        for j, name in enumerate(model.feature_names[1:]):
            if not key[j]:
                x_enc[name][row] = fresh_enc[name]
                x_dec[name][row] = fresh_dec[name]
        hidden_steps = torch.tensor([not key[n_exogenous + p] for p in period_of_step.tolist()])
        hidden_values = x_enc[TARGET][row, hidden_steps]
        if model.instance_norm == "none":
            new_values = torch.randn(len(hidden_values), generator=generator)
        else:
            new_values = hidden_values[torch.randperm(len(hidden_values), generator=generator)]
        x_enc[TARGET][row, hidden_steps] = new_values
    return x_enc, x_dec


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("instance_norm", INSTANCE_NORMS)
@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_hidden_inputs_do_not_change_the_output(config, instance_norm, architecture):
    model = make_model(instance_norm=instance_norm, **LENGTH_CONFIGS[config], **ARCHITECTURES[architecture]).eval()
    generator = torch.Generator().manual_seed(0)
    x_enc, x_dec = random_inputs(model, generator)
    mask_keys, ft_masks, attn_rows = all_masks(model)

    original = run(model, batch(x_enc, len(mask_keys)), batch(x_dec, len(mask_keys)), ft_masks, attn_rows)
    perturbed = run(model, *perturb_hidden_groups(model, x_enc, x_dec, mask_keys, generator), ft_masks, attn_rows)

    assert torch.isfinite(original).all()  # also with all feature groups hidden
    torch.testing.assert_close(perturbed, original, rtol=0, atol=1e-6)


@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_visible_inputs_change_the_output(config):
    # guards the test above against a model that ignores its inputs
    model = make_model(**LENGTH_CONFIGS[config]).eval()
    generator = torch.Generator().manual_seed(1)
    x_enc, x_dec = random_inputs(model, generator)
    mask_keys, ft_masks, attn_rows = all_masks(model)
    full_ft, full_rows = ft_masks[-1:], attn_rows[-1:]  # the last key shows all feature groups
    original = run(model, batch(x_enc, 1), batch(x_dec, 1), full_ft, full_rows)
    for name in model.feature_names:
        changed_enc, changed_dec = random_inputs(model, generator)
        x_enc_new = {**x_enc, name: changed_enc[name]}
        x_dec_new = {**x_dec, name: changed_dec[name]} if name in x_dec else x_dec
        output = run(model, batch(x_enc_new, 1), batch(x_dec_new, 1), full_ft, full_rows)
        assert not torch.allclose(output, original, atol=1e-5), name


@pytest.mark.parametrize("output_dimensions", [1, 3])
@pytest.mark.parametrize("config", LENGTH_CONFIGS)
def test_output_shape(config, output_dimensions):
    model = make_model(output_dimensions=output_dimensions, **LENGTH_CONFIGS[config]).eval()
    x_enc, x_dec = random_inputs(model, torch.Generator().manual_seed(2))
    mask_keys, ft_masks, attn_rows = all_masks(model)
    output = run(model, batch(x_enc, len(mask_keys)), batch(x_dec, len(mask_keys)), ft_masks, attn_rows)
    expected = (len(mask_keys), model.get_horizon()) + ((output_dimensions,) if output_dimensions > 1 else ())
    assert output.shape == expected
