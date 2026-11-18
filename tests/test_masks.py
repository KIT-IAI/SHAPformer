"""Feature and attention masks of the feature groups (exogenous features and periods of the past target)."""
import random

import pytest
import torch

from src.shap import (DAY_LENGTH, as_period_lengths, binary, expand_attention_masks, get_owen_mask_tensors,
                      get_owen_masks, get_owen_masks_for_keys, get_period_names, get_random_owen_masks,
                      sample_owen_permutation)

INF = float("inf")
N_FEATURES = 3  # the target and two exogenous features


def keys(*rows):
    return torch.tensor(rows, dtype=torch.bool)


def test_as_period_lengths():
    assert as_period_lengths(3) == (DAY_LENGTH,) * 3
    assert as_period_lengths((6, 3, 2, 1)) == (6, 3, 2, 1)
    assert as_period_lengths([4, 4]) == (4, 4)


def test_period_names():
    assert get_period_names("load", 3) == ["period 1", "period 2", "period 3"]
    assert get_period_names("load", 1) == ["load"]


def test_uneven_periods_hide_their_time_steps():
    # visible: second exogenous feature, periods 1 and 3 (of lengths 6, 3, 2, 1)
    ft_masks, attn_masks = get_owen_masks_for_keys(keys([False, True, True, False, True, False]), N_FEATURES,
                                                   (6, 3, 2, 1))
    assert ft_masks.tolist() == [[1.0, 0.0, 1.0]]
    expected_row = [0.0] * 6 + [-INF] * 3 + [0.0] * 2 + [-INF]
    assert attn_masks.shape == (1, 12, 12)
    for row in attn_masks[0]:
        assert row.tolist() == expected_row


def test_rows_option_returns_one_row_per_mask():
    batch = keys([True, True, False, True], [False, False, True, False])
    ft_rows, attn_rows = get_owen_masks_for_keys(batch, N_FEATURES, (4, 4), rows=True)
    ft_full, attn_full = get_owen_masks_for_keys(batch, N_FEATURES, (4, 4))
    assert attn_rows.shape == (2, 8)
    assert torch.equal(ft_rows, ft_full)
    assert torch.equal(expand_attention_masks(attn_rows), attn_full)


def test_all_periods_hidden_hides_the_target_as_a_feature():
    # with some exogenous feature visible but no period, all time steps are attended and the target is masked
    ft_masks, attn_masks = get_owen_masks_for_keys(keys([True, False, False, False, False]), N_FEATURES,
                                                   (4, 4, 4), rows=True)
    assert ft_masks.tolist() == [[0.0, 1.0, 0.0]]
    assert attn_masks.tolist() == [[0.0] * 12]


def test_empty_coalition_hides_all_time_steps():
    ft_masks, attn_masks = get_owen_masks_for_keys(keys([False] * 5), N_FEATURES, (4, 4, 4), rows=True)
    assert ft_masks.tolist() == [[1.0, 0.0, 0.0]]
    assert attn_masks.tolist() == [[-INF] * 12]


@pytest.mark.parametrize("target_visible", [True, False])
def test_single_period_masks_the_target_as_a_feature(target_visible):
    ft_masks, attn_masks = get_owen_masks_for_keys(keys([True, False, target_visible]), N_FEATURES, (12,),
                                                   rows=True)
    assert ft_masks.tolist() == [[float(target_visible), 1.0, 0.0]]
    assert attn_masks.tolist() == [[0.0] * 12]


@pytest.mark.parametrize("n_exogenous, n_days", [(1, 2), (2, 3)])
def test_vectorized_masks_match_the_mask_dictionary(n_exogenous, n_days):
    # get_owen_masks (one mask at a time, periods of DAY_LENGTH) and the vectorized version agree for every key
    device = "cpu"
    masks = get_owen_masks(n_exogenous + 1, device=device, n_days=n_days)
    for key, (ft_mask, attn_mask) in masks.items():
        ft_masks, attn_masks = get_owen_masks_for_keys(keys(list(key)), n_exogenous + 1, n_days)
        assert torch.equal(ft_masks[0], ft_mask), key
        assert torch.equal(attn_masks[0], attn_mask), key


@pytest.mark.parametrize("period_lengths", [(4, 4, 4), (6, 3, 2, 1), (12,)])
def test_mask_tensors_are_ordered_by_binary_keys(period_lengths):
    mask_keys, ft_masks, attn_rows = get_owen_mask_tensors(N_FEATURES, period_lengths, device="cpu")
    mask_length = N_FEATURES - 1 + len(period_lengths)
    assert len(mask_keys) == 2 ** mask_length
    for i, key in enumerate(mask_keys):
        assert list(key) == binary(i, mask_length)
    expected_ft, expected_rows = get_owen_masks_for_keys(keys(*mask_keys), N_FEATURES, period_lengths, rows=True)
    assert torch.equal(ft_masks, expected_ft)
    assert torch.equal(attn_rows, expected_rows)


def test_mask_tensors_are_cached():
    first = get_owen_mask_tensors(N_FEATURES, (4, 4), device="cpu")
    assert get_owen_mask_tensors(N_FEATURES, (4, 4), device="cpu") is first
    assert get_owen_mask_tensors(N_FEATURES, (2, 6), device="cpu") is not first


@pytest.mark.parametrize("period_lengths", [(4, 4, 4), (6, 3, 2, 1), (12,)])
def test_random_masks_shapes_and_extremes(period_lengths):
    input_length = sum(period_lengths)
    ft_masks, attn_masks = get_random_owen_masks(5, N_FEATURES, p_dropout=0.5, period_lengths=period_lengths,
                                                 device="cpu")
    assert ft_masks.shape == (5, N_FEATURES)
    assert attn_masks.shape == (5, input_length, input_length)

    ft_masks, attn_masks = get_random_owen_masks(5, N_FEATURES, p_dropout=0.0, period_lengths=period_lengths,
                                                 device="cpu")
    assert torch.all(ft_masks == 1)
    assert torch.all(attn_masks == 0)

    ft_masks, attn_masks = get_random_owen_masks(5, N_FEATURES, p_dropout=1.0, period_lengths=period_lengths,
                                                 device="cpu")
    if len(period_lengths) == 1:
        assert torch.all(ft_masks == 0)
        assert torch.all(attn_masks == 0)
    else:
        assert ft_masks.tolist() == [[1.0, 0.0, 0.0]] * 5
        assert torch.all(attn_masks == -INF)


@pytest.mark.parametrize("n_features, n_periods", [(3, 4), (1, 3), (4, 1), (2, 7)])
def test_permutations_keep_the_periods_together(n_features, n_periods):
    rng = random.Random(0)
    mask_length = n_features - 1 + n_periods
    period_positions = set(range(n_features - 1, mask_length))
    for _ in range(50):
        order = sample_owen_permutation(n_features, rng, n_periods)
        assert sorted(order) == list(range(mask_length))
        where = [i for i, position in enumerate(order) if position in period_positions]
        assert where == list(range(where[0], where[0] + n_periods))


def test_permutations_are_reproducible_with_a_seed():
    first = [sample_owen_permutation(3, random.Random(42), 4) for _ in range(5)]
    second = [sample_owen_permutation(3, random.Random(42), 4) for _ in range(5)]
    assert first == second
