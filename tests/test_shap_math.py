"""The coefficient matrix of get_shap_coefficients against brute-force Owen and Shapley values.

Mask positions: the exogenous features first, then the periods of the past target. Row i of the value table
belongs to the mask key binary(i), i.e. the first position is the most significant bit.
"""
import itertools
from math import factorial

import numpy as np
import pytest

from src.shap import binary, get_shap_coefficients

# (number of exogenous features, number of periods)
SIZES = [(0, 1), (1, 1), (2, 1), (0, 3), (1, 2), (2, 3), (3, 4), (2, 7)]
N_OUTPUTS = 3  # values per mask, like a forecast with three time steps


def key_index(players, mask_length: int) -> int:
    return sum(1 << (mask_length - 1 - p) for p in players)


def random_values(mask_length: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(2 ** mask_length, N_OUTPUTS))


def subsets(items):
    items = list(items)
    for size in range(len(items) + 1):
        yield from itertools.combinations(items, size)


def shapley_weight(n: int, size: int) -> float:
    return factorial(size) * factorial(n - size - 1) / factorial(n)


def brute_force_shapley(values: np.ndarray, mask_length: int) -> np.ndarray:
    result = np.zeros((mask_length, values.shape[1]))
    for player in range(mask_length):
        others = [p for p in range(mask_length) if p != player]
        for coalition in subsets(others):
            gain = values[key_index(coalition + (player,), mask_length)] - values[key_index(coalition, mask_length)]
            result[player] += shapley_weight(mask_length, len(coalition)) * gain
    return result


def brute_force_owen(values: np.ndarray, n_exogenous: int, n_periods: int) -> np.ndarray:
    # coalition structure: every exogenous feature is its own union, the periods form one union (the past target)
    mask_length = n_exogenous + n_periods
    unions = [[p] for p in range(n_exogenous)] + [list(range(n_exogenous, mask_length))]
    m = len(unions)
    result = np.zeros((mask_length, values.shape[1]))
    for k, union in enumerate(unions):
        other_unions = [u for j, u in enumerate(unions) if j != k]
        for player in union:
            rest = [p for p in union if p != player]
            for chosen_unions in subsets(range(len(other_unions))):
                outside = [p for j in chosen_unions for p in other_unions[j]]
                union_weight = shapley_weight(m, len(chosen_unions))
                for inside in subsets(rest):
                    weight = union_weight * shapley_weight(len(union), len(inside))
                    coalition = tuple(outside) + inside
                    gain = (values[key_index(coalition + (player,), mask_length)]
                            - values[key_index(coalition, mask_length)])
                    result[player] += weight * gain
    return result


def owen_values(values: np.ndarray, n_exogenous: int, n_periods: int) -> np.ndarray:
    coefficients = get_shap_coefficients(n_exogenous + n_periods, owen_values=True, device="cpu",
                                         n_periods=n_periods).numpy()
    return coefficients.T @ values


def test_binary_is_most_significant_bit_first():
    assert binary(1, digits=3) == [False, False, True]
    assert binary(4, digits=3) == [True, False, False]
    assert binary(0, digits=2) == [False, False]


@pytest.mark.parametrize("n_exogenous, n_periods", SIZES)
def test_owen_values_match_brute_force(n_exogenous, n_periods):
    values = random_values(n_exogenous + n_periods)
    np.testing.assert_allclose(owen_values(values, n_exogenous, n_periods),
                               brute_force_owen(values, n_exogenous, n_periods), atol=1e-10)


@pytest.mark.parametrize("mask_length", [1, 2, 3, 5, 7])
def test_shapley_values_match_brute_force(mask_length):
    values = random_values(mask_length, seed=1)
    coefficients = get_shap_coefficients(mask_length, owen_values=False, device="cpu", n_periods=1).numpy()
    np.testing.assert_allclose(coefficients.T @ values, brute_force_shapley(values, mask_length), atol=1e-10)


@pytest.mark.parametrize("n_exogenous", [0, 1, 2, 4])
def test_owen_values_with_a_single_period_are_shapley_values(n_exogenous):
    mask_length = n_exogenous + 1
    values = random_values(mask_length, seed=2)
    np.testing.assert_allclose(owen_values(values, n_exogenous, 1), brute_force_shapley(values, mask_length),
                               atol=1e-10)


@pytest.mark.parametrize("n_exogenous, n_periods", SIZES)
def test_efficiency(n_exogenous, n_periods):
    # the values sum to the difference between the prediction with all and with no feature groups
    values = random_values(n_exogenous + n_periods, seed=3)
    np.testing.assert_allclose(owen_values(values, n_exogenous, n_periods).sum(axis=0), values[-1] - values[0],
                               atol=1e-10)


@pytest.mark.parametrize("n_exogenous, n_periods", SIZES)
def test_additive_value_function(n_exogenous, n_periods):
    # if every position adds a fixed amount when visible, it gets exactly that amount
    mask_length = n_exogenous + n_periods
    effects = np.random.default_rng(4).normal(size=(mask_length, N_OUTPUTS))
    offset = np.random.default_rng(5).normal(size=N_OUTPUTS)
    values = np.array([offset + np.array(binary(i, mask_length), dtype=float) @ effects
                       for i in range(2 ** mask_length)])
    np.testing.assert_allclose(owen_values(values, n_exogenous, n_periods), effects, atol=1e-10)


@pytest.mark.parametrize("n_exogenous, n_periods", [(2, 3), (3, 4), (1, 2)])
@pytest.mark.parametrize("dummy", ["exogenous", "period"])
def test_dummy_position_gets_zero(n_exogenous, n_periods, dummy):
    # a position that never changes the value gets 0
    mask_length = n_exogenous + n_periods
    position = 0 if dummy == "exogenous" else mask_length - 1
    values = random_values(mask_length, seed=6)
    for i in range(2 ** mask_length):
        if binary(i, mask_length)[position]:
            values[i] = values[i ^ (1 << (mask_length - 1 - position))]
    result = owen_values(values, n_exogenous, n_periods)
    np.testing.assert_allclose(result[position], 0, atol=1e-10)


def symmetrize(values: np.ndarray, a: int, b: int, mask_length: int) -> np.ndarray:
    # value function that is unchanged when the positions a and b are swapped
    def swap(i):
        bits = binary(i, mask_length)
        bits[a], bits[b] = bits[b], bits[a]
        return key_index([p for p, bit in enumerate(bits) if bit], mask_length)
    return np.array([(values[i] + values[swap(i)]) / 2 for i in range(2 ** mask_length)])


@pytest.mark.parametrize("pair", ["exogenous", "periods"])
def test_symmetric_positions_get_equal_values(pair):
    n_exogenous, n_periods = 3, 4
    mask_length = n_exogenous + n_periods
    a, b = (0, 2) if pair == "exogenous" else (n_exogenous + 1, n_exogenous + 3)
    values = symmetrize(random_values(mask_length, seed=7), a, b, mask_length)
    result = owen_values(values, n_exogenous, n_periods)
    np.testing.assert_allclose(result[a], result[b], atol=1e-10)


def test_coefficients_are_cached():
    first = get_shap_coefficients(4, device="cpu", n_periods=2)
    assert get_shap_coefficients(4, device="cpu", n_periods=2) is first
    assert get_shap_coefficients(4, device="cpu", n_periods=3) is not first
