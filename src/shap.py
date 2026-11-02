from typing import List, Optional

import torch
import random
import numpy as np
import math
from tqdm import tqdm
from scipy.special import binom


def binary(num: int, digits: int = 7):
    return [bit == "1" for bit in bin(num)[2:].zfill(digits)]


def mask_day(attn_mask: torch.Tensor, day: int):
    attn_mask = attn_mask.clone()
    start = day * 24
    end = start + 24
    attn_mask[:, start:end] = -torch.inf
    return attn_mask


def mask_feature(ft_mask: torch.Tensor, feature: int):
    ft_mask = ft_mask.clone()
    ft_mask[feature] = 0.0
    return ft_mask


def get_full_masks(n_features, device="cuda"):
    ft_mask = torch.Tensor([1.0] * n_features).to(device)
    attn_mask = torch.zeros((168, 168)).to(device)
    return ft_mask, attn_mask


def get_random_masks(n_features, device="cuda"):
    ft_mask = torch.rand(n_features).to(device)
    attn_mask = torch.rand(168, 168).to(device)
    return ft_mask, attn_mask


def get_all_subset_masks(n_exogenous_features: int = 4, n_days: int = 7, device: str = "cuda"):
    n_mask_features = n_exogenous_features + n_days
    n_masks = 2 ** n_mask_features
    mask_dict = {}
    for i in range(n_masks):
        include_features = binary(i, digits=n_mask_features)
        ft_mask, attn_mask = get_full_masks(n_exogenous_features + 1)
        for j in range(n_exogenous_features):
            if not include_features[j]:
                ft_mask = mask_feature(ft_mask, j + 1)
        for j in range(n_days):
            if not include_features[n_exogenous_features + j]:
                attn_mask = mask_day(attn_mask, j)
        key = tuple(include_features)
        ft_mask = ft_mask.to(device)
        attn_mask = attn_mask.to(device)
        mask_dict[key] = (ft_mask, attn_mask)
    return mask_dict


def get_owen_masks(n_features, device: str = "cuda"):
    masks = get_all_subset_masks(n_features - 1, device=device)
    for key in masks:
        n_active_features = np.sum(key[:-7])
        n_active_days = np.sum(key[-7:])
        if n_active_days == 0 and n_active_features > 0:
            # attend all days and mask out the load feature instead
            ft_mask, attn_mask = masks[key]
            attn_mask = torch.zeros_like(attn_mask)
            ft_mask[0] = 0.0
            masks[key] = (ft_mask, attn_mask)
    return masks


def get_owen_masks_for_keys(keys: torch.Tensor, n_features: int, n_days: int = 7):
    # vectorized version of get_owen_masks for a batch of boolean mask keys with shape (batch, n_features - 1 + n_days)
    n_exogenous_features = n_features - 1
    feature_keys = keys[:, :n_exogenous_features]
    day_keys = keys[:, n_exogenous_features:]
    batch_size = keys.shape[0]
    ft_masks = torch.ones((batch_size, n_features), device=keys.device)
    ft_masks[:, 1:] = feature_keys.float()
    visible_hours = day_keys.repeat_interleave(24, dim=1)
    attn_masks = torch.zeros((batch_size, n_days * 24), device=keys.device)
    attn_masks[~visible_hours] = -torch.inf
    attn_masks = attn_masks[:, None, :].repeat(1, n_days * 24, 1)
    # attend all days and mask out the load feature instead
    no_days = ~day_keys.any(dim=1) & feature_keys.any(dim=1)
    attn_masks[no_days] = 0.0
    ft_masks[no_days, 0] = 0.0
    return ft_masks, attn_masks


def get_random_owen_masks(
        batch_size: int,
        n_features: int,
        p_dropout: float = 0.5,
        n_days: int = 7,
        device: str = "cuda"
):
    # samples entries of get_owen_masks(n_features) without generating all masks,
    # each feature group is masked with probability p_dropout
    keys = torch.rand((batch_size, n_features - 1 + n_days), device=device) < (1 - p_dropout)
    return get_owen_masks_for_keys(keys, n_features, n_days)


_OWEN_MASK_CACHE = {}


def get_owen_mask_tensors(n_features: int, n_days: int = 7, device: str = "cuda"):
    # all entries of get_owen_masks(n_features), in the same order, as mask keys and stacked mask tensors;
    # computed once per number of features and device and then reused
    cache_key = (n_features, n_days, str(device))
    if cache_key not in _OWEN_MASK_CACHE:
        n_mask_features = n_features - 1 + n_days
        mask_keys = [tuple(binary(i, digits=n_mask_features)) for i in range(2 ** n_mask_features)]
        ft_masks, attn_masks = get_owen_masks_for_keys(torch.tensor(mask_keys, device=device), n_features, n_days)
        _OWEN_MASK_CACHE[cache_key] = (mask_keys, ft_masks, attn_masks)
    return _OWEN_MASK_CACHE[cache_key]


def shap_weight(n_features, coalition_size):
    return 1 / binom(n_features - 1, coalition_size) / n_features


def owen_weight(n_features, days_size, nondays_size, is_day):
    n_coalitions = n_features - 6
    if is_day:
        coas_weight = shap_weight(n_coalitions, nondays_size)
        internal_weight = shap_weight(7, days_size - 1)
        return coas_weight * internal_weight
    else:
        if 0 < days_size < 7:
            return 0
        else:
            active_coalitions = nondays_size + (1 if days_size == 7 else 0)
            coas_weight = shap_weight(n_coalitions, active_coalitions - 1)
            internal_weight = shap_weight(1, 0)
            return coas_weight * internal_weight


_SHAP_COEFFICIENT_CACHE = {}


def get_shap_coefficients(mask_length: int, owen_values: bool = True, device: str = "cuda") -> torch.Tensor:
    # matrix C with shape (2 ** mask_length, mask_length), such that C.T @ predictions gives the SHAP (or Owen) values,
    # where row i of predictions is the prediction with the mask key binary(i);
    # each marginal contribution weight * (prediction(mask) - prediction(mask without feature)) adds
    # +weight to C[mask, feature] and -weight to C[mask without feature, feature].
    # Computed once per mask length and device and then reused.
    cache_key = (mask_length, owen_values, str(device))
    if cache_key not in _SHAP_COEFFICIENT_CACHE:
        coefficients = np.zeros((2 ** mask_length, mask_length))
        for i in range(1, 2 ** mask_length):
            mask = binary(i, digits=mask_length)
            coalition_size = np.sum(mask) - 1
            nondays_size = np.sum(mask[:-7])
            days_size = np.sum(mask[-7:])
            for feature in range(mask_length):
                if mask[feature]:
                    if owen_values:
                        is_day = feature >= mask_length - 7
                        weight = owen_weight(mask_length, days_size, nondays_size, is_day)
                    else:
                        weight = shap_weight(mask_length, coalition_size)
                    other_i = i ^ (1 << (mask_length - 1 - feature))
                    coefficients[i, feature] += weight
                    coefficients[other_i, feature] -= weight
        _SHAP_COEFFICIENT_CACHE[cache_key] = torch.tensor(coefficients, dtype=torch.float64, device=device)
    return _SHAP_COEFFICIENT_CACHE[cache_key]


def get_random_mask(mask_length, p_dropout):
    return tuple(random.random() < (1 - p_dropout) for _ in range(mask_length))


def predict_masked(model, x_enc, x_dec, get_masks, n_masks: int, batch_size: int = 128) -> torch.Tensor:
    # predictions for one sample under n_masks masks, evaluated in batches;
    # get_masks(start, end) returns the feature masks and attention masks with the indices start to end - 1.
    # Returns a tensor with shape (n_masks, horizon).
    x_enc_batch_full = {ft: x_enc[ft].repeat(batch_size, 1) for ft in x_enc}
    x_dec_batch_full = {ft: x_dec[ft].repeat(batch_size, 1) for ft in x_dec}
    predictions = []
    with torch.no_grad():
        for batch_start in tqdm(range(0, n_masks, batch_size), disable=n_masks <= batch_size):
            batch_end = min(n_masks, batch_start + batch_size)
            n = batch_end - batch_start
            x_enc_batch = {ft: x_enc_batch_full[ft][:n, :] for ft in x_enc}
            x_dec_batch = {ft: x_dec_batch_full[ft][:n, :] for ft in x_dec}
            ft_masks, attn_masks = get_masks(batch_start, batch_end)
            predictions.append(model(x_enc_batch, x_dec_batch, ft_masks, ft_masks[:, 1:], attn_masks))
    return torch.cat(predictions)


def _build_explanation(model, prediction, base, feature_impacts):
    # names the rows of feature_impacts: the exogenous features of the model, followed by "day 1" to "day 7"
    n_exogenous_features = len(model.feature_names) - 1
    shap_values = {}
    for i in range(len(feature_impacts)):
        if i < n_exogenous_features:
            feature = model.feature_names[i + 1]
        else:
            feature = f"day {i - n_exogenous_features + 1}"
        shap_values[feature] = feature_impacts[i]
    return {
        "prediction": prediction,
        "base": base,
        "shap_values": shap_values
    }


def explain_sample(model, sample):
    # exact Owen values, evaluating the model with all feature group subsets
    OWEN_VALUES = True

    x_enc = sample["x_enc"]
    x_dec = sample["x_dec"]

    mask_keys, all_ft_masks, all_attn_masks = get_owen_mask_tensors(n_features=len(x_enc), device=model.get_device())
    MASK_LENGTH = len(x_enc) + 6

    # rows of predictions are in the order of mask_keys, i.e. row i belongs to the mask key binary(i)
    predictions = predict_masked(
        model, x_enc, x_dec,
        get_masks=lambda start, end: (all_ft_masks[start:end], all_attn_masks[start:end]),
        n_masks=len(mask_keys)
    ).double()
    coefficients = get_shap_coefficients(MASK_LENGTH, owen_values=OWEN_VALUES, device=predictions.device)
    feature_impacts = (coefficients.T @ predictions).cpu().numpy()
    dummy_prediction = predictions[0].cpu().numpy()  # no feature group visible
    prediction = predictions[-1].cpu().numpy()  # all feature groups visible
    return _build_explanation(model, prediction, dummy_prediction, feature_impacts)


def sample_owen_permutation(n_features: int, rng: random.Random, n_days: int = 7) -> List[int]:
    # random order of the mask positions (exogenous features 0 to n_features - 2, then the days),
    # in which the days are consecutive: the exogenous features and the load (all days together)
    # are shuffled, and the days are shuffled within the load
    n_exogenous_features = n_features - 1
    units = list(range(n_exogenous_features)) + ["load"]
    rng.shuffle(units)
    days = list(range(n_exogenous_features, n_exogenous_features + n_days))
    rng.shuffle(days)
    order = []
    for unit in units:
        order.extend(days if unit == "load" else [unit])
    return order


def explain_sample_permutation(model, sample, n_permutations: int = 10, seed: Optional[int] = None):
    # approximates the Owen values of explain_sample by sampling permutations that keep the days together.
    # Like shap's PermutationExplainer, each permutation is evaluated forward (adding the feature groups one by one)
    # and backward (removing them in the same order), so each permutation needs 2 * mask_length + 1 model evaluations.
    # base + sum of all values = prediction holds exactly for any number of permutations.
    n_days = 7
    x_enc = sample["x_enc"]
    x_dec = sample["x_dec"]
    n_features = len(x_enc)
    mask_length = n_features - 1 + n_days
    n_steps = 2 * mask_length + 1
    device = model.get_device()

    rng = random.Random(seed)
    orders = torch.tensor([sample_owen_permutation(n_features, rng, n_days) for _ in range(n_permutations)],
                          device=device)
    # positions[p, g] = step at which feature group g is added in permutation p
    positions = torch.empty_like(orders)
    positions.scatter_(1, orders, torch.arange(mask_length, device=device).expand(n_permutations, -1))
    # keys[p, s, g]: whether group g is visible at step s; steps 0 to mask_length add the groups (forward),
    # steps mask_length to 2 * mask_length remove them in the same order (backward)
    steps = torch.arange(n_steps, device=device)[None, :, None]
    positions = positions[:, None, :]
    keys = torch.where(steps <= mask_length, positions < steps, positions >= steps - mask_length)
    keys = keys.reshape(-1, mask_length)

    predictions = predict_masked(
        model, x_enc, x_dec,
        get_masks=lambda start, end: get_owen_masks_for_keys(keys[start:end], n_features, n_days),
        n_masks=len(keys)
    ).double()
    predictions = predictions.reshape(n_permutations, n_steps, -1)

    forward = predictions[:, 1:mask_length + 1] - predictions[:, :mask_length]
    backward = predictions[:, mask_length:-1] - predictions[:, mask_length + 1:]
    # the k-th forward and backward difference belong to the group orders[:, k]
    feature_impacts = torch.zeros((mask_length, predictions.shape[-1]), dtype=torch.float64, device=device)
    feature_impacts.index_add_(0, orders.reshape(-1), (forward + backward).reshape(-1, predictions.shape[-1]))
    feature_impacts /= 2 * n_permutations

    dummy_prediction = predictions[0, 0].cpu().numpy()  # no feature group visible
    prediction = predictions[0, mask_length].cpu().numpy()  # all feature groups visible
    return _build_explanation(model, prediction, dummy_prediction, feature_impacts.cpu().numpy())

