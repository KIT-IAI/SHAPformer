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


def get_owen_masks(n_features):
    masks = get_all_subset_masks(n_features - 1)
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


def get_random_mask(mask_length, p_dropout):
    return tuple(random.random() < (1 - p_dropout) for _ in range(mask_length))


def explain_sample(model, sample):
    OWEN_VALUES = True
    BATCH_SIZE = 128

    x_enc = sample["x_enc"]
    x_dec = sample["x_dec"]

    masks = get_owen_masks(n_features=len(x_enc))
    mask_keys = list(masks)
    MASK_LENGTH = len(x_enc) + 6

    predictions = {}
    n_batches = math.ceil(len(masks) / BATCH_SIZE)

    x_enc_batch_full = {ft: x_enc[ft].repeat(BATCH_SIZE, 1) for ft in x_enc}
    x_dec_batch_full = {ft: x_dec[ft].repeat(BATCH_SIZE, 1) for ft in x_dec}

    with torch.no_grad():
        for batch_i in tqdm(range(n_batches)):
            batch_start = batch_i * BATCH_SIZE
            batch_end = min(len(masks), batch_start + BATCH_SIZE)
            batch_size = batch_end - batch_start
            x_enc_batch = {ft: x_enc_batch_full[ft][:batch_size, :] for ft in x_enc}
            x_dec_batch = {ft: x_dec_batch_full[ft][:batch_size, :] for ft in x_dec}
            ft_masks = []
            attn_masks = []
            for i in range(batch_start, batch_end):
                mask_key = mask_keys[i]
                ft_mask, attn_mask = masks[mask_key]
                ft_masks.append(ft_mask)
                attn_masks.append(attn_mask)
            ft_masks = torch.stack(ft_masks)
            attn_masks = torch.stack(attn_masks)
            output = model(x_enc_batch, x_dec_batch, ft_masks, ft_masks[:, 1:], attn_masks)
            for i, mask_key in enumerate(mask_keys[batch_start:batch_end]):
                predictions[mask_key] = output[i, :]

    feature_impacts = {i: [] for i in range(MASK_LENGTH)}

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

    for i in range(1, 2 ** MASK_LENGTH):
        mask = binary(i, digits=MASK_LENGTH)
        coalition_size = np.sum(mask) - 1
        if OWEN_VALUES:
            nondays_size = np.sum(mask[:-7])
            days_size = np.sum(mask[-7:])
        prediction = predictions[tuple(mask)]
        for feature in range(MASK_LENGTH):
            if mask[feature]:
                other_mask = mask.copy()
                other_mask[feature] = False
                other_prediction = predictions[tuple(other_mask)]
                impact = prediction - other_prediction
                if OWEN_VALUES:
                    is_day = feature >= MASK_LENGTH - 7
                    weight = owen_weight(MASK_LENGTH, days_size, nondays_size, is_day)
                else:
                    weight = shap_weight(MASK_LENGTH, coalition_size)
                feature_impacts[feature].append(impact * weight)

    dummy_prediction = predictions[tuple([False] * MASK_LENGTH)].cpu().numpy()
    impact_sum = dummy_prediction.copy()
    ft_importances = []

    for feature in range(MASK_LENGTH):
        feature_impact = torch.stack(feature_impacts[feature]).cpu().numpy()
        feature_impact = np.sum(feature_impact, axis=0)
        feature_impacts[feature] = feature_impact
        # plt.plot(feature_impact, color="gray", alpha=0.5)
        impact_sum += feature_impact
        ft_importance = np.mean(np.abs(feature_impact))
        #print(f"feature {feature} importance: {ft_importance}")
        ft_importances.append(ft_importance)

    prediction = predictions[tuple([True] * MASK_LENGTH)].cpu().numpy()
    #print("difference:", np.abs(prediction - impact_sum).sum())

    """for ft in x_enc:
        x_enc[ft] = x_enc[ft].cpu()

    for ft in x_dec:
        x_dec[ft] = x_dec[ft].cpu()"""

    shap_values = {
        #"base": dummy_prediction.cpu().numpy()
    }

    for i in range(MASK_LENGTH):
        if i < len(model.feature_names) - 1:
            feature = model.feature_names[i + 1]
        else:
            feature = f"day {i - len(model.feature_names) + 2}"
        shap_values[feature] = feature_impacts[i]

    explanation = {
        "prediction": prediction,
        "base": dummy_prediction,
        "shap_values": shap_values
    }

    return explanation
