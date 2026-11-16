import copy
import math
import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm
from .transformer_network import PositionalEncoding
from ..shap import (DAY_LENGTH, explain_sample, explain_sample_permutation, get_full_masks,
                    get_random_owen_masks, get_period_names)
from ..transformer.transformer import Transformer

# default lengths of the past window (input) and of the forecast (horizon), in time steps
INPUT_LENGTH = 168
HORIZON = 168
INSTANCE_NORMS = ("none", "mean", "RevIN")
REVIN_EPS = 1e-5


class CategoricalEmbedding(nn.Module):
    def __init__(self, n_embeddings, dim: int):
        super(CategoricalEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=dim)

    def forward(self, x):
        return self.embedding(x)


class LinearEmbedding(nn.Module):
    def __init__(self, dim: int):
        super(LinearEmbedding, self).__init__()
        self.embedding = nn.Linear(1, dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.embedding(x)


class FeatureEmbedding(nn.Module):
    def __init__(
            self,
            feature_names: List[str],
            categories: List[Optional[int]],
            dim: int,
            dummy_feature: bool = False
    ):
        super(FeatureEmbedding, self).__init__()
        self.feature_names = feature_names
        self.categories = categories
        self.dim = dim
        self.embeddings = nn.ParameterDict()
        self._initialize_embeddings()
        self.attention = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)
        self.dummy_feature = dummy_feature
        self.attention_weights = None
        self.n_with_nans = 0
        self.n_total = 0

    def _initialize_embeddings(self):
        for name, n_vals in zip(self.feature_names, self.categories):
            if n_vals is None:
                self.embeddings[name] = LinearEmbedding(self.dim)
            else:
                self.embeddings[name] = CategoricalEmbedding(n_vals, self.dim)

    def forward(self, x: Dict[str, torch.Tensor], mask: torch.Tensor):
        embeddings = []
        for name in self.embeddings:
            embd = self.embeddings[name](x[name])
            embeddings.append(embd)
        embeddings = torch.stack(embeddings, dim=-1)
        batch_size = embeddings.shape[0]
        seq_len = embeddings.shape[1]
        embeddings = embeddings * mask[:, None, None, :]
        n_active_features = torch.sum(mask, dim=-1)[:, None, None]
        n_active_features[n_active_features == 0] = 1
        embedding_sum = torch.sum(embeddings, dim=-1) / n_active_features
        query = embedding_sum.reshape((-1, 1, self.dim))
        embeddings = embeddings.transpose(2, 3)
        key = embeddings.reshape((-1, embeddings.shape[2], embeddings.shape[3]))
        if self.dummy_feature:
            dummy_mask = torch.zeros((batch_size, 1), device=mask.device)
            rows_without_features = torch.sum(mask, dim=1) == 0
            dummy_mask[rows_without_features, :] = 1
            mask = torch.concatenate((dummy_mask, mask), dim=1)
            dummy_key = torch.zeros((key.shape[0], 1, self.dim), device=key.device)
            key = torch.cat((dummy_key, key), dim=1)
        attn_mask = 1 - mask
        attn_mask[attn_mask == 1.0] = -torch.inf
        attn_mask = attn_mask.repeat_interleave(seq_len, dim=0)[:, None, :]
        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=key,
            attn_mask=attn_mask
        )
        self.attention_weights = attn_weights
        attn_output = attn_output.reshape((batch_size, seq_len, self.dim))
        return attn_output


class SHAPformer(nn.Module):
    def __init__(
            self,
            target: str,
            feature_categories: Dict[str, Optional[int]],
            d_model: int,
            n_layers: int,
            n_heads: int,
            output_dimensions: int = 1,
            embedding: str = "attention",
            instance_norm: str = "none",
            input_length: Optional[int] = None,
            horizon: int = HORIZON,
            period_length: Union[int, Iterable[int]] = DAY_LENGTH
    ):
        # input_length: number of past time steps the model gets (with the target and the covariates);
        # default: the sum of period_length if it is a sequence, else INPUT_LENGTH (168).
        # horizon: number of time steps to forecast (with the covariates known for them).
        # period_length: how the past target is split into groups ("periods") for the explanations:
        # - an int p: input_length / p periods of p time steps each (input_length must be a multiple of p),
        # - a sequence of ints: the lengths of the periods from the oldest to the most recent one,
        #   which must add up to input_length, e.g. (336, 168, 144, 24) for four weeks of hourly data.
        # With a single period (period_length = input_length), no time steps are masked: the past target is
        # masked like a covariate (the whole series at once), and its group is named after the target.
        # instance_norm: normalization of the target per sample, with statistics of its past window
        # (always the full past window, also when feature groups are masked, so explanations stay additive):
        # "none": only the global scaling of the target,
        # "mean": divide by the mean absolute value of the past window (as in Chronos),
        # "RevIN": subtract the mean and divide by the standard deviation of the past window, followed by
        #          a learnable affine transformation (reversible instance normalization, Kim et al. 2022).
        # With instance normalization, the target is not scaled globally.
        # Since the statistics of the past target are used for all masks, their effect (the level and scale of the
        # past target) is part of the base value of an explanation and not attributed to the past days.
        super(SHAPformer, self).__init__()
        input_length, period_lengths = self._resolve_periods(input_length, period_length)
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, but is {horizon}.")
        self.input_length = input_length
        self.horizon = horizon
        self.period_lengths = period_lengths
        if instance_norm not in INSTANCE_NORMS:
            raise ValueError(f"Unknown instance_norm '{instance_norm}', must be one of {INSTANCE_NORMS}.")
        self.instance_norm = instance_norm
        if instance_norm == "RevIN":
            self.revin_weight = nn.Parameter(torch.ones(1))
            self.revin_bias = nn.Parameter(torch.zeros(1))
        encoder_feature_names = decoder_feature_names = list(feature_categories)
        encoder_feature_names = [target] + encoder_feature_names
        encoder_categories = decoder_categories = [feature_categories[ft] for ft in feature_categories]
        encoder_categories = [None] + encoder_categories
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        if embedding == "attention":
            # the dummy feature is needed when all encoder features of a time step are masked, which only happens
            # with a single period (the past target is masked as a feature, without masking time steps)
            self.encoder_embedding = FeatureEmbedding(
                feature_names=encoder_feature_names,
                categories=encoder_categories,
                dim=d_model,
                dummy_feature=True
            )
            self.decoder_embedding = FeatureEmbedding(
                feature_names=decoder_feature_names,
                categories=decoder_categories,
                dim=d_model,
                dummy_feature=True
            )
        self.positional_encoding = PositionalEncoding(d_model, max_len=max(1000, input_length, horizon))
        self.transformer = Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=d_model,
            batch_first=True
        )
        self.projection = nn.Linear(d_model, output_dimensions)
        self.relu = nn.ReLU()
        self.output_dimensions = output_dimensions
        self.cross_attention_weights = None
        self.feature_names = encoder_feature_names
        self.feature_categories = feature_categories
        self.continuous_features = [name for name, n_vals in zip(encoder_feature_names, encoder_categories)
                                    if n_vals is None]
        self.scalers: Dict[str, StandardScaler] = {}

    def fit(
            self,
            data: pd.DataFrame,
            validation_start,
            epochs: int = 100,
            batch_size: int = 128,
            learning_rate: float = 0.0001,
            p_dropout: float = 0.5,
            patience: int = 10,
            min_improvement: float = 0.00001,
            window_stride: int = 1
    ) -> "SHAPformer":
        # trains until the validation loss has not improved for `patience` epochs (or for at most `epochs` epochs),
        # then restores the parameters with the lowest validation loss.
        # window_stride: use only every window_stride-th window start (counted from the first row) for training and
        # validation, e.g. 4 for quarter-hourly data to use only windows that start at full hours.
        # The data may be incomplete (missing values or missing time steps): only windows of
        # input_length + horizon consecutive time steps without missing values are used.
        index = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data.index, utc=True)
        validation_start = pd.Timestamp(validation_start)
        if validation_start.tzinfo is None and index.tz is not None:
            validation_start = validation_start.tz_localize(index.tz)
        train_data = data[index < validation_start]
        n_train = len(train_data)
        self.scalers = {}
        for name in self.continuous_features:
            scaler = StandardScaler()  # ignores missing values
            scaler.fit(train_data[[name]].to_numpy())
            if name == self.feature_names[0] and self._instance_norm() != "none":
                # the target is normalized per sample in forward instead
                scaler.mean_, scaler.scale_, scaler.var_ = np.zeros(1), np.ones(1), np.ones(1)
            self.scalers[name] = scaler
        # validation samples forecast from validation_start onward, but may use inputs from before
        validation_data = data.iloc[n_train - self.get_input_length():]

        step = self._time_step(index)
        if window_stride < 1:
            raise ValueError(f"window_stride must be at least 1, but is {window_stride}.")
        # the stride is counted from the first row of data, for the training and the validation windows
        validation_offset = n_train - self.get_input_length()
        indices = [start for start in self._window_starts(train_data, index[:n_train], step)
                   if start % window_stride == 0]
        validation_indices = [start for start in self._window_starts(validation_data, index[validation_offset:], step)
                              if (validation_offset + start) % window_stride == 0]
        for name, starts, subset, offset in (("training", indices, train_data, 0),
                                             ("validation", validation_indices, validation_data, validation_offset)):
            n_windows = max(len(subset) - self._window_length() + 1, 0)
            if not starts:
                raise ValueError(f"The {name} data has no window of {self._window_length()} consecutive time steps "
                                 f"(step {step}) without missing values.")
            n_windows = sum(1 for start in range(n_windows) if (offset + start) % window_stride == 0)
            if len(starts) < n_windows:
                print(f"Using {len(starts)} of {n_windows} {name} windows; the others contain missing values "
                      f"or missing time steps.")

        # missing values are only in unused windows; replace them so that the tensors can be created
        train_data = self.scale(train_data).fillna(0)
        validation_data = self.scale(validation_data).fillna(0)

        device = self.get_device()
        series = self._to_tensors(train_data, device, self.feature_names)
        validation_series = self._to_tensors(validation_data, device, self.feature_names)
        n_batches = math.ceil(len(indices) / batch_size)
        n_features = len(self.feature_names)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()

        best_validation_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.state_dict())
        epochs_without_improvement = 0

        for epoch in range(epochs):
            random.shuffle(indices)
            total_loss = 0
            for batch_i in tqdm(range(n_batches)):
                optimizer.zero_grad()

                batch_indices = indices[batch_i * batch_size:(batch_i + 1) * batch_size]
                x_enc, x_dec, y = self._get_batch(series, batch_indices)

                ft_mask, attn_mask = get_random_owen_masks(len(batch_indices), n_features, p_dropout=p_dropout,
                                                           period_lengths=self.get_period_lengths(), device=device)

                output = self.forward(x_enc, x_dec, mask_enc=ft_mask, mask_dec=ft_mask[:, 1:], attn_mask=attn_mask)

                loss = self._loss(loss_function, output, y, x_enc)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            validation_loss = self._validation_loss(validation_series, validation_indices, loss_function, batch_size=batch_size)

            if best_validation_loss - validation_loss > min_improvement:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch + 1}/{epochs}, Training loss: {total_loss / n_batches}, "
                  f"Validation loss: {validation_loss} (best: {best_validation_loss}), "
                  f"Epochs without improvement: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement == patience:
                print("Early stopping.")
                break

        self.load_state_dict(best_state)
        self.eval()
        print(f"Restored the model from epoch {best_epoch} with validation loss {best_validation_loss}.")
        return self

    @staticmethod
    def _time_step(index: pd.DatetimeIndex) -> pd.Timedelta:
        # the most frequent difference between consecutive timestamps
        if len(index) < 2:
            raise ValueError("The data must have at least two rows.")
        return pd.Series(index[1:] - index[:-1]).mode()[0]

    def get_input_length(self) -> int:
        # models saved before the lengths were configurable have no attribute input_length
        return getattr(self, "input_length", INPUT_LENGTH)

    def get_horizon(self) -> int:
        return getattr(self, "horizon", HORIZON)

    @staticmethod
    def _resolve_periods(input_length: Optional[int], period_length) -> Tuple[int, Tuple[int, ...]]:
        # the input length and the lengths of the periods of the past target, from the oldest to the most recent
        if isinstance(period_length, (int, np.integer)):
            period_length = int(period_length)
            if input_length is None:
                input_length = INPUT_LENGTH
            if period_length <= 0 or input_length <= 0 or input_length % period_length != 0:
                raise ValueError(f"input_length must be a positive multiple of period_length, but input_length is "
                                 f"{input_length} and period_length {period_length}.")
            return input_length, (period_length,) * (input_length // period_length)
        period_lengths = tuple(int(length) for length in period_length)
        if not period_lengths or any(length <= 0 for length in period_lengths):
            raise ValueError(f"period_length must contain at least one length, all positive, but is {period_lengths}.")
        if input_length is None:
            input_length = sum(period_lengths)
        if sum(period_lengths) != input_length:
            raise ValueError(f"The period lengths {period_lengths} add up to {sum(period_lengths)}, "
                             f"but input_length is {input_length}.")
        return input_length, period_lengths

    def get_period_lengths(self) -> Tuple[int, ...]:
        # lengths of the periods of the past target, from the oldest to the most recent one;
        # models saved before periods were configurable use days of DAY_LENGTH time steps
        default = (DAY_LENGTH,) * (self.get_input_length() // DAY_LENGTH)
        return tuple(getattr(self, "period_lengths", default))

    def get_n_periods(self) -> int:
        return len(self.get_period_lengths())

    def get_periods(self, past_index) -> pd.DataFrame:
        # the periods of the past target with their names, lengths and first and last time step,
        # for the index of the past window (input_length rows)
        names = get_period_names(self.feature_names[0], self.get_n_periods())
        rows = []
        end = 0
        for name, length in zip(names, self.get_period_lengths()):
            start, end = end, end + length
            rows.append({"name": name, "length": length, "start": past_index[start], "end": past_index[end - 1]})
        return pd.DataFrame(rows).set_index("name")

    def _window_length(self) -> int:
        return self.get_input_length() + self.get_horizon()

    def _window_starts(self, data: pd.DataFrame, index: pd.DatetimeIndex, step: pd.Timedelta) -> List[int]:
        # start positions of the windows of input_length + horizon rows without missing values in the features,
        # whose timestamps are consecutive (`step` apart)
        n_windows = len(data) - self._window_length() + 1
        if n_windows <= 0:
            return []
        incomplete = data[self.feature_names].isna().any(axis=1).to_numpy()
        # breaks[i]: row i does not follow row i - 1 by one step
        breaks = np.concatenate(([False], np.asarray(index[1:] - index[:-1] != step)))
        incomplete_sum = np.concatenate(([0], np.cumsum(incomplete)))
        breaks_sum = np.concatenate(([0], np.cumsum(breaks)))
        starts = np.arange(n_windows)
        ends = starts + self._window_length()
        valid = (incomplete_sum[ends] == incomplete_sum[starts]) & (breaks_sum[ends] == breaks_sum[starts + 1])
        return starts[valid].tolist()

    def _validation_loss(
            self,
            series: Dict[str, torch.Tensor],
            indices: List[int],
            loss_function,
            batch_size: int = 256
    ) -> float:
        # unmasked forecasts on the given validation windows (those without missing values or missing time steps)
        ft_mask, attn_mask = get_full_masks(len(self.feature_names), device=self.get_device(),
                                            input_length=self.get_input_length())
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                n = len(batch_indices)
                x_enc, x_dec, y = self._get_batch(series, batch_indices)
                output = self.forward(x_enc, x_dec, mask_enc=ft_mask.repeat(n, 1), mask_dec=ft_mask[1:].repeat(n, 1),
                                      attn_mask=attn_mask.repeat(n, 1, 1))
                total_loss += self._loss(loss_function, output, y, x_enc).item() * n
        self.train()
        return total_loss / len(indices)

    def predict(self, past: pd.DataFrame, future: pd.DataFrame) -> pd.Series:
        # past: the last input_length rows are used, including the target
        # future: horizon rows with the covariates only
        target = self.feature_names[0]
        x_enc, x_dec = self._prepare_inputs(past, future)
        x_enc = {name: x[None, :] for name, x in x_enc.items()}
        x_dec = {name: x[None, :] for name, x in x_dec.items()}
        ft_mask, attn_mask = get_full_masks(len(self.feature_names), device=self.get_device(),
                                            input_length=self.get_input_length())
        self.eval()
        with torch.no_grad():
            output = self.forward(x_enc, x_dec, mask_enc=ft_mask[None, :], mask_dec=ft_mask[None, 1:],
                                  attn_mask=attn_mask[None, :, :])
        prediction = self.scalers[target].inverse_transform(output.cpu().numpy().reshape(-1, 1))[:, 0]
        return pd.Series(prediction, index=future.index, name=target)

    def explain(
            self,
            past: pd.DataFrame,
            future: pd.DataFrame,
            method: str = "exact",
            n_permutations: int = 10,
            seed: Optional[int] = None,
            progress: bool = True
    ) -> Dict:
        # same inputs as predict; all outputs are in the unit of the target, with
        # prediction = base + sum of the SHAP values of all feature groups.
        # The (unscaled) inputs are included as "past" and "future".
        # method: "exact" evaluates the model with all feature group subsets,
        # "permutation" approximates the same values with n_permutations sampled permutations.
        # progress: show a progress bar over the batches of model evaluations
        x_enc, x_dec = self._prepare_inputs(past, future)
        self.eval()
        sample = {"x_enc": x_enc, "x_dec": x_dec}
        if method == "exact":
            explanation = explain_sample(self, sample, progress=progress)
        elif method == "permutation":
            explanation = explain_sample_permutation(self, sample, n_permutations=n_permutations, seed=seed,
                                                     progress=progress)
        else:
            raise ValueError(f"Unknown method '{method}', must be 'exact' or 'permutation'.")
        scaler = self.scalers[self.feature_names[0]]
        mean, std = scaler.mean_[0], scaler.scale_[0]
        return {
            "prediction": pd.Series(mean + std * explanation["prediction"], index=future.index),
            "base": pd.Series(mean + std * explanation["base"], index=future.index),
            "shap_values": pd.DataFrame(
                {name: std * values for name, values in explanation["shap_values"].items()},
                index=future.index
            ),
            "past": past.iloc[-self.get_input_length():][self.feature_names].copy(),
            "periods": self.get_periods(past.index[-self.get_input_length():]),
            "future": future[self.feature_names[1:]].copy()
        }

    def _prepare_inputs(self, past: pd.DataFrame, future: pd.DataFrame):
        input_length, horizon = self.get_input_length(), self.get_horizon()
        if len(past) < input_length:
            raise ValueError(f"past must have at least {input_length} rows, but has {len(past)}.")
        if len(future) != horizon:
            raise ValueError(f"future must have {horizon} rows, but has {len(future)}.")
        device = self.get_device()
        x_enc = self._to_tensors(self.scale(past.iloc[-input_length:]), device, self.feature_names)
        x_dec = self._to_tensors(self.scale(future), device, self.feature_names[1:])
        return x_enc, x_dec

    def _to_tensors(self, data: pd.DataFrame, device, names: List[str]) -> Dict[str, torch.Tensor]:
        for name, n_vals in self.feature_categories.items():
            if n_vals is not None and not data[name].between(0, n_vals - 1).all():
                raise ValueError(f"Feature '{name}' has {n_vals} categories, so its values must be in "
                                 f"[0, {n_vals - 1}], but they range from {data[name].min()} to {data[name].max()}.")
        return {
            name: torch.tensor(
                data[name].to_numpy(),
                dtype=torch.float32 if name in self.continuous_features else torch.long,
                device=device
            )
            for name in names
        }

    def _get_batch(self, series: Dict[str, torch.Tensor], starts: List[int]):
        def windows(name, offset, length):
            return torch.stack([series[name][start + offset:start + offset + length] for start in starts])

        input_length, horizon = self.get_input_length(), self.get_horizon()
        x_enc = {name: windows(name, 0, input_length) for name in self.feature_names}
        x_dec = {name: windows(name, input_length, horizon) for name in self.feature_names[1:]}
        y = windows(self.feature_names[0], input_length, horizon)
        return x_enc, x_dec, y

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        for name, scaler in self.scalers.items():
            if name not in data:
                continue
            data[name] =scaler.transform(data[[name]].to_numpy())[:, 0]
        return data

    def _add_ghost_vector(self, embedding, attn_mask):
        embedding_shape = embedding.shape
        new_embedding = torch.zeros((embedding_shape[0], embedding_shape[1] + 1, embedding_shape[2]),
                                    device=embedding.device)
        new_embedding[:, 1:, :] = embedding
        attn_mask_shape = attn_mask.shape
        new_attn_mask = torch.zeros((attn_mask_shape[0], attn_mask_shape[1] + 1, attn_mask_shape[2] + 1),
                                    device=attn_mask.device)
        new_attn_mask[:, 1:, 1:] = attn_mask
        new_attn_mask[:, 0, 1:] = -torch.inf
        return new_embedding, new_attn_mask

    def _instance_norm(self) -> str:
        # models saved before instance normalization was added have no attribute instance_norm
        return getattr(self, "instance_norm", "none")

    def _instance_stats(self, past_target: torch.Tensor):
        # location and scale of each sample, from its (unmasked) past target values, shape (batch, 1)
        instance_norm = self._instance_norm()
        if instance_norm == "mean":
            scale = past_target.abs().mean(dim=1, keepdim=True)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            return torch.zeros_like(scale), scale
        if instance_norm == "RevIN":
            loc = past_target.mean(dim=1, keepdim=True)
            scale = torch.sqrt(past_target.var(dim=1, keepdim=True, unbiased=False) + REVIN_EPS)
            return loc, scale
        ones = torch.ones((past_target.shape[0], 1), device=past_target.device)
        return torch.zeros_like(ones), ones

    def _loss(self, loss_function, output, y, x_enc):
        # with instance normalization, the loss is computed on the per-sample normalized values, so that all samples
        # count equally; with instance_norm="none", it is computed on the globally scaled values
        loc, scale = self._instance_stats(x_enc[self.feature_names[0]])
        return loss_function((output - loc) / scale, (y - loc) / scale)

    def forward(self, x_enc, x_dec, mask_enc, mask_dec, attn_mask):
        # inputs and outputs are in the (globally) scaled unit of the target;
        # with instance normalization, the past target is normalized before and the output denormalized after
        # the network, with the statistics of the unmasked past target, which are the same for all masks
        target = self.feature_names[0]
        instance_norm = self._instance_norm()
        if instance_norm != "none":
            loc, scale = self._instance_stats(x_enc[target])
            normalized = (x_enc[target] - loc) / scale
            if instance_norm == "RevIN":
                normalized = normalized * self.revin_weight + self.revin_bias
            x_enc = {**x_enc, target: normalized}
        out = self._forward_network(x_enc, x_dec, mask_enc, mask_dec, attn_mask)
        if instance_norm != "none":
            if instance_norm == "RevIN":
                out = (out - self.revin_bias) / (self.revin_weight + REVIN_EPS)
            out = out * scale + loc
        return out

    def _forward_network(self, x_enc, x_dec, mask_enc, mask_dec, attn_mask):
        enc_embedding = self.encoder_embedding(x_enc, mask_enc)
        enc_embedding += self.positional_encoding(enc_embedding)
        enc_embedding, attn_mask = self._add_ghost_vector(enc_embedding, attn_mask)
        dec_embedding = self.decoder_embedding(x_dec, mask_dec)
        dec_embedding += self.positional_encoding(dec_embedding)
        attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)
        # all real time steps of the encoder have the same mask row (a masked day is hidden from every query),
        # so the decoder queries use that row as well; there are horizon of them, which may differ from input_length
        memory_mask = attn_mask[:, 1:2, :].expand(-1, dec_embedding.shape[1], -1).contiguous()
        out = self.transformer(enc_embedding, dec_embedding, src_mask=attn_mask, memory_mask=memory_mask)
        out = self.projection(self.relu(out))
        if self.output_dimensions == 1:
            out = out[:, :, 0]
        return out

    def get_device(self):
        return next(self.parameters()).device
