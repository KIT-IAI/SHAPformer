from typing import Dict, List, Optional

import torch
from torch import nn
from .transformer_network import PositionalEncoding
from ..transformer.transformer import Transformer


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
        attn_mask = attn_mask.repeat_interleave(168, dim=0)[:, None, :]
        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=key,
            attn_mask=attn_mask
        )
        self.attention_weights = attn_weights
        attn_output = attn_output.reshape((batch_size, seq_len, self.dim))
        return attn_output


class ConcatEmbedding(nn.Module):
    def __init__(
            self,
            feature_names: List[str],
            categories: List[Optional[int]],
            dim: int
    ):
        super(ConcatEmbedding, self).__init__()
        self.feature_names = feature_names
        self.categories = categories
        self.dim = dim
        self.linear = nn.Linear(len(feature_names) * 2, dim)

    def _concatenate_mask_and_features(self, x: Dict[str, torch.Tensor], mask: torch.Tensor):
        ts_len = x[list(x)[0]].shape[1]
        mask_repeat = mask.repeat(1, ts_len, 1)
        feature_values = [x[feature][:, :, None] for feature in x]
        concatenated = torch.cat(feature_values, dim=-1)
        concatenated = torch.cat((mask_repeat, concatenated), dim=-1)
        return concatenated

    def forward(self, x: Dict[str, torch.Tensor], mask: torch.Tensor):
        concatenated = self._concatenate_mask_and_features(x, mask)
        # apply mask:
        concatenated[:, :, len(mask):] = concatenated[:, :, len(mask):] * mask
        # linear transformation:
        out = self.linear(concatenated)
        return out


class ExogenousTransformer(nn.Module):
    def __init__(
            self,
            encoder_feature_names: List[str],
            encoder_categories: List[Optional[int]],
            decoder_feature_names: List[str],
            decoder_categories: List[Optional[int]],
            d_model: int,
            n_layers: int,
            n_heads: int,
            output_dimensions: int = 1,
            embedding: str = "attention"
    ):
        super(ExogenousTransformer, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        if embedding == "attention":
            self.encoder_embedding = FeatureEmbedding(
                feature_names=encoder_feature_names,
                categories=encoder_categories,
                dim=d_model
            )
            self.decoder_embedding = FeatureEmbedding(
                feature_names=decoder_feature_names,
                categories=decoder_categories,
                dim=d_model,
                dummy_feature=True
            )
        else:
            self.encoder_embedding = ConcatEmbedding(
                feature_names=encoder_feature_names,
                categories=encoder_categories,
                dim=d_model
            )
            self.decoder_embedding = ConcatEmbedding(
                feature_names=decoder_feature_names,
                categories=decoder_categories,
                dim=d_model
            )
        self.positional_encoding = PositionalEncoding(d_model)
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

    def forward(self, x_enc, x_dec, mask_enc, mask_dec, attn_mask):
        enc_embedding = self.encoder_embedding(x_enc, mask_enc)
        enc_embedding += self.positional_encoding(enc_embedding)
        enc_embedding, attn_mask = self._add_ghost_vector(enc_embedding, attn_mask)
        dec_embedding = self.decoder_embedding(x_dec, mask_dec)
        dec_embedding += self.positional_encoding(dec_embedding)
        attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)
        out = self.transformer(enc_embedding, dec_embedding, src_mask=attn_mask, memory_mask=attn_mask[:, 1:, :])
        out = self.projection(self.relu(out))
        if self.output_dimensions == 1:
            out = out[:, :, 0]
        return out
