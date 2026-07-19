# Description: Unified CEModel core, legacy checkpoint shim, and legacy filename parser.

import re
import torch
from torch import nn

from .registry import build_encoder, build_scorer
from .encoders.base import validate_batch

_LEGACY_FILENAME_RE = re.compile(
    r"^swmodel_(?P<model_version>[^_]+)_E(?P<embedding_size>\d+)_B(?P<batch_size>\d+)_topk_(?P<hit_at_1>[0-9.]+)\.pth$"
)


class CEModel(nn.Module):
    def __init__(self, center_encoder, context_encoder, scorer, validate_inputs: bool = False):
        super().__init__()
        if center_encoder.output_dim != context_encoder.output_dim:
            raise ValueError(
                f"center_encoder.output_dim ({center_encoder.output_dim}) != "
                f"context_encoder.output_dim ({context_encoder.output_dim})"
            )
        self.center_encoder = center_encoder
        self.context_encoder = context_encoder
        self.scorer = scorer
        self.validate_inputs = validate_inputs
        self.config = None

    def embed_center(self, inputs):
        return self.center_encoder(inputs)

    def embed_context(self, inputs):
        return self.context_encoder(inputs)

    def forward(self, center_inputs, context_inputs):
        if self.validate_inputs:
            validate_batch(self.center_encoder, center_inputs)
            validate_batch(self.context_encoder, context_inputs)
        v = self.embed_center(center_inputs)
        u = self.embed_context(context_inputs)
        return self.scorer(v, u)

    def score_all_pairs(self, center_inputs, context_inputs):
        v = self.embed_center(center_inputs)
        u = self.embed_context(context_inputs)
        return self.scorer.all_pairs(v, u)

    @classmethod
    def from_legacy_checkpoint(cls, path, config, map_location="cpu", strict=True):
        model = build_model(config)
        legacy_state = torch.load(path, map_location=map_location)

        remapped = {}
        for key, value in legacy_state.items():
            if key.startswith("0."):
                remapped["center_encoder.inner." + key[len("0."):]] = value
            elif key.startswith("1."):
                remapped["context_encoder.inner." + key[len("1."):]] = value
            else:
                raise KeyError(f"Unexpected legacy checkpoint key prefix: '{key}'")

        model.load_state_dict(remapped, strict=strict)
        return model

    def to_legacy_state_dict(self):
        legacy_state = {}
        for key, value in self.state_dict().items():
            if key.startswith("center_encoder.inner."):
                legacy_state["0." + key[len("center_encoder.inner."):]] = value
            elif key.startswith("context_encoder.inner."):
                legacy_state["1." + key[len("context_encoder.inner."):]] = value
            else:
                raise ValueError(
                    f"Unexpected state_dict key '{key}' has no legacy '0.'/'1.' equivalent "
                    "(scorer/encoder is expected to be parameter-free outside of `inner`)"
                )
        return legacy_state


def build_model(config):
    center_encoder = build_encoder(config.center_encoder)
    context_encoder = build_encoder(config.context_encoder)
    scorer = build_scorer(config.scorer)
    model = CEModel(center_encoder, context_encoder, scorer)
    model.config = config
    return model


def parse_legacy_filename(path):
    name = str(path).rsplit("/", 1)[-1]
    match = _LEGACY_FILENAME_RE.match(name)
    if not match:
        raise ValueError(
            f"'{name}' does not match legacy filename pattern "
            "'swmodel_{model_version}_E{embedding_size}_B{batch_size}_topk_{score}.pth'"
        )
    return {
        "model_version": match.group("model_version"),
        "embedding_size": int(match.group("embedding_size")),
        "batch_size": int(match.group("batch_size")),
        "hit_at_1": float(match.group("hit_at_1")),
    }
