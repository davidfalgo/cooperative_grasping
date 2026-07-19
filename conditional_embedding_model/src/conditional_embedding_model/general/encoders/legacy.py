# Description: Composition-based adapters wrapping the frozen v4/v5 model classes as Encoders.
#
# The model classes are imported lazily (inside each __init__) rather than at module
# scope: `conditional_embedding_model.models.center_embedding_resnet` imports `wandb`
# at import time, and this module is reachable via `general/__init__.py` — an eager
# import here would make importing `conditional_embedding_model.general.training`
# (which initializes the `general` package first) transitively import wandb too.

from torchvision import transforms

from ..registry import register_encoder
from .base import Encoder

# v4/v5 transform.transforms == [Resize, Grayscale, Normalize, RandomHorizontalFlip,
# RandomAffine, ColorJitter]; the first three are deterministic.
_DETERMINISTIC_TRANSFORM_COUNT = 3


def _eval_safe_compose(inner_transform):
    ops = list(inner_transform.transforms)[:_DETERMINISTIC_TRANSFORM_COUNT]
    return transforms.Compose(ops)


class _LegacyEncoderMixin:
    """Shared forward/eval-safe-transform logic for legacy center/context adapters."""

    def __init__(self, eval_safe_transform=False):
        self.eval_safe_transform = eval_safe_transform
        if eval_safe_transform:
            self._eval_transform = _eval_safe_compose(self.inner.transform)
            self._train_transform = self.inner.transform

    def _forward_inner(self, index, feature):
        if self.eval_safe_transform:
            self.inner.transform = self._train_transform if self.training else self._eval_transform
        return self.inner(index.long(), feature)


@register_encoder("legacy_v4_center")
class LegacyV4CenterEncoder(Encoder, _LegacyEncoderMixin):
    input_spec = {"index": "[B] or [B,1] long", "feature": "[B, F] float"}

    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1, eval_safe_transform=False):
        from conditional_embedding_model.models.center_embedding_resnet import CenterEmbeddingResNetv4
        Encoder.__init__(self)
        self.inner = CenterEmbeddingResNetv4(input_dim, embedding_dim, feature_structure, dropout)
        self.output_dim = embedding_dim
        _LegacyEncoderMixin.__init__(self, eval_safe_transform)

    def forward(self, inputs):
        self.validate_inputs(inputs)
        return self._forward_inner(inputs["index"], inputs["feature"])


@register_encoder("legacy_v4_context")
class LegacyV4ContextEncoder(Encoder, _LegacyEncoderMixin):
    input_spec = {"index": "[B, K] long", "feature": "[B, F] float"}

    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1, eval_safe_transform=False):
        from conditional_embedding_model.models.center_embedding_resnet import ContextEmbeddingResNetv4
        Encoder.__init__(self)
        self.inner = ContextEmbeddingResNetv4(input_dim, embedding_dim, feature_structure, dropout)
        self.output_dim = embedding_dim
        _LegacyEncoderMixin.__init__(self, eval_safe_transform)

    def forward(self, inputs):
        self.validate_inputs(inputs)
        return self._forward_inner(inputs["index"], inputs["feature"])


@register_encoder("legacy_v5_center")
class LegacyV5CenterEncoder(Encoder, _LegacyEncoderMixin):
    input_spec = {"index": "[B] or [B,1] long", "feature": "[B, F] float"}

    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1, eval_safe_transform=False):
        from conditional_embedding_model.models.center_embedding_resnet import CenterEmbeddingResNetv5
        Encoder.__init__(self)
        self.inner = CenterEmbeddingResNetv5(input_dim, embedding_dim, feature_structure, dropout)
        self.output_dim = embedding_dim
        _LegacyEncoderMixin.__init__(self, eval_safe_transform)

    def forward(self, inputs):
        self.validate_inputs(inputs)
        return self._forward_inner(inputs["index"], inputs["feature"])


@register_encoder("legacy_v5_context")
class LegacyV5ContextEncoder(Encoder, _LegacyEncoderMixin):
    input_spec = {"index": "[B, K] long", "feature": "[B, F] float"}

    def __init__(self, input_dim, embedding_dim, feature_structure, dropout=0.1, eval_safe_transform=False):
        from conditional_embedding_model.models.center_embedding_resnet import ContextEmbeddingResNetv5
        Encoder.__init__(self)
        self.inner = ContextEmbeddingResNetv5(input_dim, embedding_dim, feature_structure, dropout)
        self.output_dim = embedding_dim
        _LegacyEncoderMixin.__init__(self, eval_safe_transform)

    def forward(self, inputs):
        self.validate_inputs(inputs)
        return self._forward_inner(inputs["index"], inputs["feature"])
