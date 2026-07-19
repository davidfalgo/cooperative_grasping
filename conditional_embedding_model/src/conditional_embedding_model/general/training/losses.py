# Description: Masked BCE loss for contrastive center/context training, with a mode
# that bit-reproduces the legacy DynamicSigmoidBCELoss (trainer.py L263-281).

import torch
from torch import nn
import torch.nn.functional as F

from ..registry import register_loss

_NORMALIZATIONS = ("valid_pairs", "legacy_batch")
_POS_WEIGHT_MODES = ("dynamic", "static")


@register_loss("contrastive_bce")
class ContrastiveBCELoss(nn.Module):
    """pred: [B,K] or [B,K,R]; label/mask: same shape as pred, or [B,K] broadcast over R."""

    def __init__(self, pos_weight_mode="dynamic", normalization="valid_pairs",
                 pos_weight=1.0, neg_weight=1.0, eps=1e-8):
        super().__init__()
        if pos_weight_mode not in _POS_WEIGHT_MODES:
            raise ValueError(f"Unknown pos_weight_mode '{pos_weight_mode}'. Available: {_POS_WEIGHT_MODES}")
        if normalization not in _NORMALIZATIONS:
            raise ValueError(f"Unknown normalization '{normalization}'. Available: {_NORMALIZATIONS}")
        self.pos_weight_mode = pos_weight_mode
        self.normalization = normalization
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.eps = eps

    def _weights(self, label, mask, clamp_neg_counts):
        if self.pos_weight_mode == "dynamic":
            if mask is None:
                raise ValueError("pos_weight_mode='dynamic' requires a mask")
            pos_counts = (label * mask.float()).sum(dim=1, keepdim=True)
            neg_counts = ((1 - label) * mask.float()).sum(dim=1, keepdim=True)
            if clamp_neg_counts:
                neg_counts = neg_counts.clamp_min(self.eps)
            neg_weights = pos_counts / neg_counts
            neg_weights = neg_weights.expand_as(label)
            weights = label + (1 - label) * neg_weights
            if mask is not None:
                weights = weights * mask.float()
        else:
            weights = label * self.pos_weight + (1 - label) * self.neg_weight
            if mask is not None:
                weights = weights * mask.float()
        return weights

    def forward(self, pred, label, mask=None):
        label = label.float()
        if mask is not None:
            mask = mask.float()

        if pred.dim() == label.dim() + 1:
            label = label.unsqueeze(-1).expand_as(pred)
            if mask is not None:
                mask = mask.unsqueeze(-1).expand_as(pred)
        elif pred.shape != label.shape:
            raise ValueError(f"pred shape {tuple(pred.shape)} incompatible with label shape {tuple(label.shape)}")

        if self.normalization == "legacy_batch":
            # Bit-reproduce DynamicSigmoidBCELoss.forward (trainer.py L267-281): no
            # clamp on neg_counts, divide by label.shape[0] (batch size).
            weights = self._weights(label, mask, clamp_neg_counts=False)
            loss = F.binary_cross_entropy_with_logits(pred, label, weight=weights, reduction="none")
            return loss.sum() / label.shape[0]

        weights = self._weights(label, mask, clamp_neg_counts=True)
        loss = F.binary_cross_entropy_with_logits(pred, label, weight=weights, reduction="none")
        denom = mask.sum().clamp_min(1) if mask is not None else label.numel()
        return loss.sum() / denom
