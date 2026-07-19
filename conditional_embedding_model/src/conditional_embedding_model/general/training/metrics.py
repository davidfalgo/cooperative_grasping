# Description: Hit@k and classification (PR-AUC/F1/accuracy) metrics matching the
# legacy evaluate_model / evaluate_model_topk procedures (trainer.py L76-157).

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
)


def _hit_at_k_2d(logits, labels, masks, ks):
    logits = logits.clone()
    logits[masks == 0] = -float("inf")
    total_samples = (labels.sum(dim=1) > 0).sum().item()
    result = {}
    for k in ks:
        top_k_indices = logits.topk(k, dim=1).indices
        hits = labels.gather(1, top_k_indices).any(dim=1).sum().item()
        result[k] = hits / max(total_samples, 1)
    return result


@torch.no_grad()
def hit_at_k(logits, labels, masks, ks=(1, 3, 5)):
    """logits/labels/masks: [B,K] -> {k: hit_rate}, or [B,K,R] (labels/masks may be
    [B,K] and are broadcast) -> {"per_relation": [...], "macro": {k: ...}}."""
    if logits.dim() == 3:
        num_relations = logits.shape[-1]
        per_relation = []
        for r in range(num_relations):
            l = labels[..., r] if labels.dim() == 3 else labels
            m = masks[..., r] if masks.dim() == 3 else masks
            per_relation.append(_hit_at_k_2d(logits[..., r], l, m, ks))
        macro = {k: sum(d[k] for d in per_relation) / num_relations for k in ks}
        return {"per_relation": per_relation, "macro": macro}
    return _hit_at_k_2d(logits, labels, masks, ks)


@torch.no_grad()
def all_pairs_hit_at_k(scores, relevance, ks=(1, 3, 5)):
    """scores/relevance: [B,M] (no padding/mask; M is the full candidate bank)."""
    total_samples = (relevance.sum(dim=1) > 0).sum().item()
    result = {}
    for k in ks:
        top_k_indices = scores.topk(k, dim=1).indices
        hits = relevance.gather(1, top_k_indices).any(dim=1).sum().item()
        result[k] = hits / max(total_samples, 1)
    return result


def _classification_metrics_2d(logits, labels, masks, threshold, method):
    valid_logits = logits[masks == 1]
    valid_labels = labels[masks == 1]
    preds = valid_logits.sigmoid().cpu().numpy()
    lab = valid_labels.cpu().numpy()

    if threshold is None:
        thresholds = np.linspace(0, 1, 101)
        f1_scores = [f1_score(lab, preds >= t, average=method) for t in thresholds]
        best_idx = int(np.argmax(f1_scores))
        threshold = float(thresholds[best_idx])

    binary_preds = (preds >= threshold).astype(int)
    accuracy = accuracy_score(lab, binary_preds)
    precision = precision_score(lab, binary_preds, average=method, zero_division=0)
    recall = recall_score(lab, binary_preds, average=method, zero_division=0)
    f1 = f1_score(lab, binary_preds, average=method, zero_division=0)
    roc_auc = roc_auc_score(lab, preds)

    precision_curve, recall_curve, _ = precision_recall_curve(lab, preds)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
    }


@torch.no_grad()
def classification_metrics(logits, labels, masks, threshold=None, method="weighted"):
    """logits/labels/masks: [B,K] -> metrics dict, or [B,K,R] (labels/masks may be
    [B,K] and are broadcast) -> {"per_relation": [...], "macro": {...}}."""
    if logits.dim() == 3:
        num_relations = logits.shape[-1]
        per_relation = []
        for r in range(num_relations):
            l = labels[..., r] if labels.dim() == 3 else labels
            m = masks[..., r] if masks.dim() == 3 else masks
            per_relation.append(_classification_metrics_2d(logits[..., r], l, m, threshold, method))
        macro = {key: sum(d[key] for d in per_relation) / num_relations for key in per_relation[0]}
        return {"per_relation": per_relation, "macro": macro}
    return _classification_metrics_2d(logits, labels, masks, threshold, method)
