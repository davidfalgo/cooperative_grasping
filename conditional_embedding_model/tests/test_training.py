# Description: Standalone (non-pytest) tests for general/training/: loss parity
# with legacy DynamicSigmoidBCELoss, metric parity with evaluate_model /
# evaluate_model_topk, resume determinism, the legacy dataloader adapter, and the
# no-wandb-at-import-time constraint.

import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc,
)

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from conditional_embedding_model.general import (
    CEModelConfig, EncoderConfig, ScorerConfig, build_model,
)
from conditional_embedding_model.general.training import (
    ContrastiveBCELoss, hit_at_k, classification_metrics,
    make_collate, LegacyBatchAdapter, Trainer, load_model,
)


# --------------------------------------------------------------------------- #
# Criterion 1: loss parity with legacy DynamicSigmoidBCELoss
# --------------------------------------------------------------------------- #

def _scalar_matches(a, b, atol=1e-7, rtol=1e-6):
    a, b = float(a), float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) and math.isinf(b):
        return a == b
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= atol + rtol * abs(b)


def _rand_case(rng, B=8, K=100, force_all_positive_row=False, force_all_negative_row=False):
    mask = (rng.random((B, K)) < 0.7).astype(np.float32)
    for b in range(B):
        if mask[b].sum() == 0:
            mask[b, 0] = 1.0
    label = ((rng.random((B, K)) < 0.3).astype(np.float32)) * mask
    if force_all_positive_row:
        label[0] = mask[0]
    if force_all_negative_row:
        label[1] = 0.0
    pred = rng.standard_normal((B, K)).astype(np.float32)
    return torch.from_numpy(pred), torch.from_numpy(label), torch.from_numpy(mask)


def test_loss_parity_legacy_batch():
    from conditional_embedding_model.training.trainer import DynamicSigmoidBCELoss

    legacy_loss = DynamicSigmoidBCELoss()
    new_loss = ContrastiveBCELoss(pos_weight_mode="dynamic", normalization="legacy_batch")
    new_loss_valid = ContrastiveBCELoss(pos_weight_mode="dynamic", normalization="valid_pairs")

    rng = np.random.default_rng(1234)
    n_cases = 100
    for i in range(n_cases):
        force_pos = (i % 10 == 0)
        force_neg = (i % 10 == 5)
        pred, label, mask = _rand_case(rng, force_all_positive_row=force_pos, force_all_negative_row=force_neg)

        legacy_out = legacy_loss(pred, label, mask)
        new_out = new_loss(pred, label, mask)
        assert _scalar_matches(legacy_out, new_out), \
            f"case {i}: legacy={legacy_out.item()} new={new_out.item()} (force_pos={force_pos}, force_neg={force_neg})"
        if force_pos:
            # An all-positive-among-valid row zeroes neg_counts -> legacy_batch must
            # reproduce the resulting inf/nan, not silently become finite.
            assert not math.isfinite(legacy_out.item()), \
                f"case {i}: expected legacy DynamicSigmoidBCELoss to be non-finite on an all-positive row"
            assert not math.isfinite(new_out.item()), \
                f"case {i}: expected legacy_batch mode to reproduce the non-finite value"

        valid_out = new_loss_valid(pred, label, mask)
        assert math.isfinite(valid_out.item()), \
            f"case {i}: valid_pairs mode produced non-finite loss {valid_out.item()}"

    print(f"[loss parity] PASS ({n_cases} cases, incl. all-positive/all-negative rows)")


# --------------------------------------------------------------------------- #
# Criterion 2: metric parity with evaluate_model / evaluate_model_topk
# --------------------------------------------------------------------------- #

def _legacy_hit_at_k(logits, labels, masks, k_list):
    """Mirrors trainer.py evaluate_model_topk L132-157 (post prediction_temperature)."""
    logits = logits.clone()
    logits[masks == 0] = -float("inf")
    top_k_correct = {k: 0 for k in k_list}
    total_samples = 0
    top_k_indices_by_k = {k: logits.topk(k, dim=1).indices for k in k_list}
    for k in k_list:
        batch_hits = labels.gather(1, top_k_indices_by_k[k]).any(dim=1).sum().item()
        top_k_correct[k] += batch_hits
    total_samples += (labels.sum(dim=1) > 0).sum().item()
    return {k: top_k_correct[k] / max(total_samples, 1) for k in k_list}


def _legacy_classification_metrics(logits, labels, masks, method="weighted"):
    """Mirrors trainer.py evaluate_model L76-115 + evaluate_metrics L117-130."""
    valid_logits = logits[masks == 1]
    valid_labels = labels[masks == 1]
    all_preds = valid_logits.sigmoid().numpy()
    all_labels = valid_labels.numpy()

    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(all_labels, all_preds >= t, average=method) for t in thresholds]
    best_idx = int(np.argmax(f1_scores))
    threshold = thresholds[best_idx]

    binary_preds = (all_preds >= threshold).astype(int)
    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, average=method, zero_division=0)
    recall = recall_score(all_labels, binary_preds, average=method, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, average=method, zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_preds)

    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall_curve, precision_curve)

    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "roc_auc": auc_roc, "pr_auc": pr_auc, "threshold": threshold}


def test_metric_parity():
    rng = np.random.default_rng(42)
    B, K = 16, 50
    logits = torch.from_numpy(rng.standard_normal((B, K)).astype(np.float32))
    masks = torch.from_numpy((rng.random((B, K)) < 0.6).astype(np.float32))
    for b in range(B):
        if masks[b].sum() == 0:
            masks[b, 0] = 1.0
    labels = torch.from_numpy((rng.random((B, K)) < 0.3).astype(np.float32)) * masks

    k_list = [1, 3, 5, 7]
    legacy_hits = _legacy_hit_at_k(logits, labels, masks, k_list)
    new_hits = hit_at_k(logits, labels, masks, ks=tuple(k_list))
    for k in k_list:
        assert _scalar_matches(legacy_hits[k], new_hits[k]), \
            f"hit@{k}: legacy={legacy_hits[k]} new={new_hits[k]}"
    print(f"[hit_at_k parity] PASS {new_hits}")

    legacy_cls = _legacy_classification_metrics(logits, labels, masks)
    new_cls = classification_metrics(logits, labels, masks)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold"):
        assert _scalar_matches(legacy_cls[key], new_cls[key]), \
            f"{key}: legacy={legacy_cls[key]} new={new_cls[key]}"
    print(f"[classification_metrics parity] PASS pr_auc={new_cls['pr_auc']:.4f}")


# --------------------------------------------------------------------------- #
# Criterion 3: resume determinism
# --------------------------------------------------------------------------- #

def _make_synthetic_loader(batch_size=4, num_batches=3, K=10, D=6, seed=99):
    g = torch.Generator().manual_seed(seed)
    n = batch_size * num_batches
    centers = torch.randn(n, D, generator=g)
    contexts = torch.randn(n, K, D, generator=g)
    labels = (torch.rand(n, K, generator=g) < 0.3).float()
    masks = (torch.rand(n, K, generator=g) < 0.8).float()
    masks[:, 0] = 1.0
    dataset = TensorDataset(centers, contexts, labels, masks)

    def collate(batch):
        c, ctx, lab, msk = zip(*batch)
        center_inputs = {"x": torch.stack(c)}
        context_inputs = {"x": torch.stack(ctx)}
        return (center_inputs, context_inputs), torch.stack(lab), torch.stack(msk)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def _resume_test_config():
    return CEModelConfig(
        embedding_dim=8,
        center_encoder=EncoderConfig(name="mlp", params=dict(
            input_dim=6, output_dim=8, hidden_dims=(16,), dropout=0.3, input_key="x")),
        context_encoder=EncoderConfig(name="mlp", params=dict(
            input_dim=6, output_dim=8, hidden_dims=(16,), dropout=0.3, input_key="x")),
        scorer=ScorerConfig(name="cosine_temperature", params={"temperature": 0.07}),
    )


def _build_resume_trainer(seed):
    torch.manual_seed(seed)
    model = build_model(_resume_test_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = ContrastiveBCELoss(pos_weight_mode="dynamic", normalization="valid_pairs")
    return Trainer(model, loss, optimizer, device="cpu", amp=False)


def test_resume_determinism():
    seed = 1234

    trainer_a = _build_resume_trainer(seed)
    hist_a = trainer_a.fit(_make_synthetic_loader(), epochs=3)

    trainer_b = _build_resume_trainer(seed)
    hist_b1 = trainer_b.fit(_make_synthetic_loader(), epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pth")
        trainer_b.save_checkpoint(ckpt_path)

        trainer_b_resumed = Trainer.resume(
            ckpt_path,
            optimizer_factory=lambda params: torch.optim.AdamW(params, lr=1e-3),
            loss=ContrastiveBCELoss(pos_weight_mode="dynamic", normalization="valid_pairs"),
            device="cpu", amp=False,
        )
    hist_b2 = trainer_b_resumed.fit(_make_synthetic_loader(), epochs=1)

    steps_a = hist_a["train_step_loss"]
    steps_b = hist_b1["train_step_loss"] + hist_b2["train_step_loss"]
    assert len(steps_a) == len(steps_b), f"step count mismatch: {len(steps_a)} vs {len(steps_b)}"
    for i, (x, y) in enumerate(zip(steps_a, steps_b)):
        assert abs(x - y) <= 1e-6, f"step {i}: uninterrupted={x} resumed={y} (diff={abs(x - y)})"

    assert trainer_b_resumed._epoch == 3
    print(f"[resume determinism] PASS ({len(steps_a)} steps matched to 1e-6)")


# --------------------------------------------------------------------------- #
# Criterion 4: legacy pipeline plugs in via LegacyBatchAdapter
# --------------------------------------------------------------------------- #

def test_legacy_pipeline_smoke():
    from conditional_embedding_model.data.dataloader import GraspDataset, batchify_wrapper

    feature_structure = {
        "features_structured": {"map": 8100, "object_footprint": 8, "grasping_approach": 600},
        "features_shape": {"map": (90, 90), "object_footprint": (4, 2), "grasping_approach": (150, 4)},
    }
    feat_dim = 8100 + 8 + 600

    n_samples = 6
    centers = [0] * n_samples
    contexts = [[1, 2, 3]] * n_samples
    negatives = [[4, 5, 6, 7, 8]] * n_samples
    features = [[0.0] * feat_dim for _ in range(n_samples)]

    dataset = GraspDataset(centers, contexts, features, negatives)
    legacy_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=batchify_wrapper(100))
    adapter = LegacyBatchAdapter(legacy_loader)

    first_batch = next(iter(adapter))
    (center_inputs, context_inputs), label, mask = first_batch
    assert center_inputs["index"].dtype == torch.long, center_inputs["index"].dtype
    assert context_inputs["index"].dtype == torch.long, context_inputs["index"].dtype
    print("[LegacyBatchAdapter dtype] PASS (index tensors are torch.long)")

    try:
        config = CEModelConfig(
            embedding_dim=16,
            center_encoder=EncoderConfig(name="legacy_v4_center", params=dict(
                input_dim=4, embedding_dim=16, feature_structure=feature_structure, dropout=0.1)),
            context_encoder=EncoderConfig(name="legacy_v4_context", params=dict(
                input_dim=4, embedding_dim=16, feature_structure=feature_structure, dropout=0.1)),
            scorer=ScorerConfig(name="cosine_temperature", params={"temperature": 0.07}),
        )
        model = build_model(config)
    except Exception as e:  # noqa: BLE001 - torchvision weight download may be unavailable offline
        print(f"[legacy_v4 Trainer.fit smoke] SKIPPED ({type(e).__name__}: {e})")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = ContrastiveBCELoss(pos_weight_mode="dynamic", normalization="valid_pairs")
    trainer = Trainer(model, loss, optimizer, device="cpu", amp=False)
    history = trainer.fit(adapter, epochs=1)
    assert len(history["train_step_loss"]) == len(legacy_loader)
    print("[legacy_v4 Trainer.fit smoke] PASS (one epoch completed via LegacyBatchAdapter)")


# --------------------------------------------------------------------------- #
# Criterion 5: importing general.training does not import wandb
# --------------------------------------------------------------------------- #

def test_no_wandb_at_import_time():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {os.path.join(REPO_ROOT, 'src')!r})\n"
        "import conditional_embedding_model.general.training\n"
        "assert 'wandb' not in sys.modules, sorted(m for m in sys.modules if 'wandb' in m)\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout
    print("[no wandb at import time] PASS")


# --------------------------------------------------------------------------- #
# Bonus: make_collate sanity (index dtype, shapes) -- not a numbered criterion
# --------------------------------------------------------------------------- #

def test_make_collate_smoke():
    feat_dim = 12
    batch = [
        (0, [1, 2], [3, 4, 5], [0.0] * feat_dim),
        (1, [2], [0, 3, 4], [1.0] * feat_dim),
    ]
    collate = make_collate(max_length=10)
    (center_inputs, context_inputs), label, mask = collate(batch)
    assert center_inputs["index"].dtype == torch.long
    assert context_inputs["index"].dtype == torch.long
    assert center_inputs["index"].shape == (2, 1)
    assert context_inputs["index"].shape == (2, 10)
    assert label.shape == (2, 10)
    assert mask.shape == (2, 10)
    print("[make_collate smoke] PASS")


if __name__ == "__main__":
    test_loss_parity_legacy_batch()
    test_metric_parity()
    test_resume_determinism()
    test_legacy_pipeline_smoke()
    test_no_wandb_at_import_time()
    test_make_collate_smoke()
    print("\nAll training-package tests passed.")
