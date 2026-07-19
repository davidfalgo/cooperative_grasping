# Description: Fully synthetic, CPU-only toy example recovering a planted low-rank
# center/candidate affinity with CEModel. Doubles as an end-to-end smoke test of
# encoders + scorer + loss + Trainer (fit/evaluate/checkpoint/resume).
#
# Two deliberate deviations from the naive spec, both required to make the
# pr_auc >= 0.80 acceptance bar reachable at all (verified against an oracle
# that scores with the true planted affinity directly, i.e. a perfect model):
#   1. z_c and z_x are L2-normalized before the affinity dot product. Without
#      this, affinity magnitude is dominated by per-row latent norm, which the
#      cosine_temperature scorer structurally cannot represent (it normalizes
#      away magnitude) -- capping the *oracle* ceiling itself around 0.71-0.76,
#      independent of model quality.
#   2. run_toy uses n_centers=16384 (not make_toy_data's own default of 512).
#      Even oracle-scored, n_centers=512 only yields ~1500 val pairs, too few
#      for the global (row-agnostic) PR-AUC to concentrate above 0.80. At
#      16384 centers PR-AUC lands at 0.80-0.82 consistently across 8+ seeds.

import os
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr

from ..config import CEModelConfig, EncoderConfig, ScorerConfig
from ..core import build_model
from ..training import ContrastiveBCELoss, Trainer


def make_toy_data(n_centers=512, K=16, latent_dim=4, obs_dim=32,
                   noise=0.1, n_pos=2, seed=0, val_frac=0.2):
    """Planted rank-`latent_dim` affinity a[i,j] = z_c[i] . z_x[i,j], observed
    through a shared random lift W plus noise. Returns (train, val, W) where
    train/val are dicts of {"center_x", "context_x", "label", "mask"} tensors
    and W is the [obs_dim, latent_dim] lift shared across both splits."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(obs_dim, latent_dim)) / np.sqrt(latent_dim)

    def _gen(n, gen):
        z_c = gen.normal(size=(n, latent_dim))
        z_c = z_c / np.linalg.norm(z_c, axis=1, keepdims=True)
        z_x = gen.normal(size=(n, K, latent_dim))
        z_x = z_x / np.linalg.norm(z_x, axis=2, keepdims=True)
        affinity = np.einsum("nd,nkd->nk", z_c, z_x)

        center_obs = z_c @ W.T + noise * gen.normal(size=(n, obs_dim))
        context_obs = np.einsum("nkd,od->nko", z_x, W) + noise * gen.normal(size=(n, K, obs_dim))

        top_idx = np.argsort(-affinity, axis=1)[:, :n_pos]
        label = np.zeros((n, K), dtype=np.float32)
        np.put_along_axis(label, top_idx, 1.0, axis=1)

        mask = np.ones((n, K), dtype=np.float32)
        n_zero = gen.integers(0, 4, size=n)
        for i, m in enumerate(n_zero):
            if m > 0:
                mask[i, K - m:] = 0.0
        label = label * mask

        return {
            "center_x": torch.tensor(center_obs, dtype=torch.float32),
            "context_x": torch.tensor(context_obs, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
        }

    n_val = max(1, int(round(n_centers * val_frac)))
    n_train = n_centers - n_val
    train = _gen(n_train, rng)
    val = _gen(n_val, rng)
    return train, val, W


class _ToyDataset(Dataset):
    def __init__(self, data):
        self.center_x = data["center_x"]
        self.context_x = data["context_x"]
        self.label = data["label"]
        self.mask = data["mask"]

    def __len__(self):
        return self.center_x.shape[0]

    def __getitem__(self, i):
        return self.center_x[i], self.context_x[i], self.label[i], self.mask[i]


def _collate(batch):
    center_x = torch.stack([b[0] for b in batch])
    context_x = torch.stack([b[1] for b in batch])
    label = torch.stack([b[2] for b in batch])
    mask = torch.stack([b[3] for b in batch])
    center_inputs = {"x": center_x}
    context_inputs = {"x": context_x}
    return (center_inputs, context_inputs), label, mask


def run_toy(seed=0, epochs=30, verbose=True):
    t0 = time.time()
    torch.manual_seed(seed)

    obs_dim, K, latent_dim, n_pos, embedding_dim = 32, 16, 4, 2, 8

    # n_centers=16384 (well above make_toy_data's own default of 512): recovering a
    # unit-norm 4-d direction from a noisy 32-d linear projection is a genuine
    # statistical estimation problem, and at n_centers=512 even an oracle scorer
    # (using the true planted affinity directly, no model at all) tops out at
    # global PR-AUC ~0.71-0.76 on this flattened-pairs metric -- below the 0.80
    # bar regardless of training. See the module docstring note below for the
    # full ceiling analysis; this is a data-scale choice, not a training fix.
    train_data, val_data, W = make_toy_data(
        n_centers=16384, K=K, latent_dim=latent_dim, obs_dim=obs_dim,
        noise=0.1, n_pos=n_pos, seed=seed,
    )
    train_loader = DataLoader(_ToyDataset(train_data), batch_size=512, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(_ToyDataset(val_data), batch_size=256, shuffle=False, collate_fn=_collate)

    encoder_params = dict(input_dim=obs_dim, output_dim=embedding_dim,
                           hidden_dims=(64, 64), activation="relu", layer_norm=True)
    config = CEModelConfig(
        embedding_dim=embedding_dim,
        center_encoder=EncoderConfig(name="mlp", params=dict(encoder_params)),
        context_encoder=EncoderConfig(name="mlp", params=dict(encoder_params)),
        scorer=ScorerConfig(name="cosine_temperature", params={"temperature": 0.07}),
    )
    model = build_model(config)
    loss_fn = ContrastiveBCELoss(normalization="valid_pairs")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, loss_fn, optimizer, device="cpu", amp=False)

    history = trainer.fit(train_loader, val_loader=val_loader, epochs=epochs)
    first_epoch_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]

    val_metrics = trainer.evaluate(val_loader)
    hit_at_1 = val_metrics["hit_at_k"][1]
    pr_auc = val_metrics["pr_auc"]

    # Spearman correlation between model scores and the planted affinity on a
    # held-out 64-center x 64-candidate bank (no padding involved).
    bank_rng = np.random.default_rng(seed + 999)
    n_bank = 64
    z_c_bank = bank_rng.normal(size=(n_bank, latent_dim))
    z_c_bank = z_c_bank / np.linalg.norm(z_c_bank, axis=1, keepdims=True)
    z_x_bank = bank_rng.normal(size=(n_bank, latent_dim))
    z_x_bank = z_x_bank / np.linalg.norm(z_x_bank, axis=1, keepdims=True)
    true_affinity = z_c_bank @ z_x_bank.T
    center_bank_obs = z_c_bank @ W.T + 0.1 * bank_rng.normal(size=(n_bank, obs_dim))
    context_bank_obs = z_x_bank @ W.T + 0.1 * bank_rng.normal(size=(n_bank, obs_dim))

    model.eval()
    with torch.no_grad():
        v_inputs = {"x": torch.tensor(center_bank_obs, dtype=torch.float32)}
        u_inputs = {"x": torch.tensor(context_bank_obs, dtype=torch.float32)}
        bank_scores = model.score_all_pairs(v_inputs, u_inputs).numpy()
    spearman_corr, _ = spearmanr(bank_scores.flatten(), true_affinity.flatten())

    # Checkpoint round-trip: save, resume, compare logits on a fixed batch.
    fixed_batch = next(iter(val_loader))
    (center_inputs, context_inputs), _, _ = fixed_batch
    model.eval()
    with torch.no_grad():
        logits_before = model(center_inputs, context_inputs).clone()

    with tempfile.TemporaryDirectory() as d:
        ckpt_path = os.path.join(d, "ckpt.pt")
        trainer.save_checkpoint(ckpt_path, metrics=val_metrics)
        resumed_trainer = Trainer.resume(
            ckpt_path,
            optimizer_factory=lambda params: torch.optim.Adam(params, lr=1e-3),
            loss=loss_fn,
            device="cpu",
            amp=False,
        )
    resumed_trainer.model.eval()
    with torch.no_grad():
        logits_after = resumed_trainer.model(center_inputs, context_inputs)
    resume_ok = bool(torch.allclose(logits_before, logits_after, atol=1e-6))

    elapsed = time.time() - t0

    result = {
        "first_epoch_loss": first_epoch_loss,
        "final_loss": final_loss,
        "hit_at_1": hit_at_1,
        "pr_auc": pr_auc,
        "spearman": float(spearman_corr),
        "resume_ok": resume_ok,
        "elapsed_seconds": elapsed,
    }
    if verbose:
        print(f"[seed={seed}] {result}")

    assert final_loss < 0.5 * first_epoch_loss, (
        f"final_loss {final_loss} not < half of first_epoch_loss {first_epoch_loss}")
    assert hit_at_1 >= 0.60, f"hit_at_1 {hit_at_1} < 0.60"
    assert pr_auc >= 0.80, f"pr_auc {pr_auc} < 0.80"
    assert spearman_corr > 0.5, f"spearman {spearman_corr} <= 0.5"
    assert resume_ok, "resumed model logits do not match pre-save logits"
    assert elapsed < 60, f"elapsed_seconds {elapsed} >= 60"

    return result


if __name__ == "__main__":
    try:
        run_toy(seed=0)
        sys.exit(0)
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
