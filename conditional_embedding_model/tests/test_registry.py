# Description: Standalone (non-pytest) tests for general/registry.py: duplicate-name
# registration raises, and the built-in encoder/scorer/loss names are all present.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conditional_embedding_model.general import ENCODER_REGISTRY, SCORER_REGISTRY, LOSS_REGISTRY
from conditional_embedding_model.general.registry import register_encoder, register_scorer, register_loss
from conditional_embedding_model.general.encoders.mlp import MLPEncoder
from conditional_embedding_model.general.scoring import DotProductScorer
from conditional_embedding_model.general.training.losses import ContrastiveBCELoss

EXPECTED_ENCODER_NAMES = {
    "legacy_v4_center", "legacy_v4_context", "legacy_v5_center", "legacy_v5_context", "mlp",
}
EXPECTED_SCORER_NAMES = {
    "cosine_temperature", "dot_product", "bilinear", "mlp_scorer", "multi_relation",
}
EXPECTED_LOSS_NAMES = {"contrastive_bce"}


def test_builtin_encoder_names_registered():
    missing = EXPECTED_ENCODER_NAMES - set(ENCODER_REGISTRY)
    assert not missing, f"missing encoder registry names: {missing}"
    print(f"[builtin encoder names] PASS {sorted(ENCODER_REGISTRY)}")


def test_builtin_scorer_names_registered():
    missing = EXPECTED_SCORER_NAMES - set(SCORER_REGISTRY)
    assert not missing, f"missing scorer registry names: {missing}"
    print(f"[builtin scorer names] PASS {sorted(SCORER_REGISTRY)}")


def test_builtin_loss_names_registered():
    missing = EXPECTED_LOSS_NAMES - set(LOSS_REGISTRY)
    assert not missing, f"missing loss registry names: {missing}"
    print(f"[builtin loss names] PASS {sorted(LOSS_REGISTRY)}")


def test_duplicate_encoder_registration_raises():
    try:
        @register_encoder("mlp")
        class _DupEncoder(MLPEncoder):
            pass
        raise AssertionError("expected ValueError on duplicate encoder registration")
    except ValueError:
        pass
    assert "mlp" in ENCODER_REGISTRY
    print("[duplicate encoder registration raises] PASS")


def test_duplicate_scorer_registration_raises():
    try:
        @register_scorer("cosine_temperature")
        class _DupScorer(DotProductScorer):
            pass
        raise AssertionError("expected ValueError on duplicate scorer registration")
    except ValueError:
        pass
    assert "cosine_temperature" in SCORER_REGISTRY
    print("[duplicate scorer registration raises] PASS")


def test_duplicate_loss_registration_raises():
    try:
        @register_loss("contrastive_bce")
        class _DupLoss(ContrastiveBCELoss):
            pass
        raise AssertionError("expected ValueError on duplicate loss registration")
    except ValueError:
        pass
    assert "contrastive_bce" in LOSS_REGISTRY
    print("[duplicate loss registration raises] PASS")


if __name__ == "__main__":
    test_builtin_encoder_names_registered()
    test_builtin_scorer_names_registered()
    test_builtin_loss_names_registered()
    test_duplicate_encoder_registration_raises()
    test_duplicate_scorer_registration_raises()
    test_duplicate_loss_registration_raises()
    print("\nAll registry tests passed.")
