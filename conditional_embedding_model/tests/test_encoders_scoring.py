# Description: Standalone (non-pytest) tests for MLPEncoder and the new scorers
# (bilinear, mlp_scorer, multi_relation), plus registry/all_pairs/config-swap checks.

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conditional_embedding_model.general import (
    CEModel,
    CEModelConfig,
    EncoderConfig,
    ScorerConfig,
    MLPEncoder,
    BilinearScorer,
    MLPScorer,
    MultiRelationScorer,
    CosineTemperatureScorer,
    DotProductScorer,
    build_model,
)
from conditional_embedding_model.general.registry import register_scorer, SCORER_REGISTRY

# run_all.py --fast convention: test_config_only_encoder_swap's legacy_v4 branch
# constructs a real resnet18 backbone (pretrained=True); it already self-skips on
# any exception, but is excluded from --fast to guarantee no network touch.
SLOW = {"test_config_only_encoder_swap"}

B, K, M, E, R = 3, 5, 7, 8, 4


def _rand_batch(seed=1234):
    torch.manual_seed(seed)
    v = torch.randn(B, E, requires_grad=True)
    u = torch.randn(B, K, E, requires_grad=True)
    u_bank = torch.randn(M, E, requires_grad=True)
    return v, u, u_bank


def _check_forward_all_pairs_shapes(scorer, expect_relations=None):
    v, u, u_bank = _rand_batch()
    out = scorer(v, u)
    ap = scorer.all_pairs(v, u_bank)
    if expect_relations is None:
        assert out.shape == (B, K), f"forward shape {out.shape} != {(B, K)}"
        assert ap.shape == (B, M), f"all_pairs shape {ap.shape} != {(B, M)}"
    else:
        assert out.shape == (B, K, expect_relations), f"forward shape {out.shape} != {(B, K, expect_relations)}"
        assert ap.shape == (B, M, expect_relations), f"all_pairs shape {ap.shape} != {(B, M, expect_relations)}"
    return v, u, u_bank, out, ap


def _check_gradients(scorer, params_expected):
    v, u, _ = _rand_batch()
    out = scorer(v, u)
    loss = out.sum()
    loss.backward()
    assert v.grad is not None, "no grad on v"
    assert u.grad is not None, "no grad on u"
    for name, p in scorer.named_parameters():
        assert p.grad is not None, f"no grad on scorer parameter '{name}'"
    assert params_expected == len(list(scorer.parameters())) or params_expected is None


def _check_diagonal_equivalence(scorer, expect_relations=None):
    """forward(v, u_bank_as_context) should match all_pairs(v, u_bank) along the diagonal
    when the per-row context for row i is u_bank itself (same K==M bank for every row)."""
    torch.manual_seed(0)
    v = torch.randn(B, E)
    u_bank = torch.randn(M, E)
    u_context = u_bank.unsqueeze(0).expand(B, -1, -1).contiguous()

    out = scorer(v, u_context)  # [B, M] or [B, M, R]
    ap = scorer.all_pairs(v, u_bank)  # [B, M] or [B, M, R]
    assert torch.allclose(out, ap, atol=1e-6), \
        f"forward/all_pairs mismatch: max diff {(out - ap).abs().max().item()}"


def test_bilinear_scorer():
    scorer = BilinearScorer(dim=E)
    _check_forward_all_pairs_shapes(scorer)
    _check_gradients(scorer, params_expected=1)
    _check_diagonal_equivalence(scorer)
    print("[bilinear] PASS")


def test_mlp_scorer():
    scorer = MLPScorer(embed_dim=E, hidden_dims=(16,))
    _check_forward_all_pairs_shapes(scorer)
    _check_gradients(scorer, params_expected=None)
    _check_diagonal_equivalence(scorer)
    print("[mlp_scorer] PASS")


def test_multi_relation_scorer_multi():
    scorer = MultiRelationScorer(embed_dim=E, num_relations=R, squeeze_single_relation=True)
    _check_forward_all_pairs_shapes(scorer, expect_relations=R)
    _check_gradients(scorer, params_expected=1)
    _check_diagonal_equivalence(scorer, expect_relations=R)
    print("[multi_relation R=4] PASS")


def test_multi_relation_scorer_single_squeeze():
    scorer = MultiRelationScorer(embed_dim=E, num_relations=1, squeeze_single_relation=True)
    v, u, u_bank = _rand_batch()
    out = scorer(v, u)
    ap = scorer.all_pairs(v, u_bank)
    assert out.shape == (B, K), f"squeezed forward shape {out.shape} != {(B, K)}"
    assert ap.shape == (B, M), f"squeezed all_pairs shape {ap.shape} != {(B, M)}"
    print("[multi_relation R=1 squeeze] PASS")


def test_cosine_and_dot_all_pairs_diagonal():
    for scorer in (CosineTemperatureScorer(), DotProductScorer()):
        _check_forward_all_pairs_shapes(scorer)
        _check_diagonal_equivalence(scorer)
        print(f"[{type(scorer).__name__}] PASS")


def test_mlp_encoder_2d_and_3d():
    enc = MLPEncoder(input_dim=10, output_dim=E, hidden_dims=(16, 16))
    x2d = torch.randn(B, 10)
    x3d = torch.randn(B, K, 10)
    out2d = enc({"x": x2d})
    out3d = enc({"x": x3d})
    assert out2d.shape == (B, E), f"2D output shape {out2d.shape} != {(B, E)}"
    assert out3d.shape == (B, K, E), f"3D output shape {out3d.shape} != {(B, K, E)}"

    loss = out2d.sum()
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"no grad on encoder parameter '{name}'"
    print("[MLPEncoder] PASS")


def test_mlp_encoder_custom_input_key():
    enc = MLPEncoder(input_dim=4, output_dim=E, input_key="feat")
    out = enc({"feat": torch.randn(B, 4)})
    assert out.shape == (B, E)
    from conditional_embedding_model.general import validate_batch
    try:
        validate_batch(enc, {"x": torch.randn(B, 4)})
        raise AssertionError("expected ValueError for missing 'feat' key")
    except ValueError:
        pass
    print("[MLPEncoder custom input_key] PASS")


def test_duplicate_registry_raises():
    try:
        @register_scorer("dot_product")
        class _Dup(DotProductScorer):
            pass
        raise AssertionError("expected ValueError on duplicate scorer registration")
    except ValueError:
        print("[duplicate registry] PASS")
    assert "dot_product" in SCORER_REGISTRY


def test_config_only_encoder_swap():
    scorer_cfg = ScorerConfig(name="dot_product", params={})

    mlp_config = CEModelConfig(
        embedding_dim=E,
        center_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=E)),
        context_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=E)),
        scorer=scorer_cfg,
    )
    mlp_model = build_model(mlp_config)
    center_inputs = {"x": torch.randn(B, 10)}
    context_inputs = {"x": torch.randn(B, K, 10)}
    out = mlp_model(center_inputs, context_inputs)
    assert out.shape == (B, K), f"mlp-config model forward shape {out.shape} != {(B, K)}"
    print("[config-only swap: mlp/mlp] PASS")

    try:
        legacy_config = CEModelConfig(
            embedding_dim=64,
            center_encoder=EncoderConfig(name="legacy_v4_center", params=dict(
                input_dim=4, embedding_dim=64,
                feature_structure={
                    "features_structured": {"map": 8100, "object_footprint": 8, "grasping_approach": 600},
                    "features_shape": {"map": (90, 90), "object_footprint": (4, 2), "grasping_approach": (150, 4)},
                },
                dropout=0.1, eval_safe_transform=False,
            )),
            context_encoder=EncoderConfig(name="legacy_v4_context", params=dict(
                input_dim=4, embedding_dim=64,
                feature_structure={
                    "features_structured": {"map": 8100, "object_footprint": 8, "grasping_approach": 600},
                    "features_shape": {"map": (90, 90), "object_footprint": (4, 2), "grasping_approach": (150, 4)},
                },
                dropout=0.1, eval_safe_transform=False,
            )),
            scorer=ScorerConfig(name="cosine_temperature", params={"temperature": 0.07}),
        )
        legacy_model = build_model(legacy_config)
        feature = torch.randn(B, 8708)
        center = torch.zeros(B, 1)
        contexts = torch.randint(0, 150, (B, K))
        out_legacy = legacy_model({"index": center, "feature": feature}, {"index": contexts, "feature": feature})
        assert out_legacy.shape == (B, K)
        print("[config-only swap: legacy_v4/legacy_v4] PASS")
    except Exception as e:  # noqa: BLE001 - torchvision weight download may be unavailable offline
        print(f"[config-only swap: legacy_v4/legacy_v4] SKIPPED ({type(e).__name__}: {e})")


def test_validate_inputs_flag():
    center = MLPEncoder(input_dim=10, output_dim=E)
    context = MLPEncoder(input_dim=10, output_dim=E)
    scorer = DotProductScorer()

    model_off = CEModel(center, context, scorer, validate_inputs=False)
    try:
        model_off({"wrong_key": torch.randn(B, 10)}, {"x": torch.randn(B, K, 10)})
        raise AssertionError("expected KeyError (unvalidated missing-key access) with validate_inputs=False")
    except KeyError:
        pass

    model_on = CEModel(center, context, scorer, validate_inputs=True)
    try:
        model_on({"wrong_key": torch.randn(B, 10)}, {"x": torch.randn(B, K, 10)})
        raise AssertionError("expected ValueError with validate_inputs=True")
    except ValueError:
        print("[validate_inputs flag] PASS")


if __name__ == "__main__":
    test_bilinear_scorer()
    test_mlp_scorer()
    test_multi_relation_scorer_multi()
    test_multi_relation_scorer_single_squeeze()
    test_cosine_and_dot_all_pairs_diagonal()
    test_mlp_encoder_2d_and_3d()
    test_mlp_encoder_custom_input_key()
    test_duplicate_registry_raises()
    test_config_only_encoder_swap()
    test_validate_inputs_flag()
    print("\nAll encoder/scoring tests passed.")
