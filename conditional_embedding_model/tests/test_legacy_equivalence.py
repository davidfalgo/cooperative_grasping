# Description: Standalone (non-pytest) verification that CEModel + legacy shim
# reproduces the frozen v4/v5 model classes bit-for-bit.

import os
import sys
import tempfile

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conditional_embedding_model.models.center_embedding_resnet import (
    CenterEmbeddingResNetv4,
    ContextEmbeddingResNetv4,
    CenterEmbeddingResNetv5,
    ContextEmbeddingResNetv5,
)
from conditional_embedding_model.general import CEModel, CEModelConfig, EncoderConfig, ScorerConfig, parse_legacy_filename
from conditional_embedding_model.general.encoders.legacy import LegacyV4CenterEncoder

FEATURE_STRUCTURE = {
    "features_structured": {"map": 8100, "object_footprint": 8, "grasping_approach": 600},
    "features_shape": {"map": (90, 90), "object_footprint": (4, 2), "grasping_approach": (150, 4)},
}


def prediction_temperature(center, contexts_and_negatives, net_v, net_u, feature, temperature=0.07):
    v = torch.nn.functional.normalize(net_v(center, feature), dim=-1)
    u = torch.nn.functional.normalize(net_u(contexts_and_negatives, feature), dim=-1)
    pred = torch.bmm(v.unsqueeze(1), u.permute(0, 2, 1)) / temperature
    return pred.squeeze(1)


def _make_batch():
    torch.manual_seed(1234)
    feature = torch.rand(4, 8708)
    center = torch.zeros(4, 1)
    contexts = torch.randint(0, 150, (4, 10))
    return center, contexts, feature


def _run_equivalence(center_cls, context_cls, center_name, context_name, tag):
    torch.manual_seed(1234)
    legacy_center = center_cls(input_dim=4, embedding_dim=64, feature_structure=FEATURE_STRUCTURE, dropout=0.1)
    legacy_context = context_cls(input_dim=4, embedding_dim=64, feature_structure=FEATURE_STRUCTURE, dropout=0.1)
    legacy_seq = nn.Sequential(legacy_center, legacy_context)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        tmp_path = f.name
    torch.save(legacy_seq.state_dict(), tmp_path)

    config = CEModelConfig(
        embedding_dim=64,
        center_encoder=EncoderConfig(name=center_name, params=dict(
            input_dim=4, embedding_dim=64, feature_structure=FEATURE_STRUCTURE, dropout=0.1,
            eval_safe_transform=False,
        )),
        context_encoder=EncoderConfig(name=context_name, params=dict(
            input_dim=4, embedding_dim=64, feature_structure=FEATURE_STRUCTURE, dropout=0.1,
            eval_safe_transform=False,
        )),
        scorer=ScorerConfig(name="cosine_temperature", params={"temperature": 0.07}),
    )
    cemodel = CEModel.from_legacy_checkpoint(tmp_path, config)

    legacy_seq.eval()
    cemodel.eval()

    center, contexts, feature = _make_batch()

    torch.manual_seed(0)
    legacy_logits = prediction_temperature(center, contexts, legacy_seq[0], legacy_seq[1], feature, 0.07)

    torch.manual_seed(0)
    new_logits = cemodel({"index": center, "feature": feature}, {"index": contexts, "feature": feature})

    assert torch.allclose(legacy_logits, new_logits, atol=1e-6), \
        f"[{tag}] logits mismatch: max diff {(legacy_logits - new_logits).abs().max().item()}"

    legacy_keys = set(torch.load(tmp_path).keys())
    new_state = cemodel.to_legacy_state_dict()
    assert set(new_state.keys()) == legacy_keys, \
        f"[{tag}] key set mismatch: {set(new_state.keys()) ^ legacy_keys}"
    legacy_sd = torch.load(tmp_path)
    for k in legacy_keys:
        assert torch.equal(new_state[k], legacy_sd[k]), f"[{tag}] tensor mismatch at key '{k}'"

    os.remove(tmp_path)
    print(f"[{tag}] PASS: logits match (atol=1e-6), state_dict round-trips exactly ({len(legacy_keys)} keys)")


def test_v4_equivalence():
    _run_equivalence(CenterEmbeddingResNetv4, ContextEmbeddingResNetv4, "legacy_v4_center", "legacy_v4_context", "v4")


def test_v5_equivalence():
    _run_equivalence(CenterEmbeddingResNetv5, ContextEmbeddingResNetv5, "legacy_v5_center", "legacy_v5_context", "v5")


def test_parse_legacy_filename():
    parsed = parse_legacy_filename("swmodel_v4_E83_B32_topk_0.912.pth")
    expected = {"model_version": "v4", "embedding_size": 83, "batch_size": 32, "hit_at_1": 0.912}
    assert parsed == expected, f"parse_legacy_filename mismatch: {parsed} != {expected}"
    print(f"[parse_legacy_filename] PASS: {parsed}")

    try:
        parse_legacy_filename("not_a_legacy_filename.pth")
        raise AssertionError("expected ValueError on garbage filename")
    except ValueError:
        print("[parse_legacy_filename] PASS: garbage name raises ValueError")


def test_eval_safe_transform_determinism():
    enc = LegacyV4CenterEncoder(input_dim=4, embedding_dim=64,
                                 feature_structure=FEATURE_STRUCTURE,
                                 eval_safe_transform=True)
    enc.eval()
    center, _, feature = _make_batch()
    out1 = enc({"index": center, "feature": feature})
    out2 = enc({"index": center, "feature": feature})
    assert torch.allclose(out1, out2), "eval_safe_transform=True should be deterministic in eval mode"
    print("[eval_safe_transform] PASS: deterministic in eval mode")


if __name__ == "__main__":
    test_v4_equivalence()
    test_v5_equivalence()
    test_parse_legacy_filename()
    test_eval_safe_transform_determinism()
    print("\nAll legacy-equivalence tests passed.")
