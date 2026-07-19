# Description: Standalone (non-pytest) tests for general/training/trainer.py's
# `load_model` dispatcher: rich (format_version=1) checkpoints route through
# CEModelConfig/build_model; flat legacy `0.*`/`1.*` state dicts route to the
# legacy path (or raise a clear, actionable error when no config is supplied).

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conditional_embedding_model.general import CEModelConfig, EncoderConfig, ScorerConfig, build_model
from conditional_embedding_model.general.training import load_model


def _tiny_config():
    return CEModelConfig(
        embedding_dim=8,
        center_encoder=EncoderConfig(name="mlp", params=dict(input_dim=6, output_dim=8, hidden_dims=(12,))),
        context_encoder=EncoderConfig(name="mlp", params=dict(input_dim=6, output_dim=8, hidden_dims=(12,))),
        scorer=ScorerConfig(name="cosine_temperature", params={"temperature": 0.07}),
    )


def test_rich_checkpoint_routes_to_config_path():
    torch.manual_seed(0)
    model = build_model(_tiny_config())
    model.eval()

    center_inputs = {"x": torch.randn(4, 6)}
    context_inputs = {"x": torch.randn(4, 5, 6)}
    with torch.no_grad():
        logits_before = model(center_inputs, context_inputs)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        ckpt = {
            "format_version": 1,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
        }
        torch.save(ckpt, path)

        loaded = load_model(path)
        assert loaded.config == model.config, "loaded model_config does not match original"
        loaded.eval()
        with torch.no_grad():
            logits_after = loaded(center_inputs, context_inputs)
        assert torch.allclose(logits_before, logits_after, atol=1e-6), \
            f"rich-checkpoint round-trip logits mismatch: max diff " \
            f"{(logits_before - logits_after).abs().max().item()}"
    print("[rich checkpoint -> config path] PASS")


def _save_flat_legacy_dict(path):
    flat = {"0.embedder.weight": torch.randn(3, 3), "1.context_gru.weight_ih_l0": torch.randn(4, 4)}
    torch.save(flat, path)


def test_legacy_flat_dict_without_matching_filename_raises_actionable_error():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "not_a_legacy_filename.pth")
        _save_flat_legacy_dict(path)
        try:
            load_model(path)
            raise AssertionError("expected ValueError for unparseable legacy filename")
        except ValueError as e:
            msg = str(e)
            assert "legacy" in msg.lower() and "config=" in msg, f"error message not actionable: {msg}"
    print("[legacy flat dict, non-matching filename] PASS (actionable ValueError)")


def test_legacy_flat_dict_with_matching_filename_but_no_config_raises():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "swmodel_v4_E8_B4_topk_0.500.pth")
        _save_flat_legacy_dict(path)
        try:
            load_model(path)
            raise AssertionError("expected ValueError when config is not supplied")
        except ValueError as e:
            msg = str(e)
            assert "v4" in msg and "config=" in msg, f"error message not actionable: {msg}"
    print("[legacy flat dict, matching filename, no config] PASS (actionable ValueError)")


def test_unrecognized_checkpoint_format_raises():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "garbage.pth")
        torch.save({"some_random_key": torch.randn(2, 2)}, path)
        try:
            load_model(path)
            raise AssertionError("expected ValueError for unrecognized checkpoint format")
        except ValueError as e:
            assert "Unrecognized checkpoint format" in str(e)
    print("[unrecognized checkpoint format] PASS")


if __name__ == "__main__":
    test_rich_checkpoint_routes_to_config_path()
    test_legacy_flat_dict_without_matching_filename_raises_actionable_error()
    test_legacy_flat_dict_with_matching_filename_but_no_config_raises()
    test_unrecognized_checkpoint_format_raises()
    print("\nAll checkpoint-dispatch tests passed.")
