# Description: Standalone (non-pytest) tests for general/config.py: to_dict/from_dict
# round-trip equality, save/from_file JSON round-trip, and invalid-config errors.

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from conditional_embedding_model.general import (
    CEModelConfig, EncoderConfig, ScorerConfig, build_model,
)


def _sample_config():
    return CEModelConfig(
        embedding_dim=8,
        center_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=8, hidden_dims=(16,))),
        context_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=8, hidden_dims=(16,))),
        scorer=ScorerConfig(name="cosine_temperature", params={"temperature": 0.05}),
    )


def test_to_dict_from_dict_roundtrip():
    config = _sample_config()
    roundtripped = CEModelConfig.from_dict(config.to_dict())
    assert roundtripped == config, f"roundtrip mismatch: {roundtripped} != {config}"

    # from_dict also accepts raw nested dicts (not just EncoderConfig/ScorerConfig
    # instances) via __post_init__ coercion -- exercise that path directly too.
    raw = CEModelConfig(
        embedding_dim=8,
        center_encoder={"name": "mlp", "params": {"input_dim": 10, "output_dim": 8}},
        context_encoder={"name": "mlp", "params": {"input_dim": 10, "output_dim": 8}},
    )
    assert isinstance(raw.center_encoder, EncoderConfig)
    assert isinstance(raw.scorer, ScorerConfig)
    print("[to_dict/from_dict roundtrip] PASS")


def test_save_from_file_json_roundtrip():
    # JSON has no tuple type (round-trips tuples as lists), so use list-only
    # params here -- the tuple-vs-list case is exercised by the in-memory
    # to_dict/from_dict roundtrip above instead.
    config = CEModelConfig(
        embedding_dim=8,
        center_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=8, hidden_dims=[16])),
        context_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=8, hidden_dims=[16])),
        scorer=ScorerConfig(name="cosine_temperature", params={"temperature": 0.05}),
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        config.save(path)
        loaded = CEModelConfig.from_file(path)
        assert loaded == config, f"JSON roundtrip mismatch: {loaded} != {config}"
    print("[save/from_file JSON roundtrip] PASS")


def test_invalid_embedding_dim_raises():
    try:
        CEModelConfig(
            embedding_dim=0,
            center_encoder=EncoderConfig(name="mlp", params={}),
            context_encoder=EncoderConfig(name="mlp", params={}),
        )
        raise AssertionError("expected ValueError for embedding_dim=0")
    except ValueError:
        pass

    try:
        CEModelConfig(
            embedding_dim=-5,
            center_encoder=EncoderConfig(name="mlp", params={}),
            context_encoder=EncoderConfig(name="mlp", params={}),
        )
        raise AssertionError("expected ValueError for embedding_dim=-5")
    except ValueError:
        pass
    print("[invalid embedding_dim raises] PASS")


def test_invalid_encoder_name_raises():
    try:
        EncoderConfig(name="")
        raise AssertionError("expected ValueError for empty EncoderConfig.name")
    except ValueError:
        pass

    try:
        ScorerConfig(name="")
        raise AssertionError("expected ValueError for empty ScorerConfig.name")
    except ValueError:
        pass
    print("[invalid encoder/scorer name raises] PASS")


def test_build_model_unknown_registry_name_raises():
    config = CEModelConfig(
        embedding_dim=8,
        center_encoder=EncoderConfig(name="not_a_real_encoder", params={}),
        context_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=8)),
    )
    try:
        build_model(config)
        raise AssertionError("expected ValueError for unknown encoder registry name")
    except ValueError as e:
        assert "not_a_real_encoder" in str(e)

    config2 = CEModelConfig(
        embedding_dim=8,
        center_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=8)),
        context_encoder=EncoderConfig(name="mlp", params=dict(input_dim=10, output_dim=8)),
        scorer=ScorerConfig(name="not_a_real_scorer", params={}),
    )
    try:
        build_model(config2)
        raise AssertionError("expected ValueError for unknown scorer registry name")
    except ValueError as e:
        assert "not_a_real_scorer" in str(e)
    print("[build_model unknown registry name raises] PASS")


if __name__ == "__main__":
    test_to_dict_from_dict_roundtrip()
    test_save_from_file_json_roundtrip()
    test_invalid_embedding_dim_raises()
    test_invalid_encoder_name_raises()
    test_build_model_unknown_registry_name_raises()
    print("\nAll config tests passed.")
