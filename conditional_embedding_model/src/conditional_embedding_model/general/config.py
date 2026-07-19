# Description: Dataclass-based configuration for building a CEModel (encoders + scorer).

import json
from dataclasses import dataclass, field, asdict


@dataclass
class EncoderConfig:
    name: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("EncoderConfig.name must be non-empty")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(name=d["name"], params=dict(d.get("params", {})))


@dataclass
class ScorerConfig:
    name: str = "cosine_temperature"
    params: dict = field(default_factory=lambda: {"temperature": 0.07})

    def __post_init__(self):
        if not self.name:
            raise ValueError("ScorerConfig.name must be non-empty")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(name=d.get("name", "cosine_temperature"),
                    params=dict(d.get("params", {"temperature": 0.07})))


@dataclass
class CEModelConfig:
    embedding_dim: int
    center_encoder: EncoderConfig
    context_encoder: EncoderConfig
    scorer: ScorerConfig = field(default_factory=ScorerConfig)

    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {self.embedding_dim}")
        if isinstance(self.center_encoder, dict):
            self.center_encoder = EncoderConfig.from_dict(self.center_encoder)
        if isinstance(self.context_encoder, dict):
            self.context_encoder = EncoderConfig.from_dict(self.context_encoder)
        if isinstance(self.scorer, dict):
            self.scorer = ScorerConfig.from_dict(self.scorer)

    def to_dict(self):
        return {
            "embedding_dim": self.embedding_dim,
            "center_encoder": self.center_encoder.to_dict(),
            "context_encoder": self.context_encoder.to_dict(),
            "scorer": self.scorer.to_dict(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            embedding_dim=d["embedding_dim"],
            center_encoder=EncoderConfig.from_dict(d["center_encoder"]),
            context_encoder=EncoderConfig.from_dict(d["context_encoder"]),
            scorer=ScorerConfig.from_dict(d["scorer"]) if "scorer" in d else ScorerConfig(),
        )

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, path):
        path = str(path)
        if path.endswith(".yaml") or path.endswith(".yml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "Reading a .yaml config requires PyYAML, which is not a declared "
                    "dependency of this package. Install it or use a .json config file."
                ) from e
            with open(path) as f:
                d = yaml.safe_load(f)
        else:
            with open(path) as f:
                d = json.load(f)
        return cls.from_dict(d)
