# Description: Public API for the generalized CEModel package (encoders, scorers, config, core).

from .registry import (
    ENCODER_REGISTRY,
    SCORER_REGISTRY,
    LOSS_REGISTRY,
    register_encoder,
    register_scorer,
    register_loss,
    build_encoder,
    build_scorer,
)
from .encoders.base import Encoder, validate_batch
from .encoders.legacy import (
    LegacyV4CenterEncoder,
    LegacyV4ContextEncoder,
    LegacyV5CenterEncoder,
    LegacyV5ContextEncoder,
)
from .encoders.mlp import MLPEncoder
from .scoring import (
    Scorer,
    CosineTemperatureScorer,
    DotProductScorer,
    BilinearScorer,
    MLPScorer,
    MultiRelationScorer,
)
from .config import EncoderConfig, ScorerConfig, CEModelConfig
from .core import CEModel, build_model, parse_legacy_filename
from .training import (
    ContrastiveBCELoss,
    hit_at_k,
    classification_metrics,
    all_pairs_hit_at_k,
    MetricLogger,
    NullLogger,
    ConsoleLogger,
    WandbLogger,
    make_collate,
    LegacyBatchAdapter,
    Trainer,
    EarlyStopping,
    load_model,
)
