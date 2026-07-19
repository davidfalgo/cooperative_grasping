# Description: Decorator-based string registries for encoders, scorers, and losses.

ENCODER_REGISTRY = {}
SCORER_REGISTRY = {}
LOSS_REGISTRY = {}


def _register(registry, name):
    def decorator(cls):
        if name in registry:
            raise ValueError(f"'{name}' is already registered in {registry!r}")
        registry[name] = cls
        return cls
    return decorator


def register_encoder(name):
    return _register(ENCODER_REGISTRY, name)


def register_scorer(name):
    return _register(SCORER_REGISTRY, name)


def register_loss(name):
    return _register(LOSS_REGISTRY, name)


def build_encoder(cfg):
    if cfg.name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{cfg.name}'. Registered: {sorted(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[cfg.name](**cfg.params)


def build_scorer(cfg):
    if cfg.name not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer '{cfg.name}'. Registered: {sorted(SCORER_REGISTRY)}")
    return SCORER_REGISTRY[cfg.name](**cfg.params)
