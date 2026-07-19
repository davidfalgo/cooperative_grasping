# `general` — generalized Conditional Embedding (CE) model package

A reusable, config-driven rewrite of the paper's `CenterEmbeddingResNetv4`-era
dual-network CE model: pluggable encoders, pluggable affinity scorers, and a
generic training loop, while keeping every legacy checkpoint loadable **without
any file conversion**. Design rationale and full analysis live in
`plans/ce-model-generalization/PLAN.md`; this file is the user-facing reference.

## Architecture overview

`CEModel` (`core.py`) is composition, not inheritance: a center encoder, a context
encoder, and a scorer, wired together at `build_model(config)` time.

```python
class CEModel(nn.Module):
    def __init__(self, center_encoder, context_encoder, scorer, validate_inputs=False): ...
    def embed_center(self, inputs: dict) -> Tensor: ...          # -> [B, E]
    def embed_context(self, inputs: dict) -> Tensor: ...         # -> [B, K, E]
    def forward(self, center_inputs, context_inputs) -> Tensor:  # -> [B, K] (or [B,K,R])
    def score_all_pairs(self, center_inputs, context_inputs) -> Tensor:  # -> [B, M] (or [B,M,R])
```

**Dict-input contract.** Encoders don't take positional tensors — they take a
`dict[str, Tensor]` and pull out the keys they declare in `input_spec`. This is
what makes encoders swappable purely via config: an `MLPEncoder` wants `{"x": ...}`
(configurable via `input_key`); a `legacy_v4_center` encoder wants
`{"index": ..., "feature": ...}`. Nothing in `CEModel` cares which.

**Affinity tensor shapes** (all scorers implement both `forward` and `all_pairs`):

| Shape | Meaning | Producer |
|---|---|---|
| `[B, K]` | per-center scores over its K padded candidates (default; consumed by masked BCE + Hit@k) | `forward` |
| `[B, K, R]` | multi-relation affinities; labels/masks broadcast over `R`; metrics report per-relation + macro-average | `forward` + `MultiRelationScorer` |
| `[B, M]` | all-pairs against a flat candidate bank (deployment-style full retrieval) | `score_all_pairs` |
| `[B, M, R]` | all-pairs, multi-relation | `score_all_pairs` + `MultiRelationScorer` |

## Config reference

Three dataclasses (`config.py`), pure stdlib (`json`) — no new dependency; `.yaml`
loading is opportunistic (`import yaml` only inside `from_file`, with a clear
`ImportError` if PyYAML isn't installed):

```python
@dataclass
class EncoderConfig:
    name: str                      # registry key, see table below
    params: dict = field(default_factory=dict)

@dataclass
class ScorerConfig:
    name: str = "cosine_temperature"
    params: dict = field(default_factory=lambda: {"temperature": 0.07})

@dataclass
class CEModelConfig:
    embedding_dim: int
    center_encoder: EncoderConfig
    context_encoder: EncoderConfig
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
```

Every dataclass has `to_dict`/`from_dict`; `CEModelConfig` additionally has
`save(path)` / `from_file(path)` (JSON, or YAML if PyYAML is available).
`build_model(config) -> CEModel` resolves registry names and instantiates.

**Registry names** (`registry.py`; `register_encoder`/`register_scorer`/
`register_loss` decorators — duplicate names raise `ValueError`, so third-party
code can register new components without touching package internals):

| Registry | Built-in names |
|---|---|
| Encoders | `legacy_v4_center`, `legacy_v4_context`, `legacy_v5_center`, `legacy_v5_context`, `mlp` |
| Scorers | `cosine_temperature`, `dot_product`, `bilinear`, `mlp_scorer`, `multi_relation` |
| Losses | `contrastive_bce` |

**The current v4 setup expressed in this schema** (this exact config is what
`tests/test_legacy_equivalence.py` constructs and checks bit-for-bit against the
frozen `CenterEmbeddingResNetv4`/`ContextEmbeddingResNetv4` classes):

```python
FEATURE_STRUCTURE = {
    "features_structured": {"map": 8100, "object_footprint": 8, "grasping_approach": 600},
    "features_shape": {"map": (90, 90), "object_footprint": (4, 2), "grasping_approach": (150, 4)},
}
v4_config = CEModelConfig(
    embedding_dim=64,
    center_encoder=EncoderConfig("legacy_v4_center", {
        "input_dim": 4, "embedding_dim": 64,
        "feature_structure": FEATURE_STRUCTURE,
        "dropout": 0.1, "eval_safe_transform": False}),
    context_encoder=EncoderConfig("legacy_v4_context", {
        "input_dim": 4, "embedding_dim": 64,
        "feature_structure": FEATURE_STRUCTURE, "dropout": 0.1}),
    scorer=ScorerConfig("cosine_temperature", {"temperature": 0.07}),
)
model = build_model(v4_config)
```

## Legacy checkpoint how-to

IBEX checkpoints (and the notebooks' `config/weights/model*.pth`) follow the
naming convention:

```
swmodel_{model_version}_E{embedding_size}_B{batch_size}_topk_{val_hit_at_1:.3f}.pth
e.g. swmodel_v4_E83_B32_topk_0.912.pth
```

and contain **only** a flat `nn.Sequential(center_model, context_model).state_dict()`
— keys prefixed `0.*` (center) / `1.*` (context), no optimizer/scheduler/config.

```python
from conditional_embedding_model.general import CEModel, parse_legacy_filename

parsed = parse_legacy_filename("swmodel_v4_E83_B32_topk_0.912.pth")
# {"model_version": "v4", "embedding_size": 83, "batch_size": 32, "hit_at_1": 0.912}

# You still need feature_structure and dropout (not recoverable from the
# filename or state dict) to build the matching config, then:
model = CEModel.from_legacy_checkpoint(path, config)   # remaps 0.*/1.* -> named submodules
...
legacy_state_dict = model.to_legacy_state_dict()        # remaps back for round-tripping
```

**v4 and v5 state dicts are mutually key-incompatible.** v5 removes the LayerNorms
v4 has throughout its head/embedder MLPs; since LayerNorms carry parameters,
removing them shifts every subsequent `nn.Sequential` child index. A v4 checkpoint
cannot be loaded into a v5-shaped model (or vice versa) even though both expose the
same `legacy_v{4,5}_{center,context}` registry interface — always match the
version parsed from the filename to the encoder name you build with.

`general.training.load_model(path, config=None)` is a single dispatcher: a dict
with `"format_version"` routes to the rich-checkpoint path (config embedded, no
extra argument needed); a flat `0.*`/`1.*` dict routes to
`from_legacy_checkpoint` if `config=` is supplied, else raises a `ValueError`
telling you what's missing (parsed version/embedding_size if the filename
matches the convention, or a hint to pass `config=` explicitly if it doesn't).

## Training

```python
class Trainer:
    def __init__(self, model, loss, optimizer, *, scheduler=None, device="cuda:0",
                 amp=False, grad_clip_norm=None, logger=None,
                 early_stopping=None, checkpoint_dir=None): ...
    def fit(self, train_loader, val_loader=None, epochs=80) -> dict:      # history
    def evaluate(self, loader, ks=(1, 3, 5)) -> dict:                     # loss, hit_at_k, classification metrics
    def save_checkpoint(self, path, *, epoch=None, metrics=None): ...
    @classmethod
    def resume(cls, path, *, optimizer_factory, scheduler_factory=None, loss,
               device="cuda:0", amp=False, ...) -> "Trainer": ...
```

**Batch contract**: loaders yield `((center_inputs: dict, context_inputs: dict),
label: Float[B,K], mask: Float[B,K])`. `make_collate(max_length, index_key="index",
feature_key="feature")` builds this from raw `(center, context, negative, feature)`
dataset items (long-dtype index tensors — the proper fix for the legacy `x.long()`
FIXME, see below). `LegacyBatchAdapter` instead wraps the **frozen**
`batchify_wrapper` 5-tuple output `(center, contexts_negatives, mask, label,
feature)` into the same contract, so the existing dataloader plugs in unmodified.

**Loss** (`losses.py`) — `ContrastiveBCELoss(pos_weight_mode="dynamic",
normalization="valid_pairs")`:
- `"valid_pairs"` (**default**) — divides by `mask.sum().clamp_min(1)`, with an
  epsilon-guarded pos/neg ratio. This is the corrected behavior, fixing the known
  `DynamicSigmoidBCELoss` bug where loss scaled with the padding ratio (dividing by
  batch size instead of valid-pair count).
- `"legacy_batch"` — bit-reproduces `DynamicSigmoidBCELoss` exactly: per-row
  `neg_weight = pos_count / neg_count` with **no** clamp (an all-positive-among-valid
  row produces `inf`/`nan`, matching the legacy hazard rather than silently fixing
  it), sum divided by `label.shape[0]` (batch size). Use this only when you need to
  bit-reproduce a published sweep number; use `"valid_pairs"` for anything new.

AMP is opt-in (`amp=True`, uses `torch.autocast` + `torch.amp.GradScaler`; legacy
training had none). Gradient clipping is opt-in via `grad_clip_norm` (applied after
`unscale_`). Scheduler defaults to whatever you pass — `ReduceLROnPlateau` on val
loss is a natural choice and is what `fit` special-cases (steps on `val_loss` if
present, else the epoch's train loss).

**Rich checkpoint format v1** (new; coexists with legacy loading):
`{"format_version": 1, "model_config", "model_state", "optimizer_state",
"scheduler_state", "scaler_state", "epoch", "global_step", "best_metric", "rng"}`
— `save_checkpoint`/`Trainer.resume` round-trip this exactly, including RNG state
(torch/numpy/python/cuda) for reproducible continuation; `tests/test_training.py`'s
`test_resume_determinism` checks a resumed run reproduces uninterrupted per-step
loss to `1e-6`.

## Known legacy quirks (preserved on purpose — do not "fix" without discussion)

| Quirk | Preserved behavior | Opt-in fix |
|---|---|---|
| **Augmentation at inference** — v3–v5 `forward()` applies `RandomHorizontalFlip`/`RandomAffine`/`ColorJitter` with no `if self.training:` guard, so eval-mode map images are randomly perturbed. | Legacy adapters default to `eval_safe_transform=False` — bit-matches historical eval behavior (including the bug), needed to reproduce published benchmark numbers. | Pass `eval_safe_transform=True` to `LegacyV{4,5}{Center,Context}Encoder`: swaps in a deterministic `Resize→Grayscale→Normalize`-only transform whenever `model.eval()` is active. `test_eval_safe_transform_determinism` checks this is actually deterministic. |
| **`x.long()` FIXME** — legacy `forward()` methods cast index tensors to `int64` internally rather than receiving them pre-cast. | Legacy `batchify_wrapper` and legacy `forward()` casts are untouched (frozen, per CLAUDE.md §10). | The new collate path (`make_collate`, `LegacyBatchAdapter`) constructs index tensors as `torch.long` from the start — the proper fix lives entirely on the new side. |
| **`pretrained=True` resnet construction** — deprecated PyTorch API (vs. `weights=ResNet18_Weights.DEFAULT`) used throughout v2–v5. | Legacy classes unchanged (harmless: ImageNet weights are immediately overwritten when loading a real checkpoint). | New non-legacy image encoders (none shipped yet) should take an explicit `pretrained: bool` and use the `weights=` API instead. |
| **`DynamicSigmoidBCELoss` batch-size normalization bug** — divides by `label.shape[0]` (batch size) instead of the valid-pair count, so loss scale drifts with the padding ratio. | Selectable via `ContrastiveBCELoss(normalization="legacy_batch")` — see Training above. | Use the `"valid_pairs"` default for all new training. |
| **`scripts/evaluate.py` stale imports** (`from TrainNS_train import *`, `from TrainNS_dataloader import *`) — refers to pre-reorganization module names, fails to import. | Out of scope for this refactor; documented as known debt. | Would need `from conditional_embedding_model.training import *` / `from conditional_embedding_model.data import *`, or better, migrate to `general.training.Trainer.evaluate`. |
| **`get_config_path` undefined** — called at `trainer.py:722` and `evaluate.py:477,486` but defined nowhere; `sweep_weights_bias` cannot run outside the environment that historically defined it. | Out of scope; documented as known debt. | N/A — needs the original definition recovered or a replacement written. |
| **`notebook 03_evaluation.ipynb`** imports a nonexistent `evaluation.evaluate` module. | Out of scope; documented as known debt. | Migrate the notebook to `general.training.Trainer.evaluate` / `evaluate_model_topk`. |
| **Dead `import yaml`** in `trainer.py`, `center_embedding_resnet.py`, `evaluate.py` — pyyaml isn't a `pyproject.toml` dependency (only in the conda `environment.yml`). | Left as-is (out of scope; harmless as long as the conda env is used). | Remove the dead imports, or add `pyyaml` as a real dependency if `.yaml` config loading (`CEModelConfig.from_file`) becomes commonly used. |
| **Redundant repo-root `setup.py`** — flat `find_packages()` cannot discover the `src`-layout package; `pyproject.toml` is the authoritative packaging file. | Not deleted (see Development note below — flagged, not removed, pending team confirmation). | Delete once confirmed nothing depends on it (e.g. IBEX deployment scripts). |

## Development

Run the test suite with plain `python` (no pytest dependency is installed or
required):

```bash
python conditional_embedding_model/tests/run_all.py          # full suite (~1 min; downloads/uses cached resnet18 weights once)
python conditional_embedding_model/tests/run_all.py --fast    # skips slow/network-touching tests; offline, well under a minute
```

Each `tests/test_*.py` file is also directly runnable
(`python tests/test_config.py`) and is written as plain-assert functions
(`test_*` names, bare `assert`, a `__main__` block) so it is already
pytest-compatible — adopting pytest later is a zero-cost drop-in.

**Proposal (not done in this task):** add `pytest` under a
`[project.optional-dependencies] dev` extra in `pyproject.toml` so contributors
can run `pytest conditional_embedding_model/tests/` with fixtures/parametrize/
`-k` filtering, while the base install stays test-runner-free. This wasn't added
here because expanding install-time dependency surface wasn't in scope for this
task — it needs a maintainer decision, not a unilateral addition.

**Also flagged (not done in this task):** the repo-root `setup.py` looks
redundant now that `conditional_embedding_model/pyproject.toml` is the
authoritative, working packaging file (`setup.py`'s flat `find_packages()` can't
even discover the `src`-layout package). It's left in place pending team
confirmation that nothing else (e.g. an IBEX deployment script) depends on it.
