# Description: Pluggable metric logging backends for Trainer. `wandb` is imported
# lazily inside WandbLogger only, so importing this module never pulls it in.

import typing


class MetricLogger(typing.Protocol):
    def log(self, metrics: dict, step: int) -> None: ...
    def finish(self) -> None: ...


class NullLogger:
    def log(self, metrics: dict, step: int) -> None:
        pass

    def finish(self) -> None:
        pass


class ConsoleLogger:
    def __init__(self, print_fn=print):
        self._print = print_fn

    def log(self, metrics: dict, step: int) -> None:
        formatted = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in metrics.items())
        self._print(f"[step {step}] {formatted}")

    def finish(self) -> None:
        pass


class WandbLogger:
    """Wraps an already-created wandb run; never calls wandb.init itself."""

    def __init__(self, run):
        self._run = run

    def log(self, metrics: dict, step: int) -> None:
        import wandb  # noqa: F401 - imported lazily to keep wandb out of module import

        self._run.log(metrics, step=step)

    def finish(self) -> None:
        self._run.finish()
