# Description: Abstract base class for CEModel encoders (center or context).

import abc
from torch import nn


class Encoder(nn.Module, abc.ABC):
    """Common interface for center/context encoders.

    Subclasses must set `self.output_dim` in `__init__` and declare a class-level
    `input_spec` mapping input key -> human-readable shape doc.
    """

    input_spec: dict = {}

    def __init__(self):
        super().__init__()
        self.output_dim: int = None

    def validate_inputs(self, inputs: dict):
        missing = [key for key in self.input_spec if key not in inputs]
        if missing:
            raise ValueError(
                f"{type(self).__name__} missing required input(s) {missing}; "
                f"expected keys {sorted(self.input_spec)}"
            )

    @abc.abstractmethod
    def forward(self, inputs: dict):
        """inputs: dict[str, Tensor] -> Tensor[..., output_dim]"""
        raise NotImplementedError


def validate_batch(model_or_encoder, inputs: dict):
    """Check that `inputs` has every key required by `model_or_encoder.input_spec`.

    Raises a readable ValueError naming the missing key(s) and the class.
    """
    spec = getattr(model_or_encoder, "input_spec", {})
    missing = [key for key in spec if key not in inputs]
    if missing:
        raise ValueError(
            f"{type(model_or_encoder).__name__} missing required input(s) {missing}; "
            f"expected keys {sorted(spec)}"
        )
