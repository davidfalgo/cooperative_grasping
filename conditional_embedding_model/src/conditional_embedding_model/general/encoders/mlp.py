# Description: Generic feed-forward encoder consuming a single flat feature key.

from torch import nn

from ..registry import register_encoder
from .base import Encoder

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


@register_encoder("mlp")
class MLPEncoder(Encoder):
    """input_key: "[B, D_in] or [B, K, D_in] float" -> [..., output_dim]"""

    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64),
                 activation="relu", layer_norm=True, dropout=0.0, input_key="x"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Available: {sorted(_ACTIVATIONS)}")
        self.input_key = input_key
        self.input_spec = {input_key: "[B, D_in] or [B, K, D_in] float"}
        self.output_dim = output_dim

        act_cls = _ACTIVATIONS[activation]
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs[self.input_key])
