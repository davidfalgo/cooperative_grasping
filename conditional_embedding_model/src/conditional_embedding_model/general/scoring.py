# Description: Parameter-free similarity scorers between center and context embeddings.

import torch
from torch import nn
import torch.nn.functional as F

from .registry import register_scorer


class Scorer(nn.Module):
    """v: [B, E], u: [B, K, E] -> [B, K]"""

    def forward(self, v, u):
        raise NotImplementedError

    def all_pairs(self, v, u):
        """v: [B, E], u: [M, E] -> [B, M]"""
        raise NotImplementedError


@register_scorer("cosine_temperature")
class CosineTemperatureScorer(Scorer):
    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, v, u):
        if self.normalize:
            v = F.normalize(v, dim=-1)
            u = F.normalize(u, dim=-1)
        pred = torch.bmm(v.unsqueeze(1), u.permute(0, 2, 1))
        if self.normalize:
            pred = pred / self.temperature
        return pred.squeeze(1)

    def all_pairs(self, v, u):
        if self.normalize:
            v = F.normalize(v, dim=-1)
            u = F.normalize(u, dim=-1)
        pred = v @ u.t()
        if self.normalize:
            pred = pred / self.temperature
        return pred


@register_scorer("dot_product")
class DotProductScorer(Scorer):
    def forward(self, v, u):
        pred = torch.bmm(v.unsqueeze(1), u.permute(0, 2, 1))
        return pred.squeeze(1)

    def all_pairs(self, v, u):
        return v @ u.t()


@register_scorer("bilinear")
class BilinearScorer(Scorer):
    """Learned bilinear form v^T W u."""

    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, v, u):
        return torch.einsum("be,ef,bkf->bk", v, self.W, u)

    def all_pairs(self, v, u):
        return v @ self.W @ u.t()


@register_scorer("mlp_scorer")
class MLPScorer(Scorer):
    """MLP over [v, u, v*u] per (center, context) pair.

    Cost is O(B*K*E) (or O(B*M*E) for all_pairs) since the concatenation is
    materialized explicitly before the MLP is applied.
    """

    def __init__(self, embed_dim, hidden_dims=(64,), activation="relu", layer_norm=True, dropout=0.0):
        super().__init__()
        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        dims = [3 * embed_dim] + list(hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def _score(self, v, u):
        """v: [B, N, E], u: [B, N, E] (already broadcast to the same shape) -> [B, N]"""
        pair = torch.cat([v, u, v * u], dim=-1)
        return self.net(pair).squeeze(-1)

    def forward(self, v, u):
        k = u.shape[1]
        v_exp = v.unsqueeze(1).expand(-1, k, -1)
        return self._score(v_exp, u)

    def all_pairs(self, v, u):
        b, m = v.shape[0], u.shape[0]
        v_exp = v.unsqueeze(1).expand(-1, m, -1)
        u_exp = u.unsqueeze(0).expand(b, -1, -1)
        return self._score(v_exp, u_exp)


@register_scorer("multi_relation")
class MultiRelationScorer(Scorer):
    """Per-relation bilinear maps: v: [B,E], u: [B,K,E] -> [B,K,R] (or [B,K] if R==1 and squeezed)."""

    def __init__(self, embed_dim, num_relations, temperature=None, normalize=False,
                 squeeze_single_relation=True):
        super().__init__()
        self.num_relations = num_relations
        self.temperature = temperature
        self.normalize = normalize
        self.squeeze_single_relation = squeeze_single_relation
        self.W = nn.Parameter(torch.empty(num_relations, embed_dim, embed_dim))
        for r in range(num_relations):
            nn.init.xavier_uniform_(self.W[r])

    def _maybe_normalize(self, v, u):
        if self.normalize:
            v = F.normalize(v, dim=-1)
            u = F.normalize(u, dim=-1)
        return v, u

    def _maybe_temperature(self, pred):
        if self.normalize and self.temperature is not None:
            pred = pred / self.temperature
        return pred

    def _maybe_squeeze(self, pred):
        if self.num_relations == 1 and self.squeeze_single_relation:
            return pred.squeeze(-1)
        return pred

    def forward(self, v, u):
        v, u = self._maybe_normalize(v, u)
        pred = torch.einsum("be,ref,bkf->bkr", v, self.W, u)
        pred = self._maybe_temperature(pred)
        return self._maybe_squeeze(pred)

    def all_pairs(self, v, u):
        v, u = self._maybe_normalize(v, u)
        pred = torch.einsum("be,ref,mf->bmr", v, self.W, u)
        pred = self._maybe_temperature(pred)
        return self._maybe_squeeze(pred)
