"""BFM-Zero.
Refs: BFM-Zero App. B.2; LeCAR-Lab/BFM-Zero humanoidverse/agents/nn_models.py
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


def _orthogonal_init_(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()


def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 1:
        return x.unsqueeze(0), True
    return x, False


class BallNorm(nn.Module):
    """z ← √d · z / ||z||  (paper Fig. 10 'BallNorm')."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class TruncatedNormal(pyd.Normal):
    """Gaussian truncated to [low, high]; sample uses detached clamp (official)."""

    def __init__(self, loc, scale, low: float = -1.0, high: float = 1.0, eps: float = 1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        return x - x.detach() + clamped.detach()

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore[override]
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps = eps * self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        return self._clamp(self.loc + eps)


class ResidualBlock(nn.Module):
    """x + Mish(Linear(LN(x))) — Fig. 10 inner residual."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.Mish())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class LinearBlock(nn.Module):
    """LN → Linear → optional Mish. Official `Block`."""

    def __init__(self, in_dim: int, out_dim: int, activation: bool):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim)]
        if activation:
            layers.append(nn.Mish())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class InputEmbedding(nn.Module):
    """Fig. 10 left: LN/Linear/Mish → (L-2) residual → LN/Linear/Mish → hidden/2.

    `num_layers` is Table 2 'Embedding Residual Blocks' (must be >= 2).
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 2:
            raise ValueError("InputEmbedding needs num_layers >= 2")
        layers: list[nn.Module] = [LinearBlock(in_dim, hidden_dim, activation=True)]
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(LinearBlock(hidden_dim, hidden_dim // 2, activation=True))
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResidualTrunk(nn.Module):
    """After concat of two embeddings: residual stack + LN/Linear head."""

    def __init__(self, hidden_dim: int, residual_blocks: int, out_dim: int):
        super().__init__()
        blocks = [ResidualBlock(hidden_dim) for _ in range(residual_blocks)]
        blocks.append(LinearBlock(hidden_dim, out_dim, activation=False))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ForwardCore(nn.Module):
    """One critic: embed(x,z) ⊕ embed(x,a) → residual trunk → R^{out}."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        hidden_dim: int,
        embedding_layers: int,
        residual_blocks: int,
        out_dim: int,
    ):
        super().__init__()
        self.embed_z = InputEmbedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_sa = InputEmbedding(obs_dim + action_dim, hidden_dim, embedding_layers)
        self.trunk = _ResidualTrunk(hidden_dim, residual_blocks, out_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z_emb = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_emb = self.embed_sa(torch.cat([obs, action], dim=-1))
        return self.trunk(torch.cat([sa_emb, z_emb], dim=-1))


class FBForward(nn.Module):
    """Critics F / Q_R / Q_D (Fig. 10). Ensemble of `num_parallel` cores.

    F: out_dim = z_dim. Q_R / Q_D: out_dim = 1.
    Returns [N, B, out] if N>1 else [B, out].
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        hidden_dim: int = 256,
        embedding_layers: int = 2,
        residual_blocks: int = 2,
        num_parallel: int = 2,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        if num_parallel < 1:
            raise ValueError("num_parallel must be >= 1")
        self.z_dim = z_dim
        self.num_parallel = num_parallel
        self.out_dim = z_dim if out_dim is None else out_dim
        self.ensemble = nn.ModuleList(
            [
                _ForwardCore(
                    obs_dim,
                    action_dim,
                    z_dim,
                    hidden_dim,
                    embedding_layers,
                    residual_blocks,
                    self.out_dim,
                )
                for _ in range(num_parallel)
            ]
        )
        self.apply(_orthogonal_init_)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs, squeezed = _ensure_batch(obs)
        z, _ = _ensure_batch(z)
        action, _ = _ensure_batch(action)
        outs = [core(obs, z, action) for core in self.ensemble]
        y = outs[0] if self.num_parallel == 1 else torch.stack(outs, dim=0)
        if squeezed:
            y = y.squeeze(-2)
        return y


class FBActor(nn.Module):
    """π(a | x, z): embed(x,z) ⊕ embed(x) → residual trunk → tanh → TruncatedNormal."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        z_dim: int,
        hidden_dim: int = 256,
        embedding_layers: int = 2,
        residual_blocks: int = 2,
    ):
        super().__init__()
        self.embed_z = InputEmbedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = InputEmbedding(obs_dim, hidden_dim, embedding_layers)
        self.policy = _ResidualTrunk(hidden_dim, residual_blocks, action_dim)
        self.apply(_orthogonal_init_)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, std: float = 0.2) -> TruncatedNormal:
        obs, squeezed = _ensure_batch(obs)
        z, _ = _ensure_batch(z)
        z_emb = self.embed_z(torch.cat([obs, z], dim=-1))
        s_emb = self.embed_s(obs)
        mu = torch.tanh(self.policy(torch.cat([s_emb, z_emb], dim=-1)))
        if squeezed:
            mu = mu.squeeze(0)
        std_t = torch.ones_like(mu) * std
        return TruncatedNormal(mu, std_t)


class BackwardMap(nn.Module):
    """B(x) → z. Table 2: 1×256 ReLU MLP + BallNorm."""

    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 1,
        ball_norm: bool = True,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        layers: list[nn.Module] = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, z_dim))
        if ball_norm:
            layers.append(BallNorm())
        self.net = nn.Sequential(*layers)
        self.apply(_orthogonal_init_)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Discriminator(nn.Module):
    """D(x, z) → (0,1). Table 2: 2×1024 ReLU MLP. GAN reward via compute_reward."""

    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        hidden_dim: int = 1024,
        hidden_layers: int = 2,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        layers: list[nn.Module] = [nn.Linear(obs_dim + z_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.trunk = nn.Sequential(*layers)
        self.apply(_orthogonal_init_)

    def compute_logits(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        obs, squeezed = _ensure_batch(obs)
        z, _ = _ensure_batch(z)
        logits = self.trunk(torch.cat([obs, z], dim=-1))
        return logits.squeeze(0) if squeezed else logits

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.compute_logits(obs, z))

    def compute_reward(self, obs: torch.Tensor, z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        p = torch.clamp(self.forward(obs, z), eps, 1.0 - eps)
        return p.log() - (1.0 - p).log()
