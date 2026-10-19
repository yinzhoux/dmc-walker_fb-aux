"""Forward-Backward (Touati & Ollivier). Lite: F, B, π, optional action-rate Q.

Q_fb = F(s, a, z)^T z.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fb.log import TimeStats, as_log_dict, scalar
from fb.nets import BackwardMap, FBActor, FBForward


@dataclass
class AlgoConfig:
    name: str = "fb"

    gamma: float = 0.98
    tau: float = 0.005
    fb_lr: float = 3e-4
    actor_lr: float = 3e-4
    actor_delayed: int = 2
    ortho_coef: float = 1.0

    z_dim: int = 50
    norm_z: bool = True
    z_horizon: int = 150
    z_mix_ratio: float = 0.5

    use_aux: bool = False
    reg_coeff_aux: float = 0.02
    action_rate_scale: float = -0.1
    scale_reg: bool = True

    actor_std: float = 0.2
    stddev_clip: float = 0.3
    fb_pessimism: float = 0.0
    actor_pessimism: float = 0.5
    critic_lr: float = 3e-4
    critic_tau: float = 0.005
    critic_pessimism: float = 0.5
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0

    hidden_dim: int = 256
    embedding_layers: int = 2
    residual_blocks: int = 2
    num_parallel: int = 2
    b_hidden_dim: int = 256
    b_hidden_layers: int = 1

    fused_adam: bool = True
    zero_grad_set_to_none: bool = True
    compile_nets: bool = True

    reward_inference_temp: float = 10.0


def _fill_cfg_defaults(cfg: AlgoConfig) -> None:
    """Old checkpoints were pickled before some fields existed."""
    fresh = AlgoConfig()
    for name in (
        "use_aux",
        "reg_coeff_aux",
        "action_rate_scale",
        "scale_reg",
        "critic_lr",
        "critic_tau",
        "critic_pessimism",
    ):
        if not hasattr(cfg, name):
            setattr(cfg, name, getattr(fresh, name))


class _RewardScaler(nn.Module):
    """Official EMA reward normalizer: divide by running std, keep the sign."""

    def __init__(self, tau: float = 0.99, epsilon: float = 1e-8):
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros((), dtype=torch.float32))
        self.register_buffer("mean_square", torch.zeros((), dtype=torch.float32))
        self.register_buffer("counter", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.detach().mean()
        sm = x.detach().pow(2).mean()
        self.mean.mul_(self.tau).add_(m, alpha=1.0 - self.tau)
        self.mean_square.mul_(self.tau).add_(sm, alpha=1.0 - self.tau)
        self.counter += 1
        norm = 1.0 - self.tau**self.counter
        ema_mean = self.mean / norm
        ema_sq = self.mean_square / norm
        var = torch.clamp(ema_sq - ema_mean**2, min=self.epsilon)
        return x / torch.sqrt(var)


class Agent:
    def __init__(
        self,
        observation_example,
        action_example,
        cfg: AlgoConfig,
        device: str = "cpu",
    ):
        _fill_cfg_defaults(cfg)
        self.cfg = cfg
        self.device = device

        self.obs_dim = int(np.asarray(observation_example).shape[-1])
        self.action_dim = int(np.asarray(action_example).shape[-1])
        arch = dict(
            hidden_dim=cfg.hidden_dim,
            embedding_layers=cfg.embedding_layers,
            residual_blocks=cfg.residual_blocks,
        )

        self.F = FBForward(
            self.obs_dim, self.action_dim, cfg.z_dim, num_parallel=cfg.num_parallel, **arch
        ).to(device)
        self.B = BackwardMap(
            self.obs_dim, cfg.z_dim, hidden_dim=cfg.b_hidden_dim, hidden_layers=cfg.b_hidden_layers
        ).to(device)
        self.actor = FBActor(self.obs_dim, self.action_dim, cfg.z_dim, **arch).to(device)
        self.F_target = copy.deepcopy(self.F).to(device)
        self.B_target = copy.deepcopy(self.B).to(device)
        self.F_target.requires_grad_(False)
        self.B_target.requires_grad_(False)

        self.fb_optimizer = torch.optim.Adam(
            list(self.F.parameters()) + list(self.B.parameters()),
            lr=cfg.fb_lr,
            weight_decay=cfg.weight_decay,
            fused=cfg.fused_adam,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            weight_decay=cfg.weight_decay,
            fused=cfg.fused_adam,
        )

        self.Q_aux = None
        self.Q_aux_target = None
        self.aux_reward_normalizer = None
        if cfg.use_aux:
            self.Q_aux = FBForward(
                self.obs_dim + self.action_dim,
                self.action_dim,
                cfg.z_dim,
                num_parallel=cfg.num_parallel,
                out_dim=1,
                **arch,
            ).to(device)
            self.Q_aux_target = copy.deepcopy(self.Q_aux).to(device)
            self.Q_aux_target.requires_grad_(False)
            self.aux_critic_optimizer = torch.optim.Adam(
                self.Q_aux.parameters(),
                lr=cfg.critic_lr,
                weight_decay=cfg.weight_decay,
                fused=cfg.fused_adam,
            )
            self.aux_reward_normalizer = _RewardScaler().to(device)

        if cfg.compile_nets:
            self.F = torch.compile(self.F)
            self.B = torch.compile(self.B)
            self.actor = torch.compile(self.actor)
            self.F_target = torch.compile(self.F_target)
            self.B_target = torch.compile(self.B_target)
            if self.Q_aux is not None:
                self.Q_aux = torch.compile(self.Q_aux)
                self.Q_aux_target = torch.compile(self.Q_aux_target)

        self._z = self.sample_z()
        self._z_r = self._z
        self._z_g = self._z
        self._z_steps = 0
        self._critic_update = 0
        self.time_stats = TimeStats()

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def sample_z(self, n: int = 1) -> torch.Tensor:
        z = torch.randn(n, self.cfg.z_dim, device=self.device)
        return self.project_z(z)

    @staticmethod
    def _as_ensemble(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.ndim == 2 else x

    @staticmethod
    def _reduce_ensemble(preds: torch.Tensor, pessimism: float) -> torch.Tensor:
        """Mean minus pairwise |F_i - F_j|. N=2 and pessimism=0.5 equals min."""
        if preds.shape[0] == 1:
            return preds.squeeze(0)
        mean = preds.mean(dim=0)
        if pessimism == 0.0:
            return mean
        n = preds.shape[0]
        diffs = (preds.unsqueeze(0) - preds.unsqueeze(1)).abs()
        unc = diffs.sum(dim=(0, 1)) / (n * n - n)
        return mean - pessimism * unc

    def _maybe_clip_grad(self, params) -> None:
        if self.cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.cfg.clip_grad_norm)

    def _soft_update(self, net, target, tau: float | None = None) -> None:
        tau = self.cfg.tau if tau is None else tau
        for param, param_target in zip(net.parameters(), target.parameters()):
            param_target.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    @torch.no_grad()
    def _sample_train_z(self, batch) -> torch.Tensor:
        n = batch["observations"].shape[0]
        z = self.sample_z(n)
        perm = torch.randperm(n, device=self.device)
        goals = self.project_z(self.B(batch["next_observations"][perm]))
        mask = torch.rand(n, 1, device=self.device) < self.cfg.z_mix_ratio
        return torch.where(mask, goals, z)

    def fb_loss(self, batch, z: torch.Tensor):
        obs = batch["observations"]
        action = batch["actions"]
        next_obs = batch["next_observations"]
        discount = self.cfg.gamma * (1.0 - batch["terminated"].float())
        n = obs.shape[0]
        eye = torch.eye(n, device=obs.device, dtype=obs.dtype)
        off = 1.0 - eye
        off_sum = off.sum()

        with torch.no_grad():
            next_action = self.actor(next_obs, z, self.cfg.actor_std).sample(
                clip=self.cfg.stddev_clip
            )
            target_F = self._as_ensemble(self.F_target(next_obs, z, next_action))
            target_B = self.B_target(next_obs)
            target_M = self._reduce_ensemble(
                torch.matmul(target_F, target_B.T), pessimism=self.cfg.fb_pessimism
            )

        Fs = self._as_ensemble(self.F(obs, z, action))
        B = self.B(next_obs)
        Ms = torch.matmul(Fs, B.T)
        diff = Ms - discount * target_M

        fb_offdiag = 0.5 * (diff * off).pow(2).sum() / off_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Fs.shape[0]
        fb_occ = fb_offdiag + fb_diag

        cov = torch.matmul(B, B.T)
        ortho_off = 0.5 * (cov * off).pow(2).sum() / off_sum
        ortho_diag = -cov.diag().mean()
        ortho = ortho_off + ortho_diag

        loss = fb_occ + self.cfg.ortho_coef * ortho
        info = {
            "train/fb_loss": scalar(loss),
            "train/fb_offdiag": scalar(fb_offdiag),
            "train/fb_diag": scalar(fb_diag),
            "train/ortho": scalar(ortho),
            "train/B_norm": scalar(B.norm(dim=-1).mean()),
            "train/z_norm": scalar(z.norm(dim=-1).mean()),
            "train/M": scalar(Ms.mean()),
        }
        return loss, info

    def _aux_obs(self, obs: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, prev_action], dim=-1)

    def actor_loss(self, batch, z: torch.Tensor):
        obs = batch["observations"]
        action = self.actor(obs, z, self.cfg.actor_std).sample(clip=self.cfg.stddev_clip)
        Fs = self._as_ensemble(self.F(obs, z, action))
        Qs = (Fs * z).sum(dim=-1)
        q = self._reduce_ensemble(Qs, pessimism=self.cfg.actor_pessimism)
        loss = -q.mean()
        info = {
            "train/actor_loss": scalar(loss),
            "train/q": scalar(q.mean()),
            "train/policy_abs_action": scalar(action.abs().mean()),
        }
        if self.Q_aux is not None:
            if "prev_actions" not in batch:
                raise ValueError("aux update needs prev_actions on each transition")
            q_aux = self._reduce_ensemble(
                self._as_ensemble(self.Q_aux(self._aux_obs(obs, batch["prev_actions"]), z, action)),
                self.cfg.actor_pessimism,
            )
            weight = q.abs().mean().detach() if self.cfg.scale_reg else 1.0
            loss = loss - q_aux.mean() * self.cfg.reg_coeff_aux * weight
            info["train/actor_loss"] = scalar(loss)
            info["train/q_aux_actor"] = scalar(q_aux.mean())
            info["train/reg_weight_aux"] = float(weight) if not torch.is_tensor(weight) else scalar(weight)
        return loss, info

    def _update_aux(self, batch, z: torch.Tensor) -> dict:
        obs = batch["observations"]
        action = batch["actions"]
        next_obs = batch["next_observations"]
        prev = batch["prev_actions"]
        reward = self.aux_reward_normalizer(batch["rewards"])
        discount = self.cfg.gamma * (1.0 - batch["terminated"].float())
        with torch.no_grad():
            next_action = self.actor(next_obs, z, self.cfg.actor_std).sample(
                clip=self.cfg.stddev_clip
            )
            next_q = self._reduce_ensemble(
                self._as_ensemble(self.Q_aux_target(self._aux_obs(next_obs, action), z, next_action)),
                self.cfg.critic_pessimism,
            )
            target = reward + discount * next_q
        qs = self._as_ensemble(self.Q_aux(self._aux_obs(obs, prev), z, action))
        expanded = target.unsqueeze(0).expand(qs.shape[0], -1, -1)
        loss = 0.5 * qs.shape[0] * F.mse_loss(qs, expanded)
        self.aux_critic_optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
        loss.backward()
        self._maybe_clip_grad(self.Q_aux.parameters())
        self.aux_critic_optimizer.step()
        return {
            "train/aux_loss": scalar(loss),
            "train/q_aux": scalar(qs.mean()),
            "train/aux_reward": scalar(reward.mean()),
        }

    def update(self, batch):
        infos = {}
        self._critic_update += 1
        z = self._sample_train_z(batch)

        with self.time_stats.span("fb"):
            fb_loss, fb_info = self.fb_loss(batch, z)
            self.fb_optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
            fb_loss.backward()
            self._maybe_clip_grad(list(self.F.parameters()) + list(self.B.parameters()))
            self.fb_optimizer.step()
        infos.update(fb_info)

        if self.Q_aux is not None:
            if "prev_actions" not in batch:
                raise ValueError("aux update needs prev_actions on each transition")
            with self.time_stats.span("aux"):
                infos.update(self._update_aux(batch, z))

        if self._critic_update % self.cfg.actor_delayed == 0:
            with self.time_stats.span("actor"):
                actor_loss, actor_info = self.actor_loss(batch, z)
                self.actor_optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
                actor_loss.backward()
                self._maybe_clip_grad(self.actor.parameters())
                self.actor_optimizer.step()
            infos.update(actor_info)

        with self.time_stats.span("target"):
            self._soft_update(self.F, self.F_target)
            self._soft_update(self.B, self.B_target)
            if self.Q_aux is not None:
                self._soft_update(self.Q_aux, self.Q_aux_target, self.cfg.critic_tau)

        return as_log_dict(infos)

    def _as_obs(self, observation) -> torch.Tensor:
        if isinstance(observation, np.ndarray):
            return torch.from_numpy(observation).to(self.device, dtype=torch.float32)
        return torch.as_tensor(observation, device=self.device, dtype=torch.float32)

    def _maybe_refresh_rollout_z(self) -> None:
        if self._z_steps > 0 and self._z_steps % self.cfg.z_horizon == 0:
            self._z = self.sample_z()
        self._z_steps += 1

    @torch.no_grad()
    def reward_inference(
        self,
        obs: torch.Tensor,
        reward: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wr = reward if weight is None else reward * weight
        if wr.ndim == 1:
            wr = wr.unsqueeze(-1)
        z = (wr.to(obs.device) * self.B(obs)).sum(dim=0, keepdim=True)
        return self.project_z(z)

    @torch.no_grad()
    def reward_wr_inference(self, obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        r = reward.reshape(-1).to(obs.device)
        temp = self.cfg.reward_inference_temp
        if temp <= 0:
            return self.reward_inference(obs, r.unsqueeze(-1))
        w = F.softmax(temp * r, dim=0).unsqueeze(-1)
        return self.reward_inference(obs, r.unsqueeze(-1), weight=w)

    @torch.no_grad()
    def goal_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.project_z(self.B(obs))

    def get_rollout_policy(self, explore=True, z: torch.Tensor | None = None):
        def func(observation):
            observation = self._as_obs(observation)
            with torch.no_grad():
                if explore:
                    self._maybe_refresh_rollout_z()
                    lat = self._z
                    return self.actor(observation, lat, self.cfg.actor_std).sample()
                lat = self._z if z is None else z
                return self.actor(observation, lat, std=0.0).mean

        return func

    @staticmethod
    def _raw(module):
        return getattr(module, "_orig_mod", module)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "F": self._raw(self.F).state_dict(),
            "B": self._raw(self.B).state_dict(),
            "actor": self._raw(self.actor).state_dict(),
            "z_r": self._z_r.detach().cpu(),
            "z_g": self._z_g.detach().cpu(),
        }
        if self.Q_aux is not None:
            payload["Q_aux"] = self._raw(self.Q_aux).state_dict()
            payload["Q_aux_target"] = self._raw(self.Q_aux_target).state_dict()
            payload["aux_reward_normalizer"] = self.aux_reward_normalizer.state_dict()
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "Agent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["cfg"]
        cfg.compile_nets = False
        cfg.fused_adam = False
        agent = cls(
            np.zeros(ckpt["obs_dim"], dtype=np.float32),
            np.zeros(ckpt["action_dim"], dtype=np.float32),
            cfg,
            device=device,
        )
        agent.F.load_state_dict(ckpt["F"])
        agent.B.load_state_dict(ckpt["B"])
        agent.actor.load_state_dict(ckpt["actor"])
        if agent.Q_aux is not None and "Q_aux" in ckpt:
            agent.Q_aux.load_state_dict(ckpt["Q_aux"])
            agent.Q_aux_target.load_state_dict(ckpt.get("Q_aux_target", ckpt["Q_aux"]))
            if "aux_reward_normalizer" in ckpt:
                agent.aux_reward_normalizer.load_state_dict(ckpt["aux_reward_normalizer"])
        agent._z_r = ckpt["z_r"].to(device)
        agent._z_g = ckpt["z_g"].to(device)
        agent._z = agent._z_r
        return agent
