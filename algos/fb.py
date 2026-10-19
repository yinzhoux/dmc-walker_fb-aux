"""
Forward-Backward (Touati & Ollivier). Lite: F, B, π only.

Q = F(s, a, z)^T z.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass

from algos.modules.fb_nets import BackwardMap, Discriminator, FBActor, FBForward
from train_utils import TimeStats, as_log_dict, scalar

class FBModel(nn.Module):
    def  __init__(
        self, 
        observation_example,
        action_example,
        device,
        cfg

    ):
        super().__init__()
        
        self.obs_example = observation_example
        self.act_example = action_example
        self.device = device


        self._obs_normalizer = None



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
    # FB-only: fraction of the batch replaced by B(s'_perm).
    # CPR: official train_goal_ratio, the multinomial mass on that goal z.
    z_mix_ratio: float = 0.5
    # Official FBcprAgent. 0 keeps the Walker FB loop unchanged.
    use_cpr: bool = False
    expert_asm_ratio: float = 0.0
    relabel_ratio: float | None = None
    seq_length: int = 8
    reg_coeff: float = 0.0
    scale_reg: bool = True
    grad_penalty: float = 10.0
    disc_lr: float = 1e-5
    disc_hidden_dim: int = 1024
    disc_hidden_layers: int = 2
    critic_lr: float = 3e-4
    critic_tau: float = 0.005
    critic_pessimism: float = 0.5
    # Aux Q on the action-rate penalty. Off keeps the Walker FB loop unchanged.
    use_aux: bool = False
    reg_coeff_aux: float = 0.02
    # Official aux_rewards_scaling['penalty_action_rate'].
    action_rate_scale: float = -0.1
    
    actor_std: float = 0.2
    # FBAgentTrainConfig: stddev_clip / fb_pessimism_penalty / actor_pessimism_penalty
    stddev_clip: float = 0.3
    fb_pessimism: float = 0.0
    actor_pessimism: float = 0.5
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0  # 0 disables, same as official

    hidden_dim: int = 256
    embedding_layers: int = 2
    residual_blocks: int = 2
    num_parallel: int = 2
    b_hidden_dim: int = 256
    b_hidden_layers: int = 1

    fused_adam: bool = True
    zero_grad_set_to_none: bool = True
    compile_nets: bool = True

    eval_inference_samples: int = 4096
    reward_inference_temp: float = 10.0  # official reward_wr; 0 = uniform Σ r B


def _fill_cfg_defaults(cfg: AlgoConfig) -> None:
    """Old checkpoints were pickled before the CPR fields existed."""
    fresh = AlgoConfig()
    for name in (
        "use_cpr",
        "expert_asm_ratio",
        "relabel_ratio",
        "seq_length",
        "reg_coeff",
        "scale_reg",
        "grad_penalty",
        "disc_lr",
        "disc_hidden_dim",
        "disc_hidden_layers",
        "critic_lr",
        "critic_tau",
        "critic_pessimism",
        "use_aux",
        "reg_coeff_aux",
        "action_rate_scale",
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
        if cfg.use_cpr and cfg.use_aux:
            raise ValueError("action-rate Q is the FB path; leave use_cpr off")
        self.cfg = cfg
        self.device = device

        self.obs_dim = int(np.asarray(observation_example).shape[-1])
        self.action_dim = int(np.asarray(action_example).shape[-1])
        obs_dim = self.obs_dim
        action_dim = self.action_dim
        arch = dict(
            hidden_dim=cfg.hidden_dim,
            embedding_layers=cfg.embedding_layers,
            residual_blocks=cfg.residual_blocks,
        )

        self.F = FBForward(
            obs_dim, action_dim, cfg.z_dim, num_parallel=cfg.num_parallel, **arch
        ).to(device)
        self.B = BackwardMap(
            obs_dim, cfg.z_dim, hidden_dim=cfg.b_hidden_dim, hidden_layers=cfg.b_hidden_layers
        ).to(device)
        self.actor = FBActor(obs_dim, action_dim, cfg.z_dim, **arch).to(device)

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
        self.discriminator = None
        self.Q = None
        self.Q_target = None
        if cfg.use_cpr:
            self.discriminator = Discriminator(
                obs_dim,
                cfg.z_dim,
                hidden_dim=cfg.disc_hidden_dim,
                hidden_layers=cfg.disc_hidden_layers,
            ).to(device)
            self.Q = FBForward(
                obs_dim,
                action_dim,
                cfg.z_dim,
                num_parallel=cfg.num_parallel,
                out_dim=1,
                **arch,
            ).to(device)
            self.Q_target = copy.deepcopy(self.Q).to(device)
            self.Q_target.requires_grad_(False)
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=cfg.disc_lr,
                weight_decay=cfg.weight_decay,
                fused=cfg.fused_adam,
            )
            self.critic_optimizer = torch.optim.Adam(
                self.Q.parameters(),
                lr=cfg.critic_lr,
                weight_decay=cfg.weight_decay,
                fused=cfg.fused_adam,
            )

        self.Q_aux = None
        self.Q_aux_target = None
        self.aux_reward_normalizer = None
        if cfg.use_aux:
            # Previous action is not in the walker observation, so Q sees it directly.
            self.Q_aux = FBForward(
                obs_dim + action_dim,
                action_dim,
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
            if self.discriminator is not None:
                self.discriminator = torch.compile(self.discriminator)
                self.Q = torch.compile(self.Q)
                self.Q_target = torch.compile(self.Q_target)
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

    def update(self, batch, expert_obs=None, expert_next_obs=None):
        if not self.cfg.use_cpr:
            return self._update_fb(batch)
        if expert_obs is None:
            raise ValueError("CPR update needs an expert observation batch")
        if "z" not in batch:
            raise ValueError("CPR update needs the z stored with each online transition")
        if expert_next_obs is None:
            expert_next_obs = expert_obs
        return self._update_cpr(batch, expert_obs, expert_next_obs)

    def _update_fb(self, batch):
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

    @torch.no_grad()
    def encode_expert(self, obs: torch.Tensor) -> torch.Tensor:
        """Official encode_expert: mean_t B(w_t), project, repeat along the sequence."""
        seq = self.cfg.seq_length
        n = obs.shape[0]
        if seq < 1 or n % seq != 0:
            raise ValueError(f"expert batch {n} must be a multiple of seq_length {seq}")
        b = self.B(obs)
        z = b.view(n // seq, seq, -1).mean(dim=1)
        z = self.project_z(z)
        return z.repeat_interleave(seq, dim=0)

    @torch.no_grad()
    def sample_mixed_z(self, next_obs: torch.Tensor, expert_z: torch.Tensor) -> torch.Tensor:
        """Official sample_mixed_z: goal B(s'_perm), expert encoding, else sphere."""
        n = next_obs.shape[0]
        p_goal = self.cfg.z_mix_ratio
        p_expert = self.cfg.expert_asm_ratio
        p_uniform = 1.0 - p_goal - p_expert
        if p_uniform < -1e-6:
            raise ValueError("z_mix_ratio + expert_asm_ratio must be <= 1")
        z = self.sample_z(n)
        prob = torch.tensor(
            [p_goal, p_expert, max(p_uniform, 0.0)],
            dtype=torch.float32,
            device=self.device,
        )
        mix = torch.multinomial(prob, num_samples=n, replacement=True).view(-1, 1)
        perm = torch.randperm(n, device=self.device)
        goals = self.project_z(self.B(next_obs[perm]))
        z = torch.where(mix == 0, goals, z)
        perm = torch.randperm(n, device=self.device)
        return torch.where(mix == 1, expert_z[perm], z)

    def _gradient_penalty(self, real_obs, real_z, fake_obs, fake_z) -> torch.Tensor:
        alpha = torch.rand(real_obs.shape[0], 1, device=real_obs.device)
        inter_obs = (alpha * real_obs + (1.0 - alpha) * fake_obs).requires_grad_(True)
        inter_z = (alpha * real_z + (1.0 - alpha) * fake_z).requires_grad_(True)
        logits = self.discriminator.compute_logits(inter_obs, inter_z)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=(inter_obs, inter_z),
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
        )
        grad = torch.cat(grads, dim=-1)
        return ((grad.norm(dim=-1) - 1.0) ** 2).mean()

    def _update_discriminator(self, expert_obs, expert_z, train_obs, train_z) -> dict:
        expert_logits = self.discriminator.compute_logits(expert_obs, expert_z)
        train_logits = self.discriminator.compute_logits(train_obs, train_z)
        expert_loss = -F.logsigmoid(expert_logits)
        train_loss = F.softplus(train_logits)
        loss = (expert_loss + train_loss).mean()
        gp = None
        if self.cfg.grad_penalty > 0:
            gp = self._gradient_penalty(expert_obs, expert_z, train_obs, train_z)
            loss = loss + self.cfg.grad_penalty * gp
        self.disc_optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
        loss.backward()
        self._maybe_clip_grad(self.discriminator.parameters())
        self.disc_optimizer.step()
        info = {
            "train/disc_loss": scalar(loss),
            "train/disc_expert_loss": scalar(expert_loss.mean()),
            "train/disc_train_loss": scalar(train_loss.mean()),
        }
        if gp is not None:
            info["train/disc_gp"] = scalar(gp)
        return info

    def _update_q(self, obs, action, next_obs, z, terminated) -> dict:
        discount = self.cfg.gamma * (1.0 - terminated.float())
        with torch.no_grad():
            reward = self.discriminator.compute_reward(obs, z)
            next_action = self.actor(next_obs, z, self.cfg.actor_std).sample(
                clip=self.cfg.stddev_clip
            )
            next_q = self._reduce_ensemble(
                self._as_ensemble(self.Q_target(next_obs, z, next_action)),
                self.cfg.critic_pessimism,
            )
            target = reward + discount * next_q
        qs = self._as_ensemble(self.Q(obs, z, action))
        expanded = target.unsqueeze(0).expand(qs.shape[0], -1, -1)
        loss = 0.5 * qs.shape[0] * F.mse_loss(qs, expanded)
        self.critic_optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
        loss.backward()
        self._maybe_clip_grad(self.Q.parameters())
        self.critic_optimizer.step()
        return {
            "train/q_loss": scalar(loss),
            "train/q_d": scalar(qs.mean()),
            "train/disc_reward": scalar(reward.mean()),
        }

    def _update_actor_cpr(self, obs, z) -> dict:
        action = self.actor(obs, z, self.cfg.actor_std).sample(clip=self.cfg.stddev_clip)
        q_d = self._reduce_ensemble(
            self._as_ensemble(self.Q(obs, z, action)),
            self.cfg.actor_pessimism,
        )
        fs = self._as_ensemble(self.F(obs, z, action))
        q_fb = self._reduce_ensemble((fs * z).sum(dim=-1), self.cfg.actor_pessimism)
        weight = q_fb.abs().mean().detach() if self.cfg.scale_reg else 1.0
        loss = -q_d.mean() * self.cfg.reg_coeff * weight - q_fb.mean()
        self.actor_optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
        loss.backward()
        self._maybe_clip_grad(self.actor.parameters())
        self.actor_optimizer.step()
        return {
            "train/actor_loss": scalar(loss),
            "train/q": scalar(q_fb.mean()),
            "train/q_d_actor": scalar(q_d.mean()),
            "train/reg_weight": float(weight) if not torch.is_tensor(weight) else scalar(weight),
            "train/policy_abs_action": scalar(action.abs().mean()),
        }

    def _update_cpr(self, batch, expert_obs: torch.Tensor, expert_next_obs: torch.Tensor):
        expert_obs = torch.as_tensor(expert_obs, device=self.device, dtype=torch.float32)
        expert_next_obs = torch.as_tensor(expert_next_obs, device=self.device, dtype=torch.float32)
        obs = batch["observations"]
        action = batch["actions"]
        next_obs = batch["next_observations"]
        stored_z = batch["z"].detach()
        infos = {}

        expert_z = self.encode_expert(expert_next_obs)
        with self.time_stats.span("disc"):
            infos.update(self._update_discriminator(expert_obs, expert_z, obs, stored_z))

        z = self.sample_mixed_z(next_obs, expert_z)
        if self.cfg.relabel_ratio is not None:
            mask = torch.rand(stored_z.shape[0], 1, device=self.device) <= self.cfg.relabel_ratio
            z = torch.where(mask, z, stored_z)

        with self.time_stats.span("fb"):
            fb_loss, fb_info = self.fb_loss(batch, z)
            self.fb_optimizer.zero_grad(set_to_none=self.cfg.zero_grad_set_to_none)
            fb_loss.backward()
            self._maybe_clip_grad(list(self.F.parameters()) + list(self.B.parameters()))
            self.fb_optimizer.step()
        infos.update(fb_info)

        with self.time_stats.span("critic"):
            infos.update(self._update_q(obs, action, next_obs, z, batch["terminated"]))
        with self.time_stats.span("actor"):
            infos.update(self._update_actor_cpr(obs, z))

        with self.time_stats.span("target"):
            self._soft_update(self.F, self.F_target)
            self._soft_update(self.B, self.B_target)
            self._soft_update(self.Q, self.Q_target, tau=self.cfg.critic_tau)
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

    @torch.no_grad()
    def prepare_eval(self, world, batch_size: int = 1024) -> None:
        n = min(batch_size, world.buffer.size)
        batch = world.sample(n)
        B_s = self.B(batch["observations"])
        self._z_r = self.project_z((batch["rewards"] * B_s).sum(dim=0, keepdim=True))
        feat = (
            batch["observations"][:, 8]
            if batch["observations"].shape[-1] > 8
            else batch["rewards"].squeeze(-1)
        )
        idx = int(feat.argmax())
        self._z_g = self.project_z(B_s[idx : idx + 1])
        self._z = self._z_r

    def eval_policies(self, world):
        self.prepare_eval(world)
        return {
            "zr": self.get_rollout_policy(explore=False, z=self._z_r),
            "zg": self.get_rollout_policy(explore=False, z=self._z_g),
        }

    def eval_skills(self, world, episode_cnt: int = 5) -> dict:
        from algos.fb_eval import run_skill_eval

        return run_skill_eval(self, world, episode_cnt=episode_cnt)

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
        if self.discriminator is not None:
            payload["discriminator"] = self._raw(self.discriminator).state_dict()
            payload["Q"] = self._raw(self.Q).state_dict()
            payload["Q_target"] = self._raw(self.Q_target).state_dict()
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
        if agent.discriminator is not None:
            agent.discriminator.load_state_dict(ckpt["discriminator"])
            agent.Q.load_state_dict(ckpt["Q"])
            agent.Q_target.load_state_dict(ckpt.get("Q_target", ckpt["Q"]))
        if agent.Q_aux is not None and "Q_aux" in ckpt:
            agent.Q_aux.load_state_dict(ckpt["Q_aux"])
            agent.Q_aux_target.load_state_dict(ckpt.get("Q_aux_target", ckpt["Q_aux"]))
            if "aux_reward_normalizer" in ckpt:
                agent.aux_reward_normalizer.load_state_dict(ckpt["aux_reward_normalizer"])
        agent._z_r = ckpt["z_r"].to(device)
        agent._z_g = ckpt["z_g"].to(device)
        agent._z = agent._z_r
        return agent
