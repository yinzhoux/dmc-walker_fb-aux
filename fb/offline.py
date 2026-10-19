"""Two-stage FB training: proto only, then mixed online rollout."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
import tqdm

from fb.agent import Agent, AlgoConfig
from fb.buffer import Buffer, resolve_offline_ratio, sample_mixed
from fb.walker import (
    GOAL_POSES,
    action_rate_reward,
    clip_action,
    done_flags,
    flat_obs,
    goal_state,
)


def build_agent(env, device: str, use_aux: bool) -> Agent:
    obs = flat_obs(env.reset())
    action_dim = int(np.prod(env.action_spec().shape))
    cfg = AlgoConfig()
    cfg.use_aux = use_aux
    cfg.compile_nets = device.startswith("cuda")
    cfg.fused_adam = device.startswith("cuda")
    return Agent(obs, np.zeros(action_dim, dtype=np.float32), cfg, device=device)


@torch.no_grad()
def _act(agent: Agent, obs: np.ndarray, z: torch.Tensor) -> np.ndarray:
    action = agent.actor(agent._as_obs(obs), z, agent.cfg.actor_std).sample()
    return clip_action(action)


def begin_episode(env, agent: Agent, goal_ratio: float):
    """Hold one z for the episode. Goal episodes use project(B(s_g))."""
    time_step = env.reset()
    obs = flat_obs(time_step)
    if random.random() < goal_ratio:
        name = random.choice(list(GOAL_POSES))
        gobs, _ = goal_state(env, name)
        z = agent.goal_inference(torch.from_numpy(gobs).to(agent.device).unsqueeze(0))
    else:
        name = "sphere"
        z = agent.sample_z()
    return obs, z, name, 0


def train_proto(
    agent: Agent,
    data,
    *,
    updates: int,
    batch_size: int,
    log_every: int,
    eval_every: int,
    log: Callable[[dict, int], None],
    evaluate: Callable[[int], None],
) -> None:
    pbar = tqdm.tqdm(range(1, updates + 1), desc="fb proto")
    for step in pbar:
        info = agent.update(data.sample(batch_size))
        if step % log_every == 0:
            pbar.set_postfix(fb=f"{info.get('train/fb_loss', 0):.3f}")
            log(info, step)
        if step % eval_every == 0 or step == updates:
            evaluate(step)


def train_online(
    agent: Agent,
    env,
    data,
    *,
    frames: int,
    buffer_size: int,
    batch_size: int,
    goal_ratio: float,
    ratio_start: float,
    ratio_end: float,
    proto_updates: int,
    log_every: int,
    eval_every: int,
    use_aux: bool,
    log: Callable[[dict, int], None],
    evaluate: Callable[[int], None],
) -> None:
    online = Buffer(
        flat_obs(env.reset()),
        np.zeros(agent.action_dim, dtype=np.float32),
        size=buffer_size,
        buffer_device="cpu",
        train_device=agent.device,
        store_prev_action=use_aux,
    )
    obs, z, z_name, ep_step = begin_episode(env, agent, goal_ratio)
    prev_action = np.zeros(agent.action_dim, dtype=np.float32)
    print(
        f"online stage: {frames} frames, "
        f"proto ratio {ratio_start} -> {ratio_end}, "
        f"goal episodes {goal_ratio}"
    )
    pbar = tqdm.tqdm(range(frames), desc="fb online")
    for t in pbar:
        if z_name == "sphere" and ep_step > 0 and ep_step % agent.cfg.z_horizon == 0:
            z = agent.sample_z()
        action = _act(agent, obs, z)
        time_step = env.step(action)
        nxt = flat_obs(time_step)
        terminated, truncated = done_flags(time_step)
        if use_aux:
            reward = float(
                np.asarray(action_rate_reward(action, prev_action, agent.cfg.action_rate_scale)).reshape(-1)[0]
            )
        else:
            reward = 0.0
        online.append(
            obs,
            action,
            nxt,
            reward,
            terminated,
            truncated,
            prev_action=prev_action if use_aux else None,
        )
        ep_step += 1
        if time_step.last():
            obs, z, z_name, ep_step = begin_episode(env, agent, goal_ratio)
            prev_action = np.zeros(agent.action_dim, dtype=np.float32)
        else:
            obs = nxt
            prev_action = action

        ratio = resolve_offline_ratio(
            "linear_decay",
            offline_size=data.size,
            online_size=online.size,
            t=t,
            T=frames,
            ratio_start=ratio_start,
            ratio_end=ratio_end,
        )
        batch = sample_mixed(data, online, batch_size, ratio)
        info = agent.update(batch)
        step = proto_updates + t + 1
        if step % log_every == 0:
            info["train/offline_ratio"] = ratio
            info["train/online_size"] = online.size
            pbar.set_postfix(fb=f"{info.get('train/fb_loss', 0):.3f}", mix=f"{ratio:.2f}")
            log(info, step)
        if step % eval_every == 0 or t + 1 == frames:
            evaluate(step)
            obs, z, z_name, ep_step = begin_episode(env, agent, goal_ratio)
