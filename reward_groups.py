"""Named reward groups attached to a Gymnasium step.

The env reward itself is unchanged. Each step also writes
``info["reward_groups"]``, a dict of floats the replay can store. FB still
trains on transitions; these groups are the signals used to infer a z per task.

Add a task by registering a function ``(obs, action, reward, info) -> dict``.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

GroupFn = Callable[[np.ndarray, np.ndarray, float, dict], dict[str, float]]


def walker2d_reward_groups(obs, action, reward, info) -> dict[str, float]:
    """Same kinematics as algos/fb_eval.py, plus the native Walker2d terms."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    height = float(obs[0])
    angle = float(obs[1])
    xvel = float(obs[8])
    groups = {
        "gym": float(reward),
        "forward": xvel,
        "backward": -xvel,
        "high": height,
        "low": -height,
        "stand": height - abs(xvel) - abs(angle),
    }
    for key in ("reward_forward", "reward_ctrl", "reward_survive"):
        if key in info:
            groups[key] = float(info[key])
    return groups


def groups_for(task_name: str) -> GroupFn | None:
    if "Walker2d" in task_name:
        return walker2d_reward_groups
    return None


class RewardGroupWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, group_fn: GroupFn):
        super().__init__(env)
        self._group_fn = group_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["reward_groups"] = self._group_fn(
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            info,
        )
        return obs, reward, terminated, truncated, info
