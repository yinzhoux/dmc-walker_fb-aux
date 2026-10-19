"""ExORL proto npz → static (s, a, s') table."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import tqdm

from fb.walker import action_rate_reward


class TransitionDataset:
    """Static (s, a, s') table. The stored task reward is ignored unless aux is on."""

    def __init__(
        self,
        observations,
        actions,
        next_observations,
        terminated,
        train_device: str,
        prev_actions=None,
        rewards=None,
    ):
        self.train_device = train_device
        self._n = int(observations.shape[0])
        self._obs = torch.from_numpy(np.ascontiguousarray(observations))
        self._actions = torch.from_numpy(np.ascontiguousarray(actions))
        self._next = torch.from_numpy(np.ascontiguousarray(next_observations))
        self._terminated = torch.from_numpy(np.ascontiguousarray(terminated))
        self._truncated = torch.zeros((self._n, 1), dtype=torch.float32)
        self._prev = None if prev_actions is None else torch.from_numpy(np.ascontiguousarray(prev_actions))
        if rewards is None:
            self._rewards = torch.zeros((self._n, 1), dtype=torch.float32)
        else:
            self._rewards = torch.from_numpy(np.ascontiguousarray(rewards))

    def __len__(self) -> int:
        return self._n

    @property
    def size(self) -> int:
        return self._n

    def sample(self, batch_size: int) -> dict:
        idx = torch.randint(0, self._n, (batch_size,))

        def take(t):
            return t[idx].to(self.train_device)

        terminated = take(self._terminated)
        truncated = take(self._truncated)
        out = {
            "observations": take(self._obs),
            "actions": take(self._actions),
            "next_observations": take(self._next),
            "rewards": take(self._rewards),
            "terminated": terminated,
            "truncated": truncated,
            "done": (terminated.bool() | truncated.bool()).float(),
        }
        if self._prev is not None:
            out["prev_actions"] = take(self._prev)
        return out


def default_replay_dir() -> Path:
    env = os.environ.get("EXORL_DATASET_DIR")
    if env:
        return Path(env).expanduser()
    candidates = [
        Path.home() / "zhoujianxun/datasets/exorl",
        Path("/Users/yinzhoux/Files/projects/exorl/datasets"),
    ]
    for path in candidates:
        if (path / "walker" / "proto" / "buffer").is_dir():
            return path
    return candidates[0]


def _read_episode(path: Path):
    """ExORL row 0 is a dummy. Transition i uses obs[i-1], action[i], obs[i]."""
    with np.load(path) as data:
        obs = np.asarray(data["observation"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        discount = np.asarray(data["discount"], dtype=np.float32)
    if obs.shape[0] < 2:
        empty = np.zeros((0, obs.shape[-1]), np.float32)
        act = np.zeros((0, action.shape[-1]), np.float32)
        return empty, act, empty, np.zeros((0, 1), np.float32), act
    terminated = (discount[1:] == 0.0).astype(np.float32)
    if terminated.ndim == 1:
        terminated = terminated[:, None]
    return obs[:-1], action[1:], obs[1:], terminated, action[:-1]


def load_proto(
    replay_dir: Path,
    task: str,
    expl_agent: str,
    max_transitions: int,
    train_device: str,
    action_rate_scale: float = 0.0,
) -> TransitionDataset:
    domain = task.split("_", 1)[0]
    folder = replay_dir / domain / expl_agent / "buffer"
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no npz under {folder}. Set EXORL_DATASET_DIR or --replay-dir to the "
            "ExORL datasets root (it should contain walker/proto/buffer/*.npz)."
        )

    obs_chunks, act_chunks, next_chunks, term_chunks, prev_chunks = [], [], [], [], []
    total = 0
    for path in tqdm.tqdm(files, desc="load proto"):
        obs, action, nxt, terminated, prev = _read_episode(path)
        if obs.shape[0] == 0:
            continue
        if max_transitions and total + obs.shape[0] > max_transitions:
            keep = max_transitions - total
            obs, action, nxt, terminated, prev = (
                obs[:keep],
                action[:keep],
                nxt[:keep],
                terminated[:keep],
                prev[:keep],
            )
        obs_chunks.append(obs)
        act_chunks.append(action)
        next_chunks.append(nxt)
        term_chunks.append(terminated)
        prev_chunks.append(prev)
        total += obs.shape[0]
        if max_transitions and total >= max_transitions:
            break

    observations = np.concatenate(obs_chunks, axis=0)
    actions = np.concatenate(act_chunks, axis=0)
    next_observations = np.concatenate(next_chunks, axis=0)
    terminated = np.concatenate(term_chunks, axis=0)
    prev_actions = np.concatenate(prev_chunks, axis=0)
    rewards = None
    if action_rate_scale != 0.0:
        rewards = action_rate_reward(actions, prev_actions, action_rate_scale)
        print(
            f"offline proto: {total} transitions from {folder} "
            f"(terminated {float(terminated.mean()):.4f}, "
            f"action-rate reward mean {float(rewards.mean()):.4f})"
        )
    else:
        prev_actions = None
        print(
            f"offline proto: {total} transitions from {folder} "
            f"(terminated {float(terminated.mean()):.4f}, reward column dropped)"
        )
    return TransitionDataset(
        observations,
        actions,
        next_observations,
        terminated,
        train_device,
        prev_actions=prev_actions,
        rewards=rewards,
    )
