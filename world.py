from dataclasses import dataclass

import numpy as np
import torch

from buffer import Buffer
from reward_groups import RewardGroupWrapper, groups_for


@dataclass
class WorldCfg:
    suite: str = "gym"
    task_name: str = "Walker2d-v5"
    dataset_dir: str = "./.ogbench"
    max_episode_steps: int = 1000
    # None: env default. False: train env never emits terminated (timeout only).
    terminate_when_unhealthy: bool | None = None
    buffer_size: int = 1000000
    train_device: str = "cuda"
    buffer_device: str = "cpu"
    state_normalization: bool = False
    obs_norm_eps: float = 1e-6


class World:
    """Owns env + buffer. Agents only see `data_out` views (e.g. obs norm)."""

    def __init__(self, cfg: WorldCfg) -> None:
        self.cfg = cfg
        self._obs_raw = None
        self._obs_mean_np = None
        self._obs_std_np = None
        self._obs_mean = None
        self._obs_std = None
        self._skill_env = None
        if cfg.suite != "gym":
            raise ValueError(f"Unknown suite {cfg.suite}. This repo only supports 'gym'.")
        self._init_gym()

    def _init_gym(self):
        import gymnasium as gym

        make_kw = {"id": self.cfg.task_name, "max_episode_steps": self.cfg.max_episode_steps}
        env_kw = dict(make_kw)
        if self.cfg.terminate_when_unhealthy is not None:
            env_kw["terminate_when_unhealthy"] = self.cfg.terminate_when_unhealthy
        env = gym.make(**env_kw)
        eval_env = gym.make(**make_kw)
        env = _maybe_flatten_obs_env(env)
        eval_env = _maybe_flatten_obs_env(eval_env)
        group_fn = groups_for(self.cfg.task_name)
        if group_fn is not None:
            env = RewardGroupWrapper(env, group_fn)

        self.training_env = env
        self.eval_env = eval_env
        self.action_space = env.action_space
        self.eval_metric = "return"
        self._obs_ex = np.asarray(env.observation_space.sample(), dtype=np.float32)
        self._act_ex = np.asarray(env.action_space.sample(), dtype=np.float32)
        self.buffer = Buffer(
            self._obs_ex,
            self._act_ex,
            size=self.cfg.buffer_size,
            buffer_device=self.cfg.buffer_device,
            train_device=self.cfg.train_device,
        )

    @property
    def observation_example(self):
        return self.data_out(self._obs_ex)

    @property
    def action_example(self):
        return self._act_ex

    def train_reward(self, reward):
        return float(np.asarray(reward, dtype=np.float32))

    def data_out(self, data):
        if not self.cfg.state_normalization:
            return data
        if isinstance(data, dict) and "observations" in data:
            out = dict(data)
            out["observations"] = self._norm_obs(data["observations"])
            if "next_observations" in data:
                out["next_observations"] = self._norm_obs(data["next_observations"])
            return out
        return self._norm_obs(data)

    def _norm_obs(self, obs):
        if self._obs_mean_np is None:
            return obs
        if torch.is_tensor(obs):
            mean = self._obs_mean.to(device=obs.device, dtype=obs.dtype)
            std = self._obs_std.to(device=obs.device, dtype=obs.dtype)
            return (obs - mean) / std
        obs = np.asarray(obs, dtype=np.float32)
        return (obs - self._obs_mean_np) / self._obs_std_np

    def sample(self, batch_size: int):
        return self.data_out(self.buffer.sample(batch_size))

    def reset(self):
        observation, info = _env_reset(self.training_env)
        self._obs_raw = np.asarray(observation, dtype=np.float32)
        return self.data_out(self._obs_raw), info

    def step(self, action, *, store: bool = True):
        next_observation, reward, terminated, truncated, info = _env_step(
            self.training_env, action
        )
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if store:
            self.buffer.append(
                self._obs_raw,
                action,
                next_observation,
                self.train_reward(reward),
                terminated,
                truncated,
                reward_groups=info.get("reward_groups"),
            )
        done = terminated or truncated
        self._obs_raw = None if done else next_observation
        return self.data_out(next_observation), reward, terminated, truncated, info

    def eval(self, eval_policy, episode_cnt: int = 10, metric: str | None = None):
        metric = self.eval_metric if metric is None else metric
        if metric != "return":
            raise ValueError(f"Unknown eval metric {metric}. Use 'return'.")

        observation, info = _env_reset(self.eval_env)
        cnt = 0
        frame_cnt = 0
        all_reward = 0.0
        while True:
            frame_cnt += 1
            action = eval_policy(self.data_out(observation)).detach().cpu().numpy()
            next_observation, reward, terminated, truncated, info = _env_step(
                self.eval_env, action
            )
            all_reward += reward
            if terminated or truncated:
                cnt += 1
                if cnt == episode_cnt:
                    break
                observation, info = _env_reset(self.eval_env)
            else:
                observation = next_observation

        mean_return = all_reward / episode_cnt
        return {
            "eval/mean_episode_len": frame_cnt / episode_cnt,
            "eval/mean_episode_return": mean_return,
            "eval/score": mean_return,
        }

    def skill_eval_env(self):
        if self._skill_env is not None:
            return self._skill_env
        import gymnasium as gym

        kw = {"id": self.cfg.task_name, "max_episode_steps": self.cfg.max_episode_steps}
        try:
            env = gym.make(**kw, terminate_when_unhealthy=False)
        except TypeError:
            env = gym.make(**kw)
        self._skill_env = _maybe_flatten_obs_env(env)
        return self._skill_env

    def eval_stats(self, eval_policy, episode_cnt: int = 5, env=None) -> dict:
        env = self.eval_env if env is None else env
        returns, lengths, heights, xvels, angles = [], [], [], [], []
        for _ in range(episode_cnt):
            observation, info = _env_reset(env)
            ret = 0.0
            n = 0
            h_sum = xv_sum = ang_sum = 0.0
            while True:
                o = np.asarray(observation, dtype=np.float32).reshape(-1)
                if o.size > 0:
                    h_sum += float(o[0])
                if o.size > 1:
                    ang_sum += abs(float(o[1]))
                if o.size > 8:
                    xv_sum += float(o[8])
                action = eval_policy(self.data_out(observation)).detach().cpu().numpy()
                observation, reward, terminated, truncated, info = _env_step(env, action)
                ret += float(reward)
                n += 1
                if terminated or truncated:
                    break
            returns.append(ret)
            lengths.append(n)
            heights.append(h_sum / max(n, 1))
            xvels.append(xv_sum / max(n, 1))
            angles.append(ang_sum / max(n, 1))
        return {
            "return": float(np.mean(returns)),
            "len": float(np.mean(lengths)),
            "height": float(np.mean(heights)),
            "xvel": float(np.mean(xvels)),
            "abs_angle": float(np.mean(angles)),
        }


def _env_reset(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        return out
    observation, reward, done, info = out
    return observation, reward, done, False, info


def _flatten_obs_seq(obs):
    if isinstance(obs, dict):
        if "observation" in obs:
            return _flatten_obs_seq(obs["observation"])
        cols = [_flatten_obs_seq(obs[k]) for k in sorted(obs.keys())]
        return np.concatenate(cols, axis=-1)
    return np.asarray(obs, dtype=np.float32)


def _maybe_flatten_obs_env(env):
    import gymnasium
    from gymnasium.spaces import Box, Dict

    if not isinstance(env.observation_space, Dict):
        return env

    class FlattenDictObs(gymnasium.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            dummy = self.observation(env.observation_space.sample())
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=dummy.shape, dtype=np.float32
            )

        def observation(self, obs):
            return _flatten_obs_seq(obs).ravel()

    return FlattenDictObs(env)
