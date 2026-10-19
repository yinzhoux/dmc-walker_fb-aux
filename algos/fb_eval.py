"""Walker skill probes in the BFM-Zero style: relabel buffer → z → greedy π_z.

Reward tasks use reward_wr_inference. Goal tasks use B(s_g) of an extremal state.
Scores are kinematics aligned with the prompt, not the gym walking return.
"""

from __future__ import annotations

import torch


def _col(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)[:, :1]


def reward_label(name: str, obs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    if name == "gym":
        return _col(rewards)
    if name == "forward":
        return _col(obs[:, 8])
    if name == "backward":
        return -_col(obs[:, 8])
    if name == "high":
        return _col(obs[:, 0])
    if name == "low":
        return -_col(obs[:, 0])
    if name == "stand":
        return _col(obs[:, 0]) - obs[:, 8:9].abs() - obs[:, 1:2].abs()
    raise ValueError(f"unknown reward task {name}")


def reward_score(name: str, stats: dict) -> float:
    if name == "gym":
        return stats["return"]
    if name == "forward":
        return stats["xvel"]
    if name == "backward":
        return -stats["xvel"]
    if name == "high":
        return stats["height"]
    if name == "low":
        return -stats["height"]
    if name == "stand":
        return stats["height"] - abs(stats["xvel"]) - stats["abs_angle"]
    raise ValueError(name)


def goal_index(name: str, obs: torch.Tensor) -> int:
    if name == "fwd":
        return int(obs[:, 8].argmax())
    if name == "bwd":
        return int(obs[:, 8].argmin())
    if name == "high":
        return int(obs[:, 0].argmax())
    if name == "low":
        return int(obs[:, 0].argmin())
    raise ValueError(f"unknown goal task {name}")


def goal_score(name: str, stats: dict) -> float:
    if name == "fwd":
        return stats["xvel"]
    if name == "bwd":
        return -stats["xvel"]
    if name == "high":
        return stats["height"]
    if name == "low":
        return -stats["height"]
    raise ValueError(name)


REWARD_TASKS = ("gym", "forward", "backward", "high", "low", "stand")
GOAL_TASKS = ("fwd", "bwd", "high", "low")


def _log_stats(prefix: str, stats: dict, score: float) -> dict:
    return {
        f"{prefix}/score": score,
        f"{prefix}/return": stats["return"],
        f"{prefix}/len": stats["len"],
        f"{prefix}/xvel": stats["xvel"],
        f"{prefix}/height": stats["height"],
        f"{prefix}/abs_angle": stats["abs_angle"],
    }


@torch.no_grad()
def run_skill_eval(agent, world, episode_cnt: int = 5) -> dict:
    n = min(agent.cfg.eval_inference_samples, world.buffer.size)
    if n <= 0:
        return {}
    batch = world.sample(n)
    obs = batch["observations"]
    rewards = batch["rewards"]
    env = world.skill_eval_env()
    logs = {}

    reward_tasks = REWARD_TASKS if obs.shape[-1] > 8 else ("gym",)
    for name in reward_tasks:
        r = reward_label(name, obs, rewards)
        z = agent.reward_wr_inference(obs, r)
        stats = world.eval_stats(
            agent.get_rollout_policy(explore=False, z=z),
            episode_cnt,
            env=env,
        )
        logs.update(_log_stats(f"eval/r/{name}", stats, reward_score(name, stats)))

    if obs.shape[-1] > 8:
        for name in GOAL_TASKS:
            idx = goal_index(name, obs)
            z = agent.goal_inference(obs[idx : idx + 1])
            stats = world.eval_stats(
                agent.get_rollout_policy(explore=False, z=z),
                episode_cnt,
                env=env,
            )
            logs.update(_log_stats(f"eval/g/{name}", stats, goal_score(name, stats)))

    scores = [logs[k] for k in logs if k.endswith("/score") and k.startswith("eval/r/")]
    if scores:
        logs["eval/r/mean_score"] = sum(scores) / len(scores)
    return logs
