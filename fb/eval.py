"""Roll out a goal z and score walker kinematics."""

from __future__ import annotations

import time

import numpy as np
import torch

from fb.agent import Agent
from fb.walker import FEAT_KEYS, clip_action, features, flat_obs, goal_state


def _mean_second_half(rows: list[dict]) -> dict[str, float]:
    half = rows[len(rows) // 2 :] or rows
    return {k: float(np.mean([row[k] for row in half])) for k in FEAT_KEYS}


def rollout_goal(env, agent: Agent, z: torch.Tensor, render: bool = False) -> list[dict]:
    viewer = None
    if render:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.physics.model.ptr, env.physics.data.ptr)
    time_step = env.reset()
    rows = []
    dt = float(env.control_timestep())
    try:
        while not time_step.last():
            if viewer is not None and not viewer.is_running():
                break
            action = agent.get_rollout_policy(explore=False, z=z)(flat_obs(time_step))
            time_step = env.step(clip_action(action))
            rows.append(features(env))
            if viewer is not None:
                viewer.sync()
                time.sleep(dt)
    finally:
        if viewer is not None:
            viewer.close()
    return rows


@torch.no_grad()
def eval_goals(env, agent: Agent, names: list[str], episodes: int, render: bool = False) -> dict:
    out = {}
    z_stand = None
    for name in names:
        obs, target = goal_state(env, name)
        z = agent.goal_inference(torch.from_numpy(obs).to(agent.device).unsqueeze(0))
        if name == "stand":
            z_stand = z
        elif z_stand is not None:
            cos = torch.nn.functional.cosine_similarity(z, z_stand, dim=-1).item()
            out[f"eval/{name}/cos_stand"] = float(cos)
        acc = []
        for _ in range(episodes):
            rows = rollout_goal(env, agent, z, render=render)
            if not rows:
                continue
            got = _mean_second_half(rows)
            acc.append(got)
            upright = float(np.mean([row["height"] > 0.8 for row in rows[len(rows) // 2 :]]))
            out.setdefault(f"eval/{name}/upright", []).append(upright)
        if not acc:
            continue
        for key in FEAT_KEYS:
            got = float(np.mean([row[key] for row in acc]))
            out[f"eval/{name}/{key}"] = got
            out[f"eval/{name}/{key}_goal"] = target[key]
        if f"eval/{name}/upright" in out:
            out[f"eval/{name}/upright"] = float(np.mean(out[f"eval/{name}/upright"]))
        print(
            f"{name:8} height {out[f'eval/{name}/height']:.2f} (goal {target['height']:.2f})"
            f"  xvel {out[f'eval/{name}/xvel']:+.2f} (goal {target['xvel']:+.2f})"
            f"  hipR {out[f'eval/{name}/right_hip']:+.2f} (goal {target['right_hip']:+.2f})"
            f"  hipL {out[f'eval/{name}/left_hip']:+.2f} (goal {target['left_hip']:+.2f})"
        )
    return out
