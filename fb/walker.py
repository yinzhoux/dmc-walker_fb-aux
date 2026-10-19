"""dm_control Walker: env, kinematics, named goal poses."""

from __future__ import annotations

import numpy as np
import torch

# qpos: rootz, rootx, rooty, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle
# Hip is about -20 deg .. +100 deg; knee only bends negative. Positive hip swings the leg forward.
_STRAIGHT = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_HIP_R = (1.2, -0.4, 0.0, 0.0, 0.0, 0.0)
_HIP_L = (0.0, 0.0, 0.0, 1.2, -0.4, 0.0)

# pitch (rooty, rad), 6 joint angles (rad), rootx velocity.
GOAL_POSES = {
    "stand": (0.0, _STRAIGHT, 0.0),
    "forward": (0.0, _STRAIGHT, 1.0),
    "backward": (0.0, _STRAIGHT, -1.0),
    "hip-r": (0.0, _HIP_R, 0.0),
    "hip-l": (0.0, _HIP_L, 0.0),
}

FEAT_KEYS = ("height", "pitch", "xvel", "right_hip", "left_hip")
WALKER_OBS_DIM = 24
WALKER_ACTION_DIM = 6


def make_dmc(task: str, seed: int):
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    domain, name = task.split("_", 1)
    env = suite.load(
        domain,
        name,
        task_kwargs={"random": seed},
        environment_kwargs={"flat_observation": True},
    )
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def flat_obs(time_step) -> np.ndarray:
    obs = time_step.observation
    if isinstance(obs, dict):
        obs = np.concatenate([np.ravel(np.asarray(v, dtype=np.float32)) for v in obs.values()])
    return np.asarray(obs, dtype=np.float32)


def task_obs(env) -> np.ndarray:
    obs = env.task.get_observation(env.physics)
    return np.concatenate([np.ravel(np.asarray(v, dtype=np.float32)) for v in obs.values()])


def features(env) -> dict[str, float]:
    physics = env.physics
    qpos = physics.data.qpos
    qvel = physics.data.qvel
    return {
        "height": float(physics.torso_height()),
        "pitch": float(qpos[2]),
        "xvel": float(qvel[1]),
        "right_hip": float(qpos[3]),
        "left_hip": float(qpos[6]),
    }


def goal_state(env, name: str) -> tuple[np.ndarray, dict[str, float]]:
    pitch, joints, xvel = GOAL_POSES[name]
    qpos = np.zeros(9, dtype=np.float64)
    qvel = np.zeros(9, dtype=np.float64)
    qpos[2] = pitch
    qpos[3:] = joints
    qvel[1] = xvel
    with env.physics.reset_context():
        env.physics.set_state(np.concatenate([qpos, qvel]))
    # forward() runs in after_reset; read obs after that or orientations stay at the default stand.
    return task_obs(env), features(env)


def goal_names(spec: str) -> list[str]:
    if spec == "all":
        return list(GOAL_POSES)
    if spec not in GOAL_POSES:
        known = ", ".join(GOAL_POSES)
        raise SystemExit(f"unknown goal {spec!r}; choices: {known}, all")
    return [spec]


def clip_action(action) -> np.ndarray:
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)


def done_flags(time_step) -> tuple[float, float]:
    discount = 1.0 if time_step.discount is None else float(time_step.discount)
    terminated = float(discount == 0.0)
    truncated = float(bool(time_step.last()) and not terminated)
    return terminated, truncated


def action_rate_reward(action: np.ndarray, prev: np.ndarray, scale: float) -> np.ndarray:
    delta = np.asarray(action, dtype=np.float32) - np.asarray(prev, dtype=np.float32)
    return (scale * np.square(delta).sum(axis=-1, keepdims=True)).astype(np.float32)


def require_walker_ckpt(agent) -> None:
    if agent.obs_dim != WALKER_OBS_DIM or agent.action_dim != WALKER_ACTION_DIM:
        raise SystemExit(
            f"checkpoint is obs {agent.obs_dim} / action {agent.action_dim}; "
            "this env is the 24-d dm_control walker"
        )
