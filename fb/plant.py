"""Named / patched goal plants for the live walker."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from fb.agent import Agent
from fb.walker import GOAL_POSES, WALKER_OBS_DIM, task_obs

_OBS_RE = re.compile(r"^(?:o|obs|s|state)\[?(\d+)\]?$", re.IGNORECASE)

JOINT_INDEX = {
    "hipr": 0,
    "righthip": 0,
    "rhip": 0,
    "kneer": 1,
    "rightknee": 1,
    "rknee": 1,
    "ankler": 2,
    "rightankle": 2,
    "rankle": 2,
    "hipl": 3,
    "lefthip": 3,
    "lhip": 3,
    "kneel": 4,
    "leftknee": 4,
    "lknee": 4,
    "anklel": 5,
    "leftankle": 5,
    "lankle": 5,
}

SCALAR_KEYS = {
    "pitch": "pitch",
    "rooty": "pitch",
    "xvel": "xvel",
    "vx": "xvel",
    "vel": "xvel",
    "height": "height",
    "h": "height",
    "rootz": "rootz",
}

JOINT_NAMES = ("hipR", "kneeR", "ankleR", "hipL", "kneeL", "ankleL")
NAMED_WATCH = {
    "stand": {"height", "xvel", "hipR", "hipL"},
    "forward": {"height", "xvel"},
    "backward": {"height", "xvel"},
    "hip-r": {"hipR", "hipL", "kneeR"},
    "hip-l": {"hipL", "hipR", "kneeL"},
}
OBS_WATCH = {14: "height"}


def _norm_key(key: str) -> str:
    return key.strip().replace("_", "").replace("-", "").lower()


@dataclass
class GoalPlant:
    name: str = "custom"
    pitch: float = 0.0
    joints: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    xvel: float = 0.0
    height: float | None = None
    rootz: float | None = None
    obs_overrides: dict[int, float] = field(default_factory=dict)
    watched: set[str] = field(default_factory=set)

    def load_named(self, name: str) -> None:
        pitch, joints, xvel = GOAL_POSES[name]
        self.name = name
        self.pitch = float(pitch)
        self.joints = np.asarray(joints, dtype=np.float64).copy()
        self.xvel = float(xvel)
        self.height = None
        self.rootz = None
        self.obs_overrides.clear()
        self.watched = set(NAMED_WATCH.get(name, ()))

    def describe(self) -> str:
        bits = [
            self.name,
            f"pitch={self.pitch:+.2f}",
            f"xvel={self.xvel:+.2f}",
            f"hipR={self.joints[0]:+.2f}",
            f"kneeR={self.joints[1]:+.2f}",
            f"hipL={self.joints[3]:+.2f}",
            f"kneeL={self.joints[4]:+.2f}",
        ]
        if self.height is not None:
            bits.append(f"height={self.height:.2f}")
        if self.rootz is not None:
            bits.append(f"rootz={self.rootz:+.2f}")
        for idx in sorted(self.obs_overrides):
            bits.append(f"o{idx}={self.obs_overrides[idx]:+.3f}")
        return " ".join(bits)


@dataclass
class GoalPatch:
    named: str | None = None
    pitch: float | None = None
    xvel: float | None = None
    height: float | None = None
    rootz: float | None = None
    joints: dict[int, float] = field(default_factory=dict)
    obs_overrides: dict[int, float] = field(default_factory=dict)
    clear_obs: bool = False

    def apply(self, plant: GoalPlant) -> GoalPlant:
        if self.named is not None:
            plant.load_named(self.named)
        if self.pitch is not None:
            plant.pitch = self.pitch
            plant.watched.add("pitch")
            if self.named is None:
                plant.name = "custom"
        if self.xvel is not None:
            plant.xvel = self.xvel
            plant.watched.add("xvel")
            if self.named is None:
                plant.name = "custom"
        if self.height is not None:
            plant.height = self.height
            plant.watched.add("height")
            if self.named is None:
                plant.name = "custom"
        if self.rootz is not None:
            plant.rootz = self.rootz
            plant.watched.add("rootz")
            if self.named is None:
                plant.name = "custom"
        for idx, value in self.joints.items():
            plant.joints[idx] = value
            plant.watched.add(JOINT_NAMES[idx])
            if self.named is None:
                plant.name = "custom"
        if self.clear_obs:
            plant.obs_overrides.clear()
        plant.obs_overrides.update(self.obs_overrides)
        for idx in self.obs_overrides:
            plant.watched.add(OBS_WATCH.get(idx, f"o{idx}"))
        return plant


def parse_patch(line: str) -> tuple[str, GoalPatch | None]:
    text = line.strip()
    if not text or text.startswith("#"):
        return "empty", None
    tokens = text.replace(",", " ").split()
    if len(tokens) == 1 and tokens[0] in {"reset", "quit", "exit", "help"}:
        return ("quit" if tokens[0] == "exit" else tokens[0]), None

    patch = GoalPatch()
    for tok in tokens:
        if "=" not in tok:
            if tok in GOAL_POSES:
                patch.named = tok
                patch.clear_obs = True
                continue
            if tok in {"reset", "quit", "exit", "help"}:
                raise ValueError(f"{tok} must be on its own line")
            raise ValueError(f"unknown token {tok!r}; names: {', '.join(GOAL_POSES)}")
        key, raw = tok.split("=", 1)
        value = float(raw)
        match = _OBS_RE.fullmatch(key.strip())
        if match:
            idx = int(match.group(1))
            if not 0 <= idx < WALKER_OBS_DIM:
                raise ValueError(f"obs index {idx} out of range 0..{WALKER_OBS_DIM - 1}")
            patch.obs_overrides[idx] = value
            continue
        norm = _norm_key(key)
        if norm in SCALAR_KEYS:
            setattr(patch, SCALAR_KEYS[norm], value)
            continue
        if norm in JOINT_INDEX:
            patch.joints[JOINT_INDEX[norm]] = value
            continue
        raise ValueError(f"unknown key {key!r}")
    return "apply", patch


def live_state(env) -> dict[str, float]:
    qpos = np.asarray(env.physics.data.qpos, dtype=float)
    qvel = np.asarray(env.physics.data.qvel, dtype=float)
    out = {
        "height": float(env.physics.torso_height()),
        "pitch": float(qpos[2]),
        "xvel": float(qvel[1]),
        "rootz": float(qpos[0]),
    }
    for i, name in enumerate(JOINT_NAMES):
        out[name] = float(qpos[3 + i])
    return out


def plant_targets(plant: GoalPlant, planted: dict[str, float]) -> dict[str, float]:
    tgt = {
        "height": float(plant.height if plant.height is not None else planted["height"]),
        "pitch": float(plant.pitch),
        "xvel": float(plant.xvel),
        "rootz": float(plant.rootz if plant.rootz is not None else planted.get("rootz", 0.0)),
    }
    for i, name in enumerate(JOINT_NAMES):
        tgt[name] = float(plant.joints[i])
    return tgt


def plant_goal(env, plant: GoalPlant) -> tuple[np.ndarray, dict[str, float]]:
    qpos = np.zeros(9, dtype=np.float64)
    qvel = np.zeros(9, dtype=np.float64)
    if plant.rootz is not None:
        qpos[0] = plant.rootz
    qpos[2] = plant.pitch
    qpos[3:] = plant.joints
    qvel[1] = plant.xvel
    with env.physics.reset_context():
        env.physics.set_state(np.concatenate([qpos, qvel]))
    if plant.height is not None:
        measured = float(env.physics.torso_height())
        qpos[0] += plant.height - measured
        with env.physics.reset_context():
            env.physics.set_state(np.concatenate([qpos, qvel]))
    obs = np.asarray(task_obs(env), dtype=np.float32).copy()
    feat = live_state(env)
    for idx, value in plant.obs_overrides.items():
        obs[idx] = np.float32(value)
    return obs, feat


@dataclass
class GhostPose:
    torso: np.ndarray
    geom_xpos: np.ndarray
    geom_xmat: np.ndarray


def snapshot_ghost(env) -> GhostPose:
    return GhostPose(
        torso=np.asarray(env.physics.named.data.xpos["torso"], dtype=float).copy(),
        geom_xpos=np.asarray(env.physics.data.geom_xpos, dtype=float).copy(),
        geom_xmat=np.asarray(env.physics.data.geom_xmat, dtype=float).copy(),
    )


@torch.no_grad()
def infer_z(
    env, agent: Agent, plant: GoalPlant
) -> tuple[torch.Tensor, dict[str, float], GhostPose]:
    saved = np.array(env.physics.get_state())
    try:
        obs, feat = plant_goal(env, plant)
        ghost = snapshot_ghost(env)
        z = agent.goal_inference(torch.from_numpy(obs).to(agent.device).unsqueeze(0))
        return z, feat, ghost
    finally:
        with env.physics.reset_context():
            env.physics.set_state(saved)
