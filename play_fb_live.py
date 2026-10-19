"""Live DMC walker: window + FIFO goal / state edits.

Uses the finished non-aux proto checkpoint. On macOS the viewer must be
started with mjpython.

  .venv/bin/mjpython play_fb_live.py --ckpt checkpoints/fb_walker_proto.pt
  echo forward > /tmp/fb_goal
  echo xvel=1.5 > /tmp/fb_goal
  echo hipR=1.2 hipL=0 > /tmp/fb_goal
  echo forward hipR=0.8 o14=1.2 > /tmp/fb_goal

Each line updates the planted goal state s_g, then z = project(B(s_g)).
Named goals load a preset. key=value patches one or more fields of the
current plant. oN=x overwrites observation dim N after planting. `reset`
restarts the walker and keeps z. `quit` closes the window.
"""

from __future__ import annotations

import argparse
import os
import re
import select
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from algos.fb import Agent
from main_fb_offline import (
    GOAL_POSES,
    _clip_action,
    _task_obs,
    flat_obs,
    make_dmc,
)

_OBS_RE = re.compile(r"^(?:o|obs|s|state)\[?(\d+)\]?$", re.IGNORECASE)

_JOINT_INDEX = {
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

_SCALAR_KEYS = {
    "pitch": "pitch",
    "rooty": "pitch",
    "xvel": "xvel",
    "vx": "xvel",
    "vel": "xvel",
    "height": "height",
    "h": "height",
    "rootz": "rootz",
}

_JOINT_NAMES = ("hipR", "kneeR", "ankleR", "hipL", "kneeL", "ankleL")
_HUD_ROWS = ("height", "xvel", "pitch", *_JOINT_NAMES)
_NAMED_WATCH = {
    "stand": {"height", "xvel", "hipR", "hipL"},
    "forward": {"height", "xvel"},
    "backward": {"height", "xvel"},
    "hip-r": {"hipR", "hipL", "kneeR"},
    "hip-l": {"hipL", "hipR", "kneeL"},
}
_OBS_WATCH = {14: "height"}
_THRESH = {
    "height": 0.15,
    "xvel": 0.30,
    "pitch": 0.20,
    "rootz": 0.15,
}


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
        self.watched = set(_NAMED_WATCH.get(name, ()))

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
            plant.watched.add(_JOINT_NAMES[idx])
            if self.named is None:
                plant.name = "custom"
        if self.clear_obs:
            plant.obs_overrides.clear()
        plant.obs_overrides.update(self.obs_overrides)
        for idx in self.obs_overrides:
            plant.watched.add(_OBS_WATCH.get(idx, f"o{idx}"))
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
            if not 0 <= idx < 24:
                raise ValueError(f"obs index {idx} out of range 0..23")
            patch.obs_overrides[idx] = value
            continue
        norm = _norm_key(key)
        if norm in _SCALAR_KEYS:
            setattr(patch, _SCALAR_KEYS[norm], value)
            continue
        if norm in _JOINT_INDEX:
            patch.joints[_JOINT_INDEX[norm]] = value
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
    for i, name in enumerate(_JOINT_NAMES):
        out[name] = float(qpos[3 + i])
    return out


def plant_targets(plant: GoalPlant, planted: dict[str, float]) -> dict[str, float]:
    tgt = {
        "height": float(plant.height if plant.height is not None else planted["height"]),
        "pitch": float(plant.pitch),
        "xvel": float(plant.xvel),
        "rootz": float(plant.rootz if plant.rootz is not None else planted.get("rootz", 0.0)),
    }
    for i, name in enumerate(_JOINT_NAMES):
        tgt[name] = float(plant.joints[i])
    return tgt


def _ok(name: str, now: float, tgt: float) -> bool:
    return abs(now - tgt) <= _THRESH.get(name, 0.25)


def hud_lines(plant: GoalPlant, now: dict[str, float], tgt: dict[str, float]) -> tuple[str, str]:
    left = [f"goal {plant.name}"]
    right = ["now / tgt"]
    for name in _HUD_ROWS:
        mark = "*" if name in plant.watched else " "
        status = "ok  " if _ok(name, now[name], tgt[name]) else "miss"
        left.append(f"{mark}{name:6} {status}")
        right.append(f"{now[name]:+.2f} / {tgt[name]:+.2f}")
    return "\n".join(left), "\n".join(right)


@dataclass
class GhostPose:
    torso: np.ndarray
    geom_xpos: np.ndarray
    geom_xmat: np.ndarray


def _snapshot_ghost(env) -> GhostPose:
    return GhostPose(
        torso=np.asarray(env.physics.named.data.xpos["torso"], dtype=float).copy(),
        geom_xpos=np.asarray(env.physics.data.geom_xpos, dtype=float).copy(),
        geom_xmat=np.asarray(env.physics.data.geom_xmat, dtype=float).copy(),
    )


def draw_ghost(viewer, env, ghost: GhostPose) -> None:
    import mujoco

    scn = viewer.user_scn
    if scn is None or ghost is None:
        return
    live = np.asarray(env.physics.named.data.xpos["torso"], dtype=float)
    offset = np.array([live[0] - ghost.torso[0], 0.85, 0.0])
    model = env.physics.model
    rgba = np.array([0.15, 0.65, 1.0, 0.35], dtype=np.float32)
    scn.ngeom = 0
    for i in range(int(model.ngeom)):
        if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if scn.ngeom >= scn.maxgeom:
            break
        geom = scn.geoms[scn.ngeom]
        pos = ghost.geom_xpos[i] + offset
        mat = ghost.geom_xmat[i]
        mujoco.mjv_initGeom(
            geom,
            int(model.geom_type[i]),
            np.asarray(model.geom_size[i], dtype=np.float64),
            pos,
            mat,
            rgba,
        )
        scn.ngeom += 1


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
    # obs/xmat are stale inside reset_context (forward runs in after_reset).
    obs = np.asarray(_task_obs(env), dtype=np.float32).copy()
    feat = live_state(env)
    for idx, value in plant.obs_overrides.items():
        obs[idx] = np.float32(value)
    return obs, feat


@torch.no_grad()
def infer_z(
    env, agent: Agent, plant: GoalPlant
) -> tuple[torch.Tensor, dict[str, float], GhostPose]:
    saved = np.array(env.physics.get_state())
    try:
        obs, feat = plant_goal(env, plant)
        ghost = _snapshot_ghost(env)
        z = agent.goal_inference(torch.from_numpy(obs).to(agent.device).unsqueeze(0))
        return z, feat, ghost
    finally:
        with env.physics.reset_context():
            env.physics.set_state(saved)


def ensure_fifo(path: Path) -> None:
    if path.exists():
        if stat.S_ISFIFO(path.stat().st_mode):
            return
        raise SystemExit(f"{path} exists and is not a fifo")
    os.mkfifo(path)


def open_fifo(path: Path):
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    return os.fdopen(fd, "r", buffering=1)


def _track_torso(viewer, env) -> None:
    import mujoco

    body_id = mujoco.mj_name2id(env.physics.model.ptr, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if body_id < 0:
        return
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body_id
    viewer.cam.distance = 3.4
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -12


def _help() -> str:
    joints = "hipR hipL kneeR kneeL ankleR ankleL"
    return (
        "names: stand forward backward hip-r hip-l\n"
        f"plant: pitch xvel height rootz {joints}\n"
        "obs:   o14=1.3  obs[14]=1.3  (24-d walker observation)\n"
        "cmd:   reset  quit  help"
    )


def play(args: argparse.Namespace) -> None:
    import mujoco
    import mujoco.viewer

    ensure_fifo(args.pipe)
    pipe = open_fifo(args.pipe)
    buf = ""

    agent = Agent.load(args.ckpt, device=args.device)
    if agent.obs_dim != 24 or agent.action_dim != 6:
        raise SystemExit(
            f"checkpoint is obs {agent.obs_dim} / action {agent.action_dim}; "
            "need the 24-d dm_control walker proto ckpt"
        )

    env = make_dmc(args.task, args.seed)
    plant = GoalPlant()
    plant.load_named(args.goal)
    z, planted, ghost = infer_z(env, agent, plant)
    tgt = plant_targets(plant, planted)
    time_step = env.reset()
    dt = float(env.control_timestep())
    last_report = 0.0

    print(_help(), flush=True)
    print(f"pipe {args.pipe}", flush=True)
    print(f"goal {plant.describe()}", flush=True)
    print(
        "blue ghost = planted pose; HUD is now / tgt. "
        "* is a field you set. ok/miss uses 0.25 rad / 0.30 xvel.",
        flush=True,
    )

    viewer = mujoco.viewer.launch_passive(env.physics.model.ptr, env.physics.data.ptr)
    _track_torso(viewer, env)
    policy = agent.get_rollout_policy(explore=False, z=z)

    try:
        while viewer.is_running():
            if select.select([pipe], [], [], 0)[0]:
                chunk = pipe.read()
                if chunk:
                    buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                try:
                    kind, patch = parse_patch(line)
                except ValueError as err:
                    print(f"pipe: {err}", flush=True)
                    continue
                if kind == "empty":
                    continue
                if kind == "help":
                    print(_help(), flush=True)
                    continue
                if kind == "quit":
                    return
                if kind == "reset":
                    time_step = env.reset()
                    print("reset walker", flush=True)
                    continue
                assert patch is not None
                patch.apply(plant)
                z, planted, ghost = infer_z(env, agent, plant)
                tgt = plant_targets(plant, planted)
                policy = agent.get_rollout_policy(explore=False, z=z)
                print(f"goal {plant.describe()}", flush=True)
                print(
                    f"planted height {planted['height']:.2f}  xvel {planted['xvel']:+.2f}  "
                    f"hipR {planted['hipR']:+.2f}  hipL {planted['hipL']:+.2f}  "
                    f"kneeR {planted['kneeR']:+.2f}  ankleR {planted['ankleR']:+.2f}",
                    flush=True,
                )

            if time_step.last():
                time_step = env.reset()
            action = policy(flat_obs(time_step))
            time_step = env.step(_clip_action(action))
            now = live_state(env)
            left, right = hud_lines(plant, now, tgt)
            with viewer.lock():
                draw_ghost(viewer, env, ghost)
            viewer.set_texts(
                (
                    None,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    left,
                    right,
                )
            )
            now_t = time.time()
            if now_t - last_report >= 2.0:
                watched = plant.watched or {"xvel", "hipR", "hipL"}
                bits = []
                for name in _HUD_ROWS:
                    if name not in watched:
                        continue
                    status = "ok" if _ok(name, now[name], tgt[name]) else "miss"
                    bits.append(f"{name} {now[name]:+.2f}/{tgt[name]:+.2f} {status}")
                print("  ".join(bits), flush=True)
                last_report = now_t
            viewer.sync()
            time.sleep(dt)
    finally:
        viewer.close()
        pipe.close()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/fb_walker_proto.pt"))
    parser.add_argument("--pipe", type=Path, default=Path("/tmp/fb_goal"))
    parser.add_argument("--goal", default="stand", choices=list(GOAL_POSES))
    parser.add_argument("--task", default="walker_stand")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    play(_parse())
