"""Passive MuJoCo viewer + FIFO goal edits."""

from __future__ import annotations

import os
import select
import stat
import time
from pathlib import Path

import numpy as np

from fb.agent import Agent
from fb.plant import (
    JOINT_NAMES,
    GoalPlant,
    GhostPose,
    infer_z,
    live_state,
    parse_patch,
    plant_targets,
)
from fb.walker import clip_action, flat_obs, make_dmc, require_walker_ckpt

HUD_ROWS = ("height", "xvel", "pitch", *JOINT_NAMES)
THRESH = {
    "height": 0.15,
    "xvel": 0.30,
    "pitch": 0.20,
    "rootz": 0.15,
}


def _ok(name: str, now: float, tgt: float) -> bool:
    return abs(now - tgt) <= THRESH.get(name, 0.25)


def hud_lines(plant: GoalPlant, now: dict[str, float], tgt: dict[str, float]) -> tuple[str, str]:
    left = [f"goal {plant.name}"]
    right = ["now / tgt"]
    for name in HUD_ROWS:
        mark = "*" if name in plant.watched else " "
        status = "ok  " if _ok(name, now[name], tgt[name]) else "miss"
        left.append(f"{mark}{name:6} {status}")
        right.append(f"{now[name]:+.2f} / {tgt[name]:+.2f}")
    return "\n".join(left), "\n".join(right)


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


def help_text() -> str:
    joints = "hipR hipL kneeR kneeL ankleR ankleL"
    return (
        "names: stand forward backward hip-r hip-l\n"
        f"plant: pitch xvel height rootz {joints}\n"
        "obs:   o14=1.3  obs[14]=1.3  (24-d walker observation)\n"
        "cmd:   reset  quit  help"
    )


def play(args) -> None:
    import mujoco
    import mujoco.viewer

    ensure_fifo(args.pipe)
    pipe = open_fifo(args.pipe)
    buf = ""

    agent = Agent.load(args.ckpt, device=args.device)
    require_walker_ckpt(agent)

    env = make_dmc(args.task, args.seed)
    plant = GoalPlant()
    plant.load_named(args.goal)
    z, planted, ghost = infer_z(env, agent, plant)
    tgt = plant_targets(plant, planted)
    time_step = env.reset()
    dt = float(env.control_timestep())
    last_report = 0.0

    print(help_text(), flush=True)
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
                    print(help_text(), flush=True)
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
            time_step = env.step(clip_action(action))
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
                for name in HUD_ROWS:
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
