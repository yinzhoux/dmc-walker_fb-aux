"""Replay a saved FB policy.

Local window (Enter restarts, Z switches skill, Esc closes):
  python play_fb.py --ckpt checkpoints/fb_Walker2d-v5.pt --prompt zr --render human
  python play_fb.py --ckpt checkpoints/fb_Walker2d-v5.pt --prompt forward --render human
  python play_fb.py --ckpt checkpoints/fb_Walker2d-v5.pt --prompt goal-stride-r --render human
  python play_fb.py --ckpt checkpoints/fb_Walker2d-v5_v1.pt --prompt goal-forward --prompt-b goal-backward --render human
  python play_fb.py --ckpt checkpoints/fb_Walker2d-v5_v1.pt --goal-seq speed-up --render human
  python play_fb.py --ckpt checkpoints/fb_Walker2d-v5_v1.pt --goal-seq legs --render human

Headless / remote (no DISPLAY):
  MUJOCO_GL=egl python play_fb.py --ckpt checkpoints/fb_Walker2d-v5.pt --prompt zr --video zr.mp4
"""

from __future__ import annotations

import argparse
import os
import sys


def _configure_gl(render: str, video: str) -> str:
    linux_headless = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
    if video:
        render = "rgb_array"
    if render == "human" and linux_headless:
        print(
            "no DISPLAY: cannot --render human on this Linux machine.\n"
            "write a video instead:\n"
            "  MUJOCO_GL=egl python play_fb.py --ckpt ... --prompt zr --video zr.mp4",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if "MUJOCO_GL" not in os.environ:
        # egl is Linux-only; macOS uses glfw (and has no DISPLAY).
        if linux_headless or render == "rgb_array" and sys.platform.startswith("linux"):
            os.environ["MUJOCO_GL"] = "egl"
        else:
            os.environ["MUJOCO_GL"] = "glfw"
    return render


def _save_video(path: str, frames, fps: int = 30) -> str:
    """imageio+pyav defaults codec=None on some builds; try explicit H.264."""
    import imageio

    attempts = (
        dict(plugin="ffmpeg", codec="libx264", pixelformat="yuv420p"),
        dict(plugin="pyav", codec="libx264", pixel_format="yuv420p"),
        dict(plugin="pyav", codec="h264", pixel_format="yuv420p"),
        dict(codec="libx264"),
    )
    last_err = None
    for kwargs in attempts:
        try:
            imageio.mimsave(path, frames, fps=fps, **kwargs)
            return path
        except Exception as err:
            last_err = err
    gif_path = os.path.splitext(path)[0] + ".gif"
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"mp4 failed ({last_err}); wrote {gif_path} instead", file=sys.stderr)
    return gif_path


def _track_camera(env, render_mode: str):
    """Keep the free camera's look-at point on the walker torso."""
    renderer = getattr(env.unwrapped, "mujoco_renderer", None)
    if renderer is None:
        return None
    viewer = renderer._get_viewer(render_mode)
    import mujoco

    body_id = getattr(viewer, "_track_body", None)
    if body_id is None:
        body_id = mujoco.mj_name2id(
            env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        if body_id < 0:
            body_id = 1
        viewer._track_body = body_id
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = body_id
    return viewer


def _overlay_help(viewer, switch_to: str | None) -> None:
    import mujoco

    corner = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
    viewer.add_overlay(corner, "Enter", "restart")
    if switch_to is not None:
        viewer.add_overlay(corner, "Z", switch_to)


def _consume_key(viewer, key: int, attr: str) -> bool:
    if viewer is None or not hasattr(viewer, "window"):
        return False
    import glfw

    try:
        down = glfw.get_key(viewer.window, key) == glfw.PRESS
    except glfw.GLFWError:
        return False
    was_down = getattr(viewer, attr, False)
    setattr(viewer, attr, down)
    return down and not was_down


def _consume_restart(viewer) -> bool:
    import glfw

    return _consume_key(viewer, glfw.KEY_ENTER, "_enter_was_down") or _consume_key(
        viewer, glfw.KEY_KP_ENTER, "_kp_enter_was_down"
    )


def _consume_switch(viewer) -> bool:
    import glfw

    return _consume_key(viewer, glfw.KEY_Z, "_z_was_down")


def _window_closed(viewer) -> bool:
    if viewer is None or not hasattr(viewer, "window"):
        return False
    import glfw

    try:
        return bool(glfw.window_should_close(viewer.window))
    except glfw.GLFWError:
        return True


SKILL_PROMPTS = ("forward", "backward", "high", "low", "stand", "gym")
# pitch (rad), joint degrees [thigh, leg, foot, thigh_left, leg_left, foot_left], xvel.
# Thigh/leg only bend backward (range -150..0 deg). Feet stay near the ground.
GOAL_POSES = {
    "stand": (0.0, [0, 0, 0, 0, 0, 0], 0.0),
    "hip-back": (0.0, [-35, 0, 0, -35, 0, 0], 0.0),
    "leg-back-r": (0.0, [-50, -20, 10, 0, 0, 0], 0.0),
    "leg-back-l": (0.0, [0, 0, 0, -50, -20, 10], 0.0),
    "lean-fwd": (0.6, [-10, -10, 0, -10, -10, 0], 0.0),
    "lean-back": (-0.6, [-10, 0, 0, -10, 0, 0], 0.0),
    "stride-r": (0.1, [-45, -15, 5, -10, -35, 10], 1.0),
    "back-stride": (-0.1, [-45, -15, 5, -10, -35, 10], -1.0),
    "forward": (0.0, [0, 0, 0, 0, 0, 0], 1.5),
    "backward": (0.0, [0, 0, 0, 0, 0, 0], -1.5),
    "fast": (0.15, [-25, -20, 5, -25, -20, 5], 2.5),
}
GOAL_PROMPTS = tuple(f"goal-{name}" for name in GOAL_POSES)


def _prompt_z(agent, name: str):
    if name == "zr":
        return agent._z_r
    if name == "zg":
        return agent._z_g
    if name == "random":
        return agent.sample_z()
    raise ValueError(f"unknown --prompt {name}")


def _infer_skill_z(agent, env, name: str, steps: int):
    """Infer z from healthy states visited by π_zr.

    A random rollout is mostly the frames just before a fall. softmax then
    builds z from those states, and the skill itself falls. π_zr stays up, so
    forward / high / low are variations inside a gait that already survives.
    """
    import numpy as np
    import torch

    from reward_groups import walker2d_reward_groups

    policy = agent.get_rollout_policy(explore=False, z=agent._z_r)
    observations = []
    rewards = []
    skipped = 0
    observation, _ = env.reset()
    while len(observations) < steps:
        action = policy(np.asarray(observation, dtype=np.float32)).detach().cpu().numpy()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            skipped += 1
            observation, _ = env.reset()
            continue
        groups = walker2d_reward_groups(observation, action, reward, info)
        observations.append(np.asarray(observation, dtype=np.float32))
        rewards.append(groups[name])
        if truncated:
            observation, _ = env.reset()
    obs = torch.from_numpy(np.stack(observations)).to(agent.device)
    rew = torch.tensor(rewards, dtype=torch.float32, device=agent.device)
    print(
        f"prompt {name}: inferred from {steps} healthy π_zr steps, "
        f"reward mean {rew.mean().item():.3f}, skipped falls {skipped}"
    )
    return agent.reward_wr_inference(obs, rew)


def _pose_observation(env, pitch: float, joints_deg, xvel: float):
    """Plant one pose, then read the observation B was trained on."""
    import mujoco
    import numpy as np

    base = env.unwrapped
    qpos = np.array(base.init_qpos, dtype=np.float64)
    qpos[2] = pitch
    qpos[3:] = np.deg2rad(joints_deg)
    qpos[1] = float(base.init_qpos[1])
    base.set_state(qpos, np.zeros(base.model.nv))
    mujoco.mj_forward(base.model, base.data)
    feet = [
        mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        for geom_name in ("foot_geom", "foot_left_geom")
    ]
    lowest = min(float(base.data.geom_xpos[i, 2]) - 0.06 for i in feet)
    qpos[1] -= lowest
    qvel = np.zeros(base.model.nv)
    qvel[0] = xvel
    base.set_state(qpos, qvel)
    return np.asarray(base._get_obs(), dtype=np.float32)


def _goal_observation(env, name: str):
    kind = name.removeprefix("goal-")
    pitch, joints_deg, xvel = GOAL_POSES[kind]
    return _pose_observation(env, pitch, joints_deg, xvel)


def _goal_z(agent, env, name: str):
    import torch

    obs = _goal_observation(env, name)
    z = agent.goal_inference(torch.from_numpy(obs).to(agent.device).unsqueeze(0))
    print(f"prompt {name}: z = project(B(s_g)), height {obs[0]:.2f}, xvel {obs[8]:.2f}")
    return z


def _keys_from_poses(schedule):
    keys = []
    for t, name in schedule:
        pitch, joints, xvel = GOAL_POSES[name]
        keys.append((t, pitch, joints, xvel))
    return keys


# 5 s speed-up. Upright stance; root xvel holds +0.6, then ramps to +1.4.
_STAND = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SPEED_UP = [
    (0.0, 0.0, _STAND, 0.6),
    (2.0, 0.0, _STAND, 0.6),
    (2.8, 0.0, _STAND, 1.4),
    (5.0, 0.0, _STAND, 1.4),
]
# One leg back, then the other. Both holds stay up and the free leg stays near 0.
LEGS = _keys_from_poses([
    (0.0, "stand"),
    (0.4, "leg-back-r"),
    (1.8, "leg-back-r"),
    (2.4, "stand"),
    (2.8, "leg-back-l"),
    (4.2, "leg-back-l"),
    (4.8, "stand"),
    (5.0, "stand"),
])
# Right leg, lean forward, left leg.
POSES = _keys_from_poses([
    (0.0, "stand"),
    (0.3, "leg-back-r"),
    (1.6, "leg-back-r"),
    (2.0, "lean-fwd"),
    (3.3, "lean-fwd"),
    (3.7, "leg-back-l"),
    (4.8, "leg-back-l"),
    (5.0, "stand"),
])
GOAL_SEQS = {
    "speed-up": (SPEED_UP, ((0.2, 2.0, "slow"), (3.0, 5.0, "fast"))),
    "legs": (LEGS, ((0.6, 1.6, "right"), (3.0, 4.0, "left"))),
    "poses": (POSES, ((0.5, 1.4, "right"), (2.2, 3.1, "lean"), (3.9, 4.6, "left"))),
}


def _sample_goal(keys, t: float):
    import numpy as np

    if t <= keys[0][0]:
        _, pitch, joints, xvel = keys[0]
        return pitch, np.asarray(joints, dtype=np.float64), xvel
    if t >= keys[-1][0]:
        _, pitch, joints, xvel = keys[-1]
        return pitch, np.asarray(joints, dtype=np.float64), xvel
    for (t0, p0, j0, v0), (t1, p1, j1, v1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0)
            joints = (1.0 - u) * np.asarray(j0, dtype=np.float64) + u * np.asarray(j1, dtype=np.float64)
            return (1.0 - u) * p0 + u * p1, joints, (1.0 - u) * v0 + u * v1
    _, pitch, joints, xvel = keys[-1]
    return pitch, np.asarray(joints, dtype=np.float64), xvel


def _goal_z_sequence(agent, env, keys, duration: float):
    """One planted goal state per control step, then z = project(B(s_g))."""
    import numpy as np
    import torch

    dt = float(env.unwrapped.dt)
    n = int(round(duration / dt))
    goals = []
    for k in range(n):
        pitch, joints, xvel = _sample_goal(keys, k * dt)
        goals.append(_pose_observation(env, pitch, joints, xvel))
    obs = torch.from_numpy(np.stack(goals)).to(agent.device)
    z = agent.goal_inference(obs)
    env.reset()
    return z, np.stack(goals), dt


def _play_goal_sequence(agent, env, render_mode, keys, duration, episodes, video, windows):
    """Frozen π. At step k, z_k = project(B(s_g(t_k)))."""
    import numpy as np

    z_seq, goals, dt = _goal_z_sequence(agent, env, keys, duration)
    n = z_seq.shape[0]
    print(f"goal sequence: {n} states, dt {dt:.3f}s, duration {n * dt:.2f}s")
    print(f"{'t':>5} {'g_h':>6} {'g_xvel':>7}")
    mark = 0.0
    for k in range(n):
        t = k * dt
        if t + 1e-8 >= mark:
            print(f"{t:5.1f} {goals[k, 0]:6.2f} {goals[k, 8]:+7.2f}")
            mark += 0.5

    frames = []
    interactive = render_mode == "human"
    if interactive:
        print("Enter: restart    Esc: close")
    ep = 0
    while interactive or ep < episodes:
        obs, _ = env.reset()
        xs, xvels, seen = [], [], []
        restarted = False
        fell = None
        k = 0
        while k < n:
            if render_mode == "human":
                viewer = _track_camera(env, "human")
                if viewer is not None:
                    import mujoco

                    corner = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
                    viewer.add_overlay(corner, "Enter", "restart")
                    viewer.add_overlay(
                        corner,
                        f"{k * dt:.2f}s",
                        "pitch {:+.0f}  thigh {:+.0f}/{:+.0f}".format(
                            np.degrees(goals[k, 1]),
                            np.degrees(goals[k, 2]),
                            np.degrees(goals[k, 5]),
                        ),
                    )
                env.render()
                if _consume_restart(viewer):
                    restarted = True
                    break
                if _window_closed(viewer):
                    env.close()
                    return
            elif render_mode == "rgb_array":
                _track_camera(env, "rgb_array")
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            action = _action(agent, obs, z_seq[k : k + 1])
            obs, _, terminated, truncated, _ = env.step(action)
            xs.append(float(env.unwrapped.data.qpos[0]))
            xvels.append(float(env.unwrapped.data.qvel[0]))
            seen.append(np.asarray(obs, dtype=np.float32))
            k += 1
            if terminated or truncated:
                fell = "fall" if terminated else "timeout"
                break
        if restarted:
            print("restart")
            continue
        _report_sequence(xs, xvels, seen, goals, dt, fell, windows)
        ep += 1
    env.close()
    if video:
        if not frames:
            raise RuntimeError("no frames captured")
        written = _save_video(video, frames)
        print(f"wrote {written} ({len(frames)} frames)")


def _report_sequence(xs, xvels, seen, goals, dt, fell, windows):
    import numpy as np

    xs = np.asarray(xs)
    seen = np.stack(seen)
    n = len(xs)
    t = np.arange(n) * dt
    goal_t = np.arange(len(goals)) * dt
    end = "fell" if fell == "fall" else ("timeout" if fell else "full 5s")
    print(f"result: {end}, steps {n}, t={n * dt:.2f}s, x(end)={xs[-1]:.2f} m")

    def _deg(a, i):
        return float(np.degrees(a[:, i].mean()))

    for t0, t1, label in windows:
        m = (t >= t0) & (t < t1)
        gm = (goal_t >= t0) & (goal_t < t1)
        if not np.any(m):
            print(f"  {label} {t0:.1f}-{t1:.1f}s  (not reached)")
            continue
        o, g = seen[m], goals[gm]
        print(
            f"  {label} {t0:.1f}-{t1:.1f}s  "
            f"pitch { _deg(o, 1):+.0f}° (goal {_deg(g, 1):+.0f})  "
            f"thigh {_deg(o, 2):+.0f}/{_deg(o, 5):+.0f} "
            f"(goal {_deg(g, 2):+.0f}/{_deg(g, 5):+.0f})  "
            f"xvel {float(o[:, 8].mean()):+.2f} (goal {float(g[:, 8].mean()):+.2f})"
        )


def _resolve_prompt(agent, env, name: str, inference_steps: int):
    if name in SKILL_PROMPTS:
        return _infer_skill_z(agent, env, name, inference_steps)
    if name in GOAL_PROMPTS:
        return _goal_z(agent, env, name)
    return _prompt_z(agent, name)


def _action(agent, obs, z):
    import numpy as np

    return (
        agent.get_rollout_policy(explore=False, z=z)(np.asarray(obs, dtype=np.float32))
        .detach()
        .cpu()
        .numpy()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="Walker2d-v5")
    p.add_argument(
        "--prompt",
        default="zr",
        choices=("zr", "zg", "random", *SKILL_PROMPTS, *GOAL_PROMPTS),
    )
    p.add_argument(
        "--prompt-b",
        default="",
        choices=("", "zr", "zg", "random", *SKILL_PROMPTS, *GOAL_PROMPTS),
        help="second skill z; Z toggles, or --switch-every steps",
    )
    p.add_argument(
        "--switch-every",
        type=int,
        default=0,
        help="auto-toggle the two z's every N steps (0 = only the Z key)",
    )
    p.add_argument(
        "--goal-seq",
        default="",
        choices=("", *GOAL_SEQS),
        help="5s goal-state sequence; z_t = project(B(s_g(t)))",
    )
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--inference-steps", type=int, default=4096)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--render", default="human", choices=("human", "rgb_array", "none"))
    p.add_argument("--video", default="")
    args = p.parse_args()

    args.render = _configure_gl(args.render, args.video)

    import gymnasium as gym
    import numpy as np
    import torch

    from algos.fb import Agent
    from world import _maybe_flatten_obs_env

    agent = Agent.load(args.ckpt, device=args.device)

    render_mode = None if args.render == "none" else args.render
    env = gym.make(args.task, max_episode_steps=args.max_steps, render_mode=render_mode)
    env = _maybe_flatten_obs_env(env)
    if args.goal_seq:
        keys, windows = GOAL_SEQS[args.goal_seq]
        _play_goal_sequence(
            agent,
            env,
            render_mode,
            keys,
            args.seconds,
            1 if not render_mode == "human" else args.episodes,
            args.video,
            windows,
        )
        return
    names = [args.prompt]
    zs = [_resolve_prompt(agent, env, args.prompt, args.inference_steps)]
    if args.prompt_b:
        names.append(args.prompt_b)
        zs.append(_resolve_prompt(agent, env, args.prompt_b, args.inference_steps))
        z0 = zs[0].reshape(-1).float()
        z1 = zs[1].reshape(-1).float()
        cosine = torch.nn.functional.cosine_similarity(z0, z1, dim=0).item()
        print(f"two z: {names[0]} / {names[1]}, cosine {cosine:.3f}")
    active = 0

    frames = []
    interactive = render_mode == "human"
    if interactive:
        extra = "    Z: switch skill" if args.prompt_b else ""
        print(f"Enter: restart{extra}    Esc: close")
    ep = 0
    while interactive or ep < args.episodes:
        obs, _ = env.reset()
        ret = 0.0
        steps = 0
        restarted = False
        while True:
            if render_mode == "human":
                viewer = _track_camera(env, "human")
                if viewer is not None:
                    other = names[1 - active] if len(zs) == 2 else None
                    _overlay_help(viewer, other)
                env.render()
                if _consume_restart(viewer):
                    restarted = True
                    break
                if _window_closed(viewer):
                    print(f"episode {ep}: return={ret:.1f} len={steps} (window closed)")
                    env.close()
                    return
                manual = _consume_switch(viewer)
            else:
                viewer = None
                manual = False
                if render_mode == "rgb_array":
                    _track_camera(env, "rgb_array")
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
            timed = (
                len(zs) == 2
                and args.switch_every > 0
                and steps > 0
                and steps % args.switch_every == 0
            )
            if len(zs) == 2 and (manual or timed):
                prev = active
                active = 1 - active
                a_prev = _action(agent, obs, zs[prev])
                a_next = _action(agent, obs, zs[active])
                delta = float(np.linalg.norm(a_next - a_prev))
                why = "key" if manual else f"every {args.switch_every}"
                print(
                    f"step {steps}: {names[prev]} -> {names[active]} ({why}), "
                    f"||Δa||={delta:.3f}"
                )
            action = _action(agent, obs, zs[active])
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            steps += 1
            if terminated or truncated:
                break
        how = "restart" if restarted else "done"
        print(f"episode {ep}: return={ret:.1f} len={steps} ({how})")
        ep += 1
    env.close()

    if args.video:
        if not frames:
            raise RuntimeError("no frames captured; egl/osmesa render failed")
        written = _save_video(args.video, frames)
        print(f"wrote {written} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
