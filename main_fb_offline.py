"""FB on the ExORL walker proto buffer, then a mixed online stage.

Stage 1 fits F/B/π on proto transitions only. The stored reward is ignored,
and the episodes are not expert sequences. Train z is the Walker mix
(half the sphere, half project(B(s'))).

Stage 2 rolls out in the same dm_control walker and mixes proto into each
batch, with the offline fraction decaying from 1 to 0.5. Half of those
episodes hold a goal z = project(B(s_g)); the rest hold a sphere z and
refresh it every z_horizon steps.

An auxiliary Q learns the action-rate penalty -0.1 * ||a_t - a_{t-1}||^2.
The actor maximizes Q_fb and that Q. FB itself still ignores the reward.

  uv run python main_fb_offline.py --device cuda
  uv run python main_fb_offline.py --play --ckpt checkpoints/fb_walker_proto.pt --goal forward
  uv run python main_fb_offline.py --play --render --goal stand
  .venv/bin/mjpython play_fb_live.py --ckpt checkpoints/fb_walker_proto.pt
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

from algos.fb import Agent, AlgoConfig
from buffer import Buffer, resolve_offline_ratio, sample_mixed

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

_FEAT_KEYS = ("height", "pitch", "xvel", "right_hip", "left_hip")


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


def _task_obs(env) -> np.ndarray:
    obs = env.task.get_observation(env.physics)
    return np.concatenate([np.ravel(np.asarray(v, dtype=np.float32)) for v in obs.values()])


def _features(env) -> dict[str, float]:
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
    return _task_obs(env), _features(env)


def _action_rate_reward(action: np.ndarray, prev: np.ndarray, scale: float) -> np.ndarray:
    delta = np.asarray(action, dtype=np.float32) - np.asarray(prev, dtype=np.float32)
    return (scale * np.square(delta).sum(axis=-1, keepdims=True)).astype(np.float32)


def _read_episode(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    # action[0] is the dummy row of zeros, so the first real step has prev = 0.
    return obs[:-1], action[1:], obs[1:], terminated, action[:-1]


class TransitionDataset:
    """Static (s, a, s') table. The reward column is dropped."""

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


def load_proto(
    replay_dir: Path,
    task: str,
    expl_agent: str,
    max_transitions: int,
    train_device: str,
    action_rate_scale: float = 0.0,
):
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
        rewards = _action_rate_reward(actions, prev_actions, action_rate_scale)
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


def _clip_action(action) -> np.ndarray:
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)


def _done_flags(time_step) -> tuple[float, float]:
    discount = 1.0 if time_step.discount is None else float(time_step.discount)
    terminated = float(discount == 0.0)
    truncated = float(bool(time_step.last()) and not terminated)
    return terminated, truncated


@torch.no_grad()
def _act(agent: Agent, obs: np.ndarray, z: torch.Tensor) -> np.ndarray:
    action = agent.actor(agent._as_obs(obs), z, agent.cfg.actor_std).sample()
    return _clip_action(action)


def _begin_episode(env, agent: Agent, goal_ratio: float):
    """Hold one z for the episode. Goal episodes use project(B(s_g))."""
    time_step = env.reset()
    obs = flat_obs(time_step)
    if random.random() < goal_ratio:
        name = random.choice(list(GOAL_POSES))
        gobs, _ = goal_state(env, name)
        z = agent.goal_inference(torch.from_numpy(gobs).to(agent.device).unsqueeze(0))
    else:
        name = "sphere"
        z = agent.sample_z()
    return obs, z, name, 0


def _mean_second_half(rows: list[dict]) -> dict[str, float]:
    half = rows[len(rows) // 2 :] or rows
    return {k: float(np.mean([row[k] for row in half])) for k in _FEAT_KEYS}


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
            time_step = env.step(_clip_action(action))
            rows.append(_features(env))
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
        for key in _FEAT_KEYS:
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


def _goal_names(spec: str) -> list[str]:
    if spec == "all":
        return list(GOAL_POSES)
    if spec not in GOAL_POSES:
        known = ", ".join(GOAL_POSES)
        raise SystemExit(f"unknown goal {spec!r}; choices: {known}, all")
    return [spec]


def _build_agent(env, device: str, use_aux: bool) -> Agent:
    time_step = env.reset()
    obs = flat_obs(time_step)
    action_dim = int(np.prod(env.action_spec().shape))
    cfg = AlgoConfig()
    cfg.use_cpr = False
    cfg.use_aux = use_aux
    cfg.compile_nets = device.startswith("cuda")
    cfg.fused_adam = device.startswith("cuda")
    return Agent(obs, np.zeros(action_dim, dtype=np.float32), cfg, device=device)


def _default_replay_dir() -> Path:
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


def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=Path, default=_default_replay_dir())
    parser.add_argument("--task", default="walker_stand")
    parser.add_argument("--expl-agent", default="proto")
    parser.add_argument("--max-transitions", type=int, default=0, help="0 loads every proto transition")
    parser.add_argument("--updates", type=int, default=1_000_000, help="stage-1 gradient steps on proto only")
    parser.add_argument("--online-frames", type=int, default=1_000_000, help="stage-2 env steps; 0 skips it")
    parser.add_argument("--online-buffer-size", type=int, default=1_000_000)
    parser.add_argument("--offline-ratio-start", type=float, default=1.0)
    parser.add_argument("--offline-ratio-end", type=float, default=0.5)
    parser.add_argument("--goal-episode-ratio", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/fb_walker_proto_aux.pt"))
    parser.add_argument("--aux", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--goal", default="all")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = _parse()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    names = _goal_names(args.goal)
    env = make_dmc(args.task, args.seed)

    if args.play:
        agent = Agent.load(args.ckpt, device=args.device)
        if agent.obs_dim != 24 or agent.action_dim != 6:
            raise SystemExit(
                f"checkpoint is obs {agent.obs_dim} / action {agent.action_dim}; "
                "this env is the 24-d dm_control walker"
            )
        eval_goals(env, agent, names, episodes=1, render=args.render)
        return

    agent = _build_agent(env, args.device, args.aux)
    data = load_proto(
        args.replay_dir,
        args.task,
        args.expl_agent,
        args.max_transitions,
        args.device,
        action_rate_scale=agent.cfg.action_rate_scale if args.aux else 0.0,
    )
    run = None
    if args.wandb != "disabled":
        import wandb

        run = wandb.init(
            project="oorl",
            group="fb-offline",
            name=f"fb_{args.expl_agent}_{args.task}",
            mode=args.wandb,
            config=vars(args),
        )

    def _log(info: dict, step: int) -> None:
        if run is not None and info:
            run.log(info, step=step)

    def _eval(step: int) -> None:
        print(f"\ngoal eval @ {step}")
        metrics = eval_goals(env, agent, names, args.eval_episodes, render=False)
        _log(metrics, step)
        agent.save(args.ckpt)
        print(f"saved {args.ckpt}")

    pbar = tqdm.tqdm(range(1, args.updates + 1), desc="fb proto")
    for step in pbar:
        info = agent.update(data.sample(args.batch_size))
        if step % args.log_every == 0:
            pbar.set_postfix(fb=f"{info.get('train/fb_loss', 0):.3f}")
            _log(info, step)
        if step % args.eval_every == 0 or step == args.updates:
            _eval(step)

    if args.online_frames > 0:
        online = Buffer(
            flat_obs(env.reset()),
            np.zeros(agent.action_dim, dtype=np.float32),
            size=args.online_buffer_size,
            buffer_device="cpu",
            train_device=args.device,
            store_prev_action=args.aux,
        )
        obs, z, z_name, ep_step = _begin_episode(env, agent, args.goal_episode_ratio)
        prev_action = np.zeros(agent.action_dim, dtype=np.float32)
        print(
            f"online stage: {args.online_frames} frames, "
            f"proto ratio {args.offline_ratio_start} -> {args.offline_ratio_end}, "
            f"goal episodes {args.goal_episode_ratio}"
        )
        pbar = tqdm.tqdm(range(args.online_frames), desc="fb online")
        for t in pbar:
            if z_name == "sphere" and ep_step > 0 and ep_step % agent.cfg.z_horizon == 0:
                z = agent.sample_z()
            action = _act(agent, obs, z)
            time_step = env.step(action)
            nxt = flat_obs(time_step)
            terminated, truncated = _done_flags(time_step)
            if args.aux:
                reward = float(
                    np.asarray(
                        _action_rate_reward(action, prev_action, agent.cfg.action_rate_scale)
                    ).reshape(-1)[0]
                )
            else:
                reward = 0.0
            online.append(
                obs,
                action,
                nxt,
                reward,
                terminated,
                truncated,
                prev_action=prev_action if args.aux else None,
            )
            ep_step += 1
            if time_step.last():
                obs, z, z_name, ep_step = _begin_episode(env, agent, args.goal_episode_ratio)
                prev_action = np.zeros(agent.action_dim, dtype=np.float32)
            else:
                obs = nxt
                prev_action = action

            ratio = resolve_offline_ratio(
                "linear_decay",
                offline_size=data.size,
                online_size=online.size,
                t=t,
                T=args.online_frames,
                ratio_start=args.offline_ratio_start,
                ratio_end=args.offline_ratio_end,
            )
            batch = sample_mixed(data, online, args.batch_size, ratio)
            info = agent.update(batch)
            step = args.updates + t + 1
            if step % args.log_every == 0:
                info["train/offline_ratio"] = ratio
                info["train/online_size"] = online.size
                pbar.set_postfix(fb=f"{info.get('train/fb_loss', 0):.3f}", mix=f"{ratio:.2f}")
                _log(info, step)
            if step % args.eval_every == 0 or t + 1 == args.online_frames:
                _eval(step)
                obs, z, z_name, ep_step = _begin_episode(env, agent, args.goal_episode_ratio)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
