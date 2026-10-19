"""Online FB-CPR. ExORL is the expert buffer M, not the TD replay.

Same split as official FBcprAgent: online transitions train F/B/π and Q_D;
expert sequences are action-free. z is encode_expert(mean_t B(w_t)), mixed
with goal z and sphere z, and the discriminator treats expert (s, z) as real.

The online env is the same dm_control walker the npz was collected in, so
observations match. Walker2d-v5 stays on main_fb.py.

  uv pip install dm_control
  # inside an ExORL checkout: ./download.sh walker proto
  python main_fb_cpr.py \
    --task walker_walk \
    --expl-agent proto \
    --replay-dir /path/to/exorl/datasets
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import tqdm

from algos.fb import Agent, AlgoConfig
from buffer import Buffer


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
        obs = np.concatenate([np.ravel(np.asarray(v)) for v in obs.values()])
    return np.asarray(obs, dtype=np.float32)


def _done_flags(time_step) -> tuple[float, float]:
    discount = 1.0 if time_step.discount is None else float(time_step.discount)
    terminated = float(discount == 0.0)
    truncated = float(bool(time_step.last()) and not terminated)
    return terminated, truncated


class ExpertSequences:
    """Action-free windows, laid out so encode_expert can view [N, seq, obs]."""

    def __init__(self, episodes: list[np.ndarray], seq_length: int):
        self.seq_length = seq_length
        self.episodes = [ep for ep in episodes if len(ep) >= seq_length + 1]
        if not self.episodes:
            raise RuntimeError(f"no expert episode is at least {seq_length + 1} steps")

    def _window(self, length: int) -> np.ndarray:
        ep = self.episodes[random.randrange(len(self.episodes))]
        start = random.randrange(0, len(ep) - length + 1)
        return ep[start : start + length]

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Official expert_slicer row: obs_t for D, next_obs for encode_expert."""
        seq = self.seq_length
        if batch_size % seq != 0:
            raise ValueError(f"batch_size {batch_size} must be a multiple of seq_length {seq}")
        obs_chunks = []
        next_chunks = []
        for _ in range(batch_size // seq):
            window = self._window(seq + 1)
            obs_chunks.append(window[:-1])
            next_chunks.append(window[1:])
        return (
            torch.from_numpy(np.concatenate(obs_chunks, axis=0)),
            torch.from_numpy(np.concatenate(next_chunks, axis=0)),
        )


def load_expert(replay_dir: Path, task: str, expl_agent: str, max_transitions: int, seq_length: int):
    domain = task.split("_", 1)[0]
    folder = replay_dir / domain / expl_agent / "buffer"
    if not folder.is_dir():
        folder = replay_dir
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz under {folder}")
    episodes = []
    total = 0
    for path in files:
        with np.load(path) as data:
            obs = np.asarray(data["observation"], dtype=np.float32)
        if obs.shape[0] <= 1:
            continue
        obs = obs[1:]
        episodes.append(obs)
        total += len(obs)
        if total >= max_transitions:
            break
    print(f"expert M: {len(episodes)} episodes, {total} frames from {folder}")
    return ExpertSequences(episodes, seq_length)


def _clip_action(action) -> np.ndarray:
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def _rollout(env, agent: Agent, z: torch.Tensor) -> float:
    time_step = env.reset()
    total = 0.0
    while not time_step.last():
        action = agent.get_rollout_policy(explore=False, z=z)(flat_obs(time_step))
        time_step = env.step(_clip_action(action))
        total += float(time_step.reward or 0.0)
    return total


@torch.no_grad()
def _eval(env, agent: Agent, online: Buffer, expert: ExpertSequences, episodes: int) -> dict:
    n = min(agent.cfg.eval_inference_samples, online.size)
    batch = online.sample(n)
    z_r = agent.reward_wr_inference(batch["observations"], batch["rewards"])
    returns = [_rollout(env, agent, z_r) for _ in range(episodes)]

    _, expert_next = expert.sample(agent.cfg.seq_length)
    z_e = agent.encode_expert(expert_next.to(agent.device))[:1]
    expert_returns = [_rollout(env, agent, z_e) for _ in range(episodes)]
    return {
        "eval/zr_return": float(np.mean(returns)),
        "eval/expert_return": float(np.mean(expert_returns)),
    }


def _append(buf: Buffer, obs, action, time_step, z) -> np.ndarray:
    nxt = flat_obs(time_step)
    terminated, truncated = _done_flags(time_step)
    buf.append(obs, action, nxt, float(time_step.reward or 0.0), terminated, truncated, z=z)
    return nxt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="walker_walk")
    parser.add_argument("--expl-agent", default="proto")
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--max-transitions", type=int, default=1_000_000)
    parser.add_argument("--frames", type=int, default=1_000_000)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = AlgoConfig()
    cfg.use_cpr = True
    cfg.compile_nets = False
    cfg.fused_adam = args.device.startswith("cuda")
    # FBcprAgentTrainConfig / train_bfm_zero
    cfg.z_mix_ratio = 0.2
    cfg.expert_asm_ratio = 0.6
    cfg.relabel_ratio = 0.8
    cfg.seq_length = 8
    cfg.reg_coeff = 0.05
    cfg.scale_reg = True
    cfg.grad_penalty = 10.0
    cfg.actor_delayed = 1
    if args.batch_size % cfg.seq_length != 0:
        raise ValueError("batch-size must be a multiple of seq_length 8")

    expert = load_expert(
        args.replay_dir, args.task, args.expl_agent, args.max_transitions, cfg.seq_length
    )
    env = make_dmc(args.task, args.seed)
    time_step = env.reset()
    obs = flat_obs(time_step)
    action_dim = int(np.prod(env.action_spec().shape))

    agent = Agent(obs, np.zeros(action_dim, dtype=np.float32), cfg, device=args.device)
    online = Buffer(
        obs,
        np.zeros(action_dim, dtype=np.float32),
        size=args.buffer_size,
        buffer_device="cpu",
        train_device=args.device,
        z_dim=cfg.z_dim,
    )

    for _ in tqdm.tqdm(range(args.warmup), desc="warmup"):
        action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        z = agent.sample_z(1).squeeze(0).detach().cpu().numpy()
        time_step = env.step(action)
        obs = _append(online, obs, action, time_step, z)
        if time_step.last():
            time_step = env.reset()
            obs = flat_obs(time_step)

    pbar = tqdm.tqdm(range(args.frames), desc="fb-cpr")
    episode_return = 0.0
    for step in pbar:
        action_t = agent.get_rollout_policy(explore=True)(obs)
        z = agent._z.squeeze(0).detach().cpu().numpy()
        action = _clip_action(action_t)
        time_step = env.step(action)
        episode_return += float(time_step.reward or 0.0)
        obs = _append(online, obs, action, time_step, z)
        done = bool(time_step.last())
        if done:
            pbar.set_postfix(ret=f"{episode_return:.1f}")
            episode_return = 0.0
            time_step = env.reset()
            obs = flat_obs(time_step)

        if online.size < args.batch_size:
            continue
        batch = online.sample(args.batch_size)
        expert_obs, expert_next = expert.sample(args.batch_size)
        info = agent.update(batch, expert_obs, expert_next)
        if step % 1000 == 0:
            pbar.set_postfix(fb=f"{info.get('train/fb_loss', 0):.3f}", disc=f"{info.get('train/disc_loss', 0):.3f}")
        if step > 0 and step % args.eval_every == 0:
            ev = _eval(env, agent, online, expert, args.eval_episodes)
            print(f"step {step}: {ev}")
            time_step = env.reset()
            obs = flat_obs(time_step)

    ev = _eval(env, agent, online, expert, args.eval_episodes)
    print("final", ev)
    ckpt = Path("checkpoints") / f"fb_cpr_{args.task}_{args.expl_agent}.pt"
    agent.save(ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
