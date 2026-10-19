"""Train FB on ExORL proto, then a mixed online stage.

  uv run python train.py --device cuda
  uv run python train.py --play --ckpt checkpoints/fb_walker_proto.pt --goal forward
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from fb.agent import Agent
from fb.eval import eval_goals
from fb.offline import build_agent, train_online, train_proto
from fb.proto import default_replay_dir, load_proto
from fb.walker import goal_names, make_dmc, require_walker_ckpt


def _parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=default_replay_dir())
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

    names = goal_names(args.goal)
    env = make_dmc(args.task, args.seed)

    if args.play:
        agent = Agent.load(args.ckpt, device=args.device)
        require_walker_ckpt(agent)
        eval_goals(env, agent, names, episodes=1, render=args.render)
        return

    agent = build_agent(env, args.device, args.aux)
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
            project="fb",
            group="fb-offline",
            name=f"fb_{args.expl_agent}_{args.task}",
            mode=args.wandb,
            config=vars(args),
        )

    def log(info: dict, step: int) -> None:
        if run is not None and info:
            run.log(info, step=step)

    def evaluate(step: int) -> None:
        print(f"\ngoal eval @ {step}")
        metrics = eval_goals(env, agent, names, args.eval_episodes, render=False)
        log(metrics, step)
        agent.save(args.ckpt)
        print(f"saved {args.ckpt}")

    train_proto(
        agent,
        data,
        updates=args.updates,
        batch_size=args.batch_size,
        log_every=args.log_every,
        eval_every=args.eval_every,
        log=log,
        evaluate=evaluate,
    )
    if args.online_frames > 0:
        train_online(
            agent,
            env,
            data,
            frames=args.online_frames,
            buffer_size=args.online_buffer_size,
            batch_size=args.batch_size,
            goal_ratio=args.goal_episode_ratio,
            ratio_start=args.offline_ratio_start,
            ratio_end=args.offline_ratio_end,
            proto_updates=args.updates,
            log_every=args.log_every,
            eval_every=args.eval_every,
            use_aux=args.aux,
            log=log,
            evaluate=evaluate,
        )
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
