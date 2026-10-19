"""Unsupervised Forward-Backward training (F / B / π).

Collects with a held z; updates on a fresh mixed-z batch.
Evals walking z_r/z_g plus BFM-Zero-style reward/goal skill probes.
Train env is timeout-only when world.terminate_when_unhealthy=false.

  python main_fb.py
  python main_fb.py --task Walker2d-v5
  python main_fb.py --config configs/fb/Walker2d-v5.yaml
"""

import datetime
from pathlib import Path

import tqdm

from algos.fb import Agent
from config import FBConfig, parse
from train_utils import as_log_dict, init_wandb
from world import World


def _eval_prompts(agent: Agent, world: World, episode_cnt: int) -> dict:
    out = {}
    for name, policy in agent.eval_policies(world).items():
        ev = world.eval(policy, episode_cnt=episode_cnt)
        out.update({k.replace("eval/", f"eval/{name}/", 1): v for k, v in ev.items()})
    skill_episodes = min(episode_cnt, 5)
    out.update(agent.eval_skills(world, episode_cnt=skill_episodes))
    return out


def main():
    cfg = parse(FBConfig)
    if cfg.algo.name != "fb":
        raise ValueError("main_fb.py only trains fb")

    world = World(cfg.world)
    agent = Agent(
        world.observation_example,
        world.action_example,
        cfg.algo,
        device=cfg.world.train_device,
    )

    run_name = f"fb_{cfg.world.task_name}_{datetime.datetime.now()}"
    run = init_wandb(name=run_name, config=cfg)

    observation, info = world.reset()
    for _ in tqdm.tqdm(range(cfg.train.warmup_steps), desc="buffer warmup"):
        action = world.action_space.sample()
        next_observation, reward, terminated, truncated, info = world.step(action)
        if terminated or truncated:
            observation, info = world.reset()
        else:
            observation = next_observation

    pbar = tqdm.tqdm(range(cfg.train.total_frames), desc="fb train")
    update_cnt = 0
    episode_len = 0
    episode_return = 0.0
    for i in pbar:
        infos = {}
        action = agent.get_rollout_policy()(observation).cpu().numpy()
        next_observation, reward, terminated, truncated, info = world.step(action)
        if terminated or truncated:
            infos.update(
                {
                    "rollout/episode_len": episode_len + 1,
                    "rollout/episode_return": episode_return + reward,
                    "rollout/terminated": float(terminated),
                    "rollout/truncated": float(truncated),
                }
            )
            episode_len = 0
            episode_return = 0.0
            observation, info = world.reset()
        else:
            episode_len += 1
            episode_return += reward
            observation = next_observation

        if world.buffer.size > cfg.train.batch_size and i % cfg.train.rev_utd_ratio == 0:
            update_cnt += 1
            train_info = agent.update(world.sample(cfg.train.batch_size))
            if update_cnt % cfg.train.log_freq == 0:
                infos.update(train_info)

        if update_cnt > 0 and update_cnt % cfg.train.eval_freq == 0:
            infos.update(_eval_prompts(agent, world, cfg.train.eval_episodes))

        if infos:
            run.log(data=as_log_dict(infos), step=i)

    final = _eval_prompts(agent, world, cfg.train.eval_episodes)
    keys = [k for k in final if k.endswith("mean_episode_return") or k.endswith("/score")]
    print("final eval:", {k: final[k] for k in keys})
    run.log(data=as_log_dict(final), step=cfg.train.total_frames - 1)

    ckpt = Path("checkpoints") / f"fb_{cfg.world.task_name}.pt"
    agent.save(ckpt)
    print(f"saved {ckpt}")
    run.finish()


if __name__ == "__main__":
    main()
