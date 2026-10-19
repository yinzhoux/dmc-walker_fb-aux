"""Nested dataclass configs: yaml archive + tyro CLI overlay."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import tyro
import yaml

from algos import choose_algo, fb
from world import WorldCfg


@dataclass
class WandbCfg:
    project: str = "fb"
    group: str = "fb"
    mode: str = "online"


@dataclass
class OnlineTrainCfg:
    total_frames: int = 10_000_000
    rev_utd_ratio: int = 1
    warmup_steps: int = 10_000
    batch_size: int = 256
    log_freq: int = 1000
    eval_freq: int = 10_000
    eval_episodes: int = 10


def _fb_world() -> WorldCfg:
    return WorldCfg(
        suite="gym",
        task_name="Walker2d-v5",
        max_episode_steps=1000,
        terminate_when_unhealthy=False,
        buffer_size=1_000_000,
        train_device="cuda",
        buffer_device="cpu",
        state_normalization=False,
    )


@dataclass
class FBConfig:
    """Unsupervised FB. Extra flags: --task NAME, --config PATH.yaml, --dump-config PATH.yaml."""

    world: WorldCfg = field(default_factory=_fb_world)
    train: OnlineTrainCfg = field(
        default_factory=lambda: OnlineTrainCfg(
            total_frames=1_000_000,
            batch_size=512,
        )
    )
    wandb: WandbCfg = field(default_factory=lambda: WandbCfg(group="fb"))
    algo: fb.AlgoConfig = field(default_factory=fb.AlgoConfig)


def _is_yaml_value(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_yaml_value(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_yaml_value(x) for k, x in v.items())
    return False


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        out = {}
        for f in fields(obj):
            v = getattr(obj, f.name)
            if not _is_yaml_value(v) and not is_dataclass(v):
                continue
            out[f.name] = _to_plain(v)
        return out
    if isinstance(obj, tuple):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    return obj


def to_yaml(cfg) -> str:
    payload = {
        "world": _to_plain(cfg.world),
        "train": _to_plain(cfg.train),
        "wandb": _to_plain(cfg.wandb),
        "algo_name": cfg.algo.name,
        "algo": _to_plain(cfg.algo),
    }
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)


def _overlay(cls, data: dict | None):
    inst = cls()
    if not data:
        return inst
    valid = {f.name for f in fields(cls)}
    for k, v in data.items():
        if k not in valid:
            continue
        cur = getattr(inst, k)
        if is_dataclass(cur) and isinstance(v, dict):
            setattr(inst, k, _overlay(type(cur), v))
        elif callable(cur) and not isinstance(cur, type):
            continue
        elif _is_yaml_value(v):
            setattr(inst, k, v)
    return inst


def from_yaml(config_cls, text: str):
    raw = yaml.safe_load(text) or {}
    algo_name = raw.get("algo_name")
    if algo_name is None and isinstance(raw.get("algo"), dict):
        algo_name = raw["algo"].get("name")
    if not algo_name:
        raise ValueError("yaml needs algo_name (or algo.name)")
    algo_cls = choose_algo(algo_name).AlgoConfig
    train_cls = get_type_hints(config_cls)["train"]
    return config_cls(
        world=_overlay(WorldCfg, raw.get("world")),
        train=_overlay(train_cls, raw.get("train")),
        wandb=_overlay(WandbCfg, raw.get("wandb")),
        algo=_overlay(algo_cls, raw.get("algo")),
    )


_CONFIGS_ROOT = Path(__file__).resolve().parent / "configs"


def yaml_path(config_cls, algo_name: str, task_name: str) -> Path:
    stem = task_name.replace("/", "_")
    if stem.endswith(".yaml"):
        stem = stem[: -len(".yaml")]
    return _CONFIGS_ROOT / "fb" / f"{stem}.yaml"


def _peel_flag(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    rest = []
    value = None
    i = 0
    flag = f"--{name}"
    eq = f"--{name}="
    while i < len(argv):
        a = argv[i]
        if a == flag and i + 1 < len(argv):
            value = argv[i + 1]
            i += 2
            continue
        if a.startswith(eq):
            value = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    return value, rest


def _resolve_yaml(config_cls, config_path: str | None, algo_name: str | None, task_name: str | None) -> Path | None:
    if config_path and (algo_name or task_name):
        raise ValueError("use --config PATH or --algo/--task, not both")
    if config_path:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"--config not found: {path}")
        return path
    if algo_name is None and task_name is None:
        return None
    default = config_cls()
    algo_name = algo_name or default.algo.name
    task_name = task_name or default.world.task_name
    path = yaml_path(config_cls, algo_name, task_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"no yaml for algo={algo_name!r} task={task_name!r} (looked at {path})"
        )
    return path


def parse(config_cls, argv: list[str] | None = None):
    """Load yaml then tyro CLI overlay.

    Extra flags (not shown by tyro --help):
      --algo NAME --task NAME   → configs/fb/{task}.yaml
      --config PATH.yaml        → explicit file
      --dump-config PATH.yaml   → write yaml and exit
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path, argv = _peel_flag(argv, "config")
    algo_name, argv = _peel_flag(argv, "algo")
    task_name, argv = _peel_flag(argv, "task")
    dump_path, argv = _peel_flag(argv, "dump-config")

    default = config_cls()
    yaml_file = _resolve_yaml(config_cls, config_path, algo_name, task_name)
    if yaml_file is not None:
        default = from_yaml(config_cls, yaml_file.read_text())

    cfg = tyro.cli(config_cls, default=default, args=argv)
    if dump_path:
        out = Path(dump_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_yaml(cfg))
        raise SystemExit(0)
    return cfg
