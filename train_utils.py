import time
import types
from contextlib import contextmanager
from dataclasses import fields, is_dataclass

import torch
import wandb


class TimeStats:
    """Accumulate wall-clock spans; flush averages into a wandb-style dict."""

    def __init__(self, sync_cuda: bool = False):
        self.sync_cuda = sync_cuda
        self._t = {}
        self._n = {}

    def _sync(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def span(self, name: str):
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t0
            self._t[name] = self._t.get(name, 0.0) + dt
            self._n[name] = self._n.get(name, 0) + 1

    def flush(self, prefix: str = "time/") -> dict:
        out = {}
        for k, t in self._t.items():
            out[f"{prefix}{k}"] = t / max(self._n[k], 1)
        self._t.clear()
        self._n.clear()
        return out


def scalar(x):
    return x.detach() if torch.is_tensor(x) else x


def add_prefix(infos: dict, prefix: str):
    """Prefix keys. Use trailing '/' (e.g. 'explore/') for nested wandb groups."""
    return {f"{prefix}{k}": v for k, v in infos.items()}


def remap_eval_prefix(infos: dict, new_root: str) -> dict:
    """Rename `eval/...` keys to `{new_root}/...` (e.g. eval_explore, eval_retrain)."""
    out = {}
    for k, v in infos.items():
        if k.startswith("eval/"):
            out[f"{new_root}/{k[len('eval/'):]}"] = v
        else:
            out[k] = v
    return out


def as_log_dict(info: dict) -> dict:
    return {k: (v.item() if torch.is_tensor(v) else v) for k, v in info.items()}


def _wandb_value(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if is_dataclass(v):
        return as_wandb_config(v)
    if isinstance(v, dict):
        return {k: _wandb_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_wandb_value(x) for x in v]
    name = getattr(v, "__name__", None)
    if isinstance(name, str):
        return name
    return str(v)


def as_wandb_config(cfg) -> dict:
    """Serialize a (possibly nested) dataclass, including class-level attrs like activation."""
    if not is_dataclass(cfg):
        return {}
    out = {}
    field_names = {f.name for f in fields(cfg)}
    for f in fields(cfg):
        out[f.name] = _wandb_value(getattr(cfg, f.name))
    for k, v in vars(type(cfg)).items():
        if k.startswith("_") or k in field_names:
            continue
        if isinstance(v, (classmethod, staticmethod, property, types.FunctionType)):
            continue
        out[k] = _wandb_value(v)
    return out


def init_wandb(*, name: str, config):
    run = wandb.init(
        project=config.wandb.project,
        name=name,
        config=as_wandb_config(config),
        save_code=True,
        group=config.wandb.group,
        mode=config.wandb.mode,
    )
    run.log_code(
        ".",
        include_fn=lambda path, root=None: str(path).endswith(".py"),
        exclude_fn=lambda path, root=None: any(
            part in str(path)
            for part in (".venv", "__pycache__", ".ogbench", "/wandb/", "wandb/")
        ),
    )
    return run