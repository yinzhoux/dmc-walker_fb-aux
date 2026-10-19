import time
from contextlib import contextmanager

import torch


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


def as_log_dict(info: dict) -> dict:
    return {k: (v.item() if torch.is_tensor(v) else v) for k, v in info.items()}
