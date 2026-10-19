import numpy as np
import torch

TRANSITION_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "terminated",
    "truncated",
)


class Buffer:
    """Circular replay. Keys: observations, actions, next_observations, rewards, terminated, truncated."""

    def __init__(
        self,
        observation_example: np.ndarray,
        action_example: np.ndarray,
        size: int = 1000000,
        buffer_device: str = "cpu",
        train_device: str = "cuda",
        z_dim: int | None = None,
        store_prev_action: bool = False,
    ):
        self.buffer_device = buffer_device
        self.train_device = train_device

        def zeros_from_example(example, size):
            example = np.asarray(example)
            return torch.zeros((size, *example.shape), dtype=torch.float32, device=self.buffer_device)

        self._observations = zeros_from_example(observation_example, size)
        self._next_observations = zeros_from_example(observation_example, size)
        self._actions = zeros_from_example(action_example, size)
        self._rewards = torch.zeros((size, 1), dtype=torch.float32, device=self.buffer_device)
        self._terminated = torch.zeros((size, 1), dtype=torch.float32, device=self.buffer_device)
        self._truncated = torch.zeros((size, 1), dtype=torch.float32, device=self.buffer_device)
        self._z = (
            None
            if z_dim is None
            else torch.zeros((size, z_dim), dtype=torch.float32, device=self.buffer_device)
        )
        self._prev_actions = (
            zeros_from_example(action_example, size) if store_prev_action else None
        )

        self._ptr = 0
        self._curr_size = 0
        self._size = size
        self._frozen = False
        self._group_names: tuple[str, ...] | None = None
        self._reward_groups = None

    @classmethod
    def from_dataset(cls, dataset: dict, *, train_device: str, buffer_device: str = "cpu"):
        n = int(dataset["observations"].shape[0])
        buf = cls(
            dataset["observations"][0],
            dataset["actions"][0],
            size=n,
            buffer_device=buffer_device,
            train_device=train_device,
        )
        buf._load_all(dataset)
        buf._frozen = True
        return buf

    @property
    def size(self) -> int:
        return self._curr_size

    @property
    def observation_example(self):
        return self._observations[0].detach().cpu().numpy()

    @property
    def action_example(self):
        return self._actions[0].detach().cpu().numpy()

    def __len__(self):
        return self._curr_size

    def _put_row(self, dst, value, idx):
        t = torch.as_tensor(value, dtype=torch.float32, device=self.buffer_device)
        dst[idx] = t.reshape(dst[idx].shape)

    def _load_all(self, dataset: dict):
        n = int(dataset["observations"].shape[0])
        if n > self._size:
            raise ValueError(f"dataset length {n} exceeds buffer size {self._size}")

        def put(dst, src):
            t = torch.as_tensor(np.asarray(src, dtype=np.float32), device=self.buffer_device)
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            dst[:n] = t

        put(self._observations, dataset["observations"])
        put(self._next_observations, dataset["next_observations"])
        put(self._actions, dataset["actions"])
        put(self._rewards, dataset["rewards"])
        put(self._terminated, dataset["terminated"])
        put(self._truncated, dataset["truncated"])
        if self._prev_actions is not None:
            if "prev_actions" not in dataset:
                raise ValueError("this buffer stores prev_actions; dataset has none")
            put(self._prev_actions, dataset["prev_actions"])
        self._curr_size = n
        self._ptr = 0 if n == self._size else n

    def append(
        self,
        observation,
        action,
        next_observation,
        reward,
        terminated,
        truncated,
        z=None,
        reward_groups: dict | None = None,
        prev_action=None,
    ):
        if self._frozen:
            raise RuntimeError("Cannot append to a frozen dataset buffer")
        i = self._ptr
        self._put_row(self._observations, observation, i)
        self._put_row(self._next_observations, next_observation, i)
        self._put_row(self._actions, action, i)
        self._put_row(self._rewards, reward, i)
        self._put_row(self._terminated, terminated, i)
        self._put_row(self._truncated, truncated, i)
        if self._z is not None:
            if z is None:
                raise ValueError("this buffer stores z; append() needs z")
            self._put_row(self._z, z, i)
        if self._prev_actions is not None:
            if prev_action is None:
                raise ValueError("this buffer stores prev_actions; append() needs prev_action")
            self._put_row(self._prev_actions, prev_action, i)
        if reward_groups:
            self._store_groups(reward_groups, i)
        self._ptr = (self._ptr + 1) % self._size
        self._curr_size = min(self._size, self._curr_size + 1)

    def sample(self, batch_size: int = 256):
        if self._curr_size == 0:
            raise RuntimeError("Cannot sample from an empty buffer")
        idx = torch.randint(low=0, high=self._curr_size, size=(batch_size,), device=self.buffer_device)
        terminated = self._terminated[idx].to(self.train_device)
        truncated = self._truncated[idx].to(self.train_device)
        out = {
            "observations": self._observations[idx].to(self.train_device),
            "actions": self._actions[idx].to(self.train_device),
            "next_observations": self._next_observations[idx].to(self.train_device),
            "rewards": self._rewards[idx].to(self.train_device),
            "terminated": terminated,
            "truncated": truncated,
            "done": (terminated.bool() | truncated.bool()).float(),
        }
        if self._z is not None:
            out["z"] = self._z[idx].to(self.train_device)
        if self._prev_actions is not None:
            out["prev_actions"] = self._prev_actions[idx].to(self.train_device)
        if self._reward_groups is not None:
            cols = self._reward_groups[idx].to(self.train_device)
            out["reward_groups"] = {
                name: cols[:, i : i + 1] for i, name in enumerate(self._group_names)
            }
        return out

    def _store_groups(self, groups: dict, idx: int) -> None:
        if self._group_names is None:
            self._group_names = tuple(groups)
            self._reward_groups = torch.zeros(
                (self._size, len(self._group_names)),
                dtype=torch.float32,
                device=self.buffer_device,
            )
        missing = [name for name in self._group_names if name not in groups]
        if missing:
            raise ValueError(f"reward_groups missing {missing}")
        row = torch.tensor(
            [float(groups[name]) for name in self._group_names],
            dtype=torch.float32,
            device=self.buffer_device,
        )
        self._reward_groups[idx] = row


def resolve_offline_ratio(
    mode: str,
    *,
    offline_size: int,
    online_size: int,
    offline_ratio: float = 0.5,
    t: int = 0,
    T: int = 1,
    ratio_start: float = 1.0,
    ratio_end: float = 0.5,
) -> float:
    """Compute offline fraction for one mixed batch.

    Modes:
      fixed:        use `offline_ratio`
      natural:      |D_off| / (|D_off| + |D_on|)  (1.0 if online empty)
      online_only:  0.0
      linear_decay: lerp(ratio_start, ratio_end) by online progress t/T
    """
    mode = (mode or "fixed").lower()
    if mode == "fixed":
        ratio = float(offline_ratio)
    elif mode == "natural":
        total = int(offline_size) + int(online_size)
        ratio = 1.0 if total == 0 or online_size <= 0 else offline_size / total
    elif mode == "online_only":
        ratio = 0.0
    elif mode == "linear_decay":
        if T <= 1:
            frac = 1.0
        else:
            frac = min(max(float(t) / float(T - 1), 0.0), 1.0)
        ratio = float(ratio_start) + (float(ratio_end) - float(ratio_start)) * frac
    else:
        raise ValueError(
            f"Unknown data_schedule {mode!r}. "
            "Use 'fixed', 'natural', 'online_only', or 'linear_decay'."
        )
    return min(max(ratio, 0.0), 1.0)


def schedule_offline_ratio(
    train_cfg,
    offline: Buffer,
    online: Buffer,
    *,
    t: int,
    T: int | None = None,
    for_retrain: bool = False,
) -> float:
    """Resolve ratio from O2OTrainCfg fields.

    During online, pass frame index `t` and total frames `T` (defaults to online_frames).
    For final retrain, `for_retrain=True` freezes linear_decay at `offline_ratio_end`.
    """
    mode = getattr(train_cfg, "data_schedule", "fixed")
    if for_retrain and str(mode).lower() == "linear_decay":
        mode = "fixed"
        offline_ratio = float(train_cfg.offline_ratio_end)
    else:
        offline_ratio = float(train_cfg.offline_ratio)
    total = int(T) if T is not None else int(train_cfg.online_frames)
    return resolve_offline_ratio(
        mode,
        offline_size=offline.size,
        online_size=online.size,
        offline_ratio=offline_ratio,
        t=t,
        T=total,
        ratio_start=float(getattr(train_cfg, "offline_ratio_start", 1.0)),
        ratio_end=float(getattr(train_cfg, "offline_ratio_end", 0.5)),
    )


def sample_mixed(offline: Buffer, online: Buffer, batch_size: int, offline_ratio: float):
    """Draw `offline_ratio` of the batch from `offline`, the rest from `online`."""
    ratio = min(max(float(offline_ratio), 0.0), 1.0)
    n_off = int(round(batch_size * ratio))
    n_on = batch_size - n_off

    if n_on > 0 and online.size < n_on:
        if n_off == 0:
            return None
        return offline.sample(batch_size)

    batches = []
    if n_off > 0:
        batches.append(offline.sample(n_off))
    if n_on > 0:
        batches.append(online.sample(n_on))
    if not batches:
        return None
    if len(batches) == 1:
        return batches[0]

    out = {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}
    perm = torch.randperm(out["observations"].shape[0], device=out["observations"].device)
    return {k: v[perm] for k, v in out.items()}
