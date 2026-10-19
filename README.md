# FB

从 [oorl](../oorl) 抽出的 lite Forward-Backward（F / B / π）。环境用 `uv` 管理。

## 安装

```bash
cd /Users/yinzhoux/Files/projects/fb
uv sync
# dm_control walker / 录像
uv sync --extra dmc
```

## 训练

Gym Walker2d（在线采集）：

```bash
uv run python main_fb.py --task Walker2d-v5
uv run python main_fb.py --config configs/fb/Walker2d-v5.yaml
```

ExORL proto 离线 + 在线混合（需要 `dm_control` 和 proto npz）：

```bash
uv run python main_fb_offline.py --device cuda
```

FB-CPR（专家序列来自 ExORL，不是 TD replay）：

```bash
uv run python main_fb_cpr.py --task walker_walk --expl-agent proto --replay-dir /path/to/exorl/datasets
```

## 回放

```bash
uv run python play_fb.py --ckpt checkpoints/fb_Walker2d-v5.pt --prompt zr --render human
# macOS 实时 FIFO
.venv/bin/mjpython play_fb_live.py --ckpt checkpoints/fb_walker_proto.pt
```

## 布局

| 路径 | 内容 |
| --- | --- |
| `algos/fb.py` | Agent、FB / CPR / aux 损失 |
| `algos/modules/fb_nets.py` | F、B、π、BallNorm |
| `algos/fb_eval.py` | Walker 技能探测 |
| `main_fb.py` | Gym 在线训练 |
| `main_fb_offline.py` | proto 两阶段 |
| `main_fb_cpr.py` | FB-CPR |
| `play_fb.py` / `play_fb_live.py` | 回放 |
| `configs/fb/` | Walker2d yaml |
