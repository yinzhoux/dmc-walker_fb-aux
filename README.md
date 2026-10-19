# FB

Lite Forward-Backward（F / B / π）。只覆盖 **ExORL proto 离线训练** 和 **dm_control Walker**。

## 安装

```bash
cd /Users/yinzhoux/Files/projects/fb
uv sync
```

## 训练

两阶段：先在 ExORL proto 上拟合 F / B / π，再在同一个 dm_control walker 上混入在线 rollout。

```bash
export EXORL_DATASET_DIR=/path/to/exorl/datasets
uv run python train.py --device cuda
```

`main_fb_offline.py` 仍指向同一个入口。

## 回放

```bash
uv run python train.py --play --ckpt checkpoints/fb_walker_proto.pt --goal forward
# macOS 实时 FIFO
.venv/bin/mjpython play.py --ckpt checkpoints/fb_walker_proto.pt
```

## 布局

| 路径 | 内容 |
| --- | --- |
| `train.py` / `play.py` | CLI，只做参数和调度 |
| `fb/agent.py` | F / B / π 和 aux Q |
| `fb/nets.py` | 网络 |
| `fb/walker.py` | dm_control walker、命名姿态 |
| `fb/proto.py` | ExORL npz 加载 |
| `fb/offline.py` | proto / 在线两段训练循环 |
| `fb/eval.py` | goal rollout 打分 |
| `fb/plant.py` / `fb/live.py` | FIFO 目标、ghost、viewer |
| `fb/buffer.py` | 在线 replay 与 mixed sample |
