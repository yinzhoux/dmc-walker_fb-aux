# FB

dm_control Walker 上的 Forward-Backward（F / B / π）。先在 ExORL proto 上离线拟合，再混入在线 rollout。

## 安装

```bash
cd /Users/yinzhoux/Files/projects/fb
uv sync
```

数据根目录要有 `walker/proto/buffer/*.npz`：

```bash
export EXORL_DATASET_DIR=/path/to/exorl/datasets
```

## 训练

```bash
uv run python train.py --device cuda
```

默认：100 万步只看 proto，再 100 万步在线混合（offline 比例从 1 降到 0.5）。权重写到 `checkpoints/fb_walker_proto_aux.pt`。

常用参数：

```bash
# 只离线、少跑几步看通
uv run python train.py --device cuda --updates 10000 --online-frames 0 --eval-every 5000

# 指定数据和权重
uv run python train.py --device cuda --replay-dir /path/to/exorl/datasets --ckpt checkpoints/fb.pt

# wandb
uv run python train.py --device cuda --wandb online
```