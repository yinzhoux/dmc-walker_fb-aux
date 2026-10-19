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

`--goal` 可选 `stand` / `forward` / `backward` / `hip-r` / `hip-l` / `all`（默认 `all`，训练中按这些姿态评测）。

## 评测

```bash
uv run python train.py --play --ckpt checkpoints/fb_walker_proto_aux.pt --goal forward
uv run python train.py --play --ckpt checkpoints/fb_walker_proto_aux.pt --goal all --render
```

## 实时控制

macOS 必须用 `mjpython`。窗口打开后往 FIFO 写一行，就会改目标姿态再算 `z = project(B(s_g))`。

```bash
.venv/bin/mjpython play.py --ckpt checkpoints/fb_walker_proto_aux.pt
```

另开一个终端：

```bash
echo forward > /tmp/fb_goal
echo xvel=1.5 > /tmp/fb_goal
echo hipR=1.2 hipL=0 > /tmp/fb_goal
echo forward hipR=0.8 o14=1.2 > /tmp/fb_goal
echo reset > /tmp/fb_goal
```

命名目标：`stand` `forward` `backward` `hip-r` `hip-l`。  
可改字段：`pitch` `xvel` `height` `rootz` `hipR` `hipL` `kneeR` `kneeL` `ankleR` `ankleL`，以及观测 `o14=1.3`。  
命令：`reset` `quit` `help`。
