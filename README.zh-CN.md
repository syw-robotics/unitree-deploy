# Unitree Deploy

[English](README.md)

面向 Unitree 机器人的轻量级部署代码库。

本项目把 ONNX 策略、观测构造、DDS 通信、MuJoCo 模型和控制器状态机整理成一套可复用流程，适合先在仿真中验证策略，再用相同控制路径部署到 Unitree 真机。

## ✨ 特性

- 同一套 controller runtime 支持 `sim` 和 `real` 模式。
- 通过可配置的 `yaml` 文件加载 ONNX policy。
- 支持多 policy 切换。
- 提供模板生成器，方便自定义部署需求，例如自定义 observation。

## 📦 项目结构

```text
src/unitree_deploy/
├── cli/              # 命令行入口
├── runtime/          # controller 与 sim bridge 主循环
├── policy/           # ONNX policy wrapper
├── obs/              # observation term
├── visualization/    # 机器人状态可视化
├── utils/            # 通用工具
└── robot_model/      # 内置 MuJoCo 机器人与地形资源

ckpt/
├── g1/               # G1 示例策略与 multi-ckpt 配置
└── go2/              # Go2 示例策略
```

## 🚀 安装

```bash
uv sync
source .venv/bin/activate
```

如果需要 viser 可视化依赖，用于实时机器人状态可视化：

```bash
uv sync --extra viewer
```

`unitree_sdk2_python` 需要单独安装到同一个 Python 环境中。

## 🕹️ 快速开始

启动 MuJoCo 到 DDS 的仿真桥：

```bash
unitree-sim-bridge --robot g1
```

如果需要仿真传感器，启动 sim bridge 时传入 sensor 配置，运行时会注入
MuJoCo 相机：

```bash
unitree-sim-bridge --robot go2 --terrain rough --sensor ckpt/go2/perceptive_locomotion/sensor_depth_camera.yaml
```

对于仿真深度相机，bridge 会同时打开实时深度图预览窗口；不需要时可以加
`--no-depth-preview`。

启动仿真 controller：

```bash
unitree-controller --mode sim --ckpt ckpt/g1/vanilla_ppo_flat/policy.yaml
```

使用 multi-policy 配置：

```bash
unitree-controller --mode sim --multi-ckpt ckpt/g1/multi_ckpt.yaml
```

启动 viser 可视化：

```bash
unitree-visualizer --mode sim --robot g1
```

真机运行时需要显式指定 DDS 网卡：

```bash
unitree-controller --mode real --net <interface> --ckpt ckpt/g1/vanilla_ppo_flat/policy.yaml
```

## 🧩 部署目录

一个 policy deployment 目录通常包含：

```text
ckpt/<robot>/<policy>/
├── policy.yaml             # policy、observation、关节顺序和增益配置
├── policy.onnx             # 导出的 ONNX，也可以在 YAML 中改成其他相对路径
├── custom_observations.py  # [可选] 自定义 observation
└── custom_policy.py        # [可选] 自定义推理或 action 后处理
```

交互式生成模板：

```bash
unitree-plugin-template
```

也可以脚本化生成：

```bash
unitree-plugin-template ckpt --robot g1 --name my_policy
```

`policy.yaml` 中最关键的字段：

```yaml
policy_path: "policy.onnx"
obs_joint_order: [...]
action_joint_order: [...]
sdk_joint_order: [...]
```

`obs_joint_order`、`action_joint_order` 和 `sdk_joint_order` 会按关节名自动推导重排索引，不需要手写 index。

## 🔁 多策略切换

需要在运行时切换多个 policy 时，使用 `--multi-ckpt`：

```yaml
default: vanilla_ppo_flat

ckpts:
  vanilla_ppo_flat: "./vanilla_ppo_flat"
  unitree_rl_lab_flat: "./unitree_rl_lab_flat"

switch:
  enabled: true
  button: B
  order: [vanilla_ppo_flat, unitree_rl_lab_flat]
  only_when: [run_policy]
  on_switch: null
```

仿真中按 `b` 切换 policy；真机遥控器默认使用 `B`。同一个 manifest 下的 policy 必须共享 `sdk_joint_order` 和 `policy_step_dt`，但 observation、action、gain 和 ONNX 文件可以不同。

## ⌨️ 默认控制

默认状态机使用这些遥控器按键：

- `A`: 移动到默认关节位置。
- `Start`: 运行当前 policy。
- `B`: 在启用 multi-ckpt 时切换 policy。
- `X`: 回到 damping。

仿真中的键盘映射是：`enter` 对应 `A`，`\` 对应 `Start`，`b` 对应 `B`，`x` 对应 `X`。
