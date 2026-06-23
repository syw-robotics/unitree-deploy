# Unitree Deploy

这个项目用于在 Unitree 机器人或 MuJoCo 仿真中部署策略，目前还处在整理和重构阶段，接口和目录结构都可能继续调整。

## 当前结构

核心代码在 `src/unitree_deploy/`：

```text
src/unitree_deploy/
├── cli/              # 命令行入口
├── runtime/          # controller / sim bridge 运行逻辑
├── policy/           # policy wrapper
├── obs/              # observation 定义
├── visualization/    # visualizer 和 scene 配置
├── utils/            # 共享工具
└── robot_model/      # 内置 MuJoCo 模型资源
```

策略部署文件暂时仍放在 `ckpt/`：

```text
ckpt/<robot>/<policy>/
├── controller.yaml
├── policy.yaml
└── *.onnx
```

## 安装

```bash
uv sync
source .venv/bin/activate
```

如果需要浏览器端 viewer / viser 可视化：

```bash
uv sync --extra viewer
```

`unitree_sdk2_python` 目前仍需要在外部单独安装到环境中。

## 入口

仿真桥：

```bash
unitree-sim-bridge --robot g1
```

Controller：

```bash
unitree-controller --mode sim --ckpt ckpt/g1/loco_flat
```

也可以用一个 multi-ckpt YAML 管理多个可切换 policy：

```bash
unitree-controller --mode sim --multi-ckpt ckpt/g1/multi_ckpt.yaml
```

Visualizer：

```bash
unitree-visualizer --mode sim --robot g1
```

真机运行时需要指定网卡：

```bash
unitree-controller --mode real --net <interface> --ckpt ckpt/g1/loco_flat
```

## 插件

如果需要自定义 policy 或 observation，可以先生成一个模板：

```bash
unitree-plugin-template
```

命令会交互式询问 base path、项目名和机器人类型，并生成类似 `ckpt/g1/my_policy/` 的 deployment 目录。也可以脚本化：

```bash
unitree-plugin-template ckpt --robot g1 --name my_policy
```

一个 deployment 目录通常需要：

```text
ckpt/<robot>/<policy>/
├── controller.yaml       # controller topic / PD 参数
├── policy.yaml           # ONNX、action、joint、observation 配置
├── policy.onnx           # 导出的 ONNX；文件名可在 policy.yaml 里改
├── custom_observations.py  # 可选
└── custom_policy.py        # 可选
```

ONNX 推荐直接放在 deployment 目录下，例如 `policy.onnx`，然后在 `policy.yaml` 里写：

```yaml
policy_path: "policy.onnx"
obs_joint_order: [...]
action_joint_order: [...]
sdk_joint_order: [...]
```

如果 ONNX 放在子目录，也可以写相对路径，例如：

```yaml
policy_path: "models/policy.onnx"
```

`obs_joint_order` 是 observation、`default_qpos` 和 policy 返回 `target_q` 使用的关节顺序，`action_joint_order` 是 ONNX action 输出和 `action_scale` 的顺序，`sdk_joint_order` 是 Unitree SDK `LowState/LowCmd` 的关节顺序。obs/action/sdk 之间的 reorder index 会由关节名自动推导，不需要手写。

`policy.yaml` 同时负责 observation 排布和自定义 observation 类型：

```yaml
observation_types:
  custom_obs: "custom_observations:CustomObservation"

observations:
  - type: custom_obs
    history_len: 1
    scale: 1.0
    clip: [-10.0, 10.0]
```

`scale` 是 observation term 的通用标量，会作用于当前 term 的所有维度；`clip` 是 observation term 的通用裁剪范围，会在 scale 后、写入 history buffer 前生效，可以写成 `[min, max]`，也可以按 term 维度写成 `[[min, max], ...]`。自定义 observation 的其他额外参数可以写在 `params` 下，会传给 observation 构造函数。旧的 `observation_modules` 方式仍兼容，但推荐直接用 `observation_types` 显式声明。只有需要改 action 后处理或推理逻辑时，才需要在 `policy.yaml` 里配置 `policy_class`。

## Policy 切换

单个 policy 仍然直接用 `--ckpt`。如果需要在同一个 controller 进程里切换多个 policy，新建一个 `multi_ckpt.yaml`：

```yaml
default: flat

ckpts:
  flat: "vanilla_ppo_flat"
  lab_flat: "unitree_rl_lab_flat"
  rough: "rough_loco"

switch:
  enabled: true
  button: B
  order: [flat, lab_flat, rough]
  # 只允许 policy 已经在运行时切换；按 B 后按 order 切到下一个 policy。
  only_when: [run_policy]
  # null 表示切换后保持当前 controller 状态；这里会继续留在 run_policy。
  on_switch: null
```

`ckpts` 的值相对 `multi_ckpt.yaml` 所在目录解析，可以直接写 ckpt 目录，也可以写 `{policy_yaml: ".../policy.yaml"}`。切换时 controller 会重置目标 policy；上面的配置表示当前正在 `run_policy` 时，按 B 会直接从 policy A 切到 policy B 并继续运行。`on_switch` 可以设成 `move_to_default_qpos`，用于切换后先回到新 policy 的默认姿态，再按 Start 运行。

为了保持运行时简单和安全，同一个 `multi_ckpt.yaml` 下的 ckpt 必须使用相同的 `sdk_joint_order` 和 `policy_step_dt`；`obs_joint_order`、`action_joint_order`、ONNX、observation 配置和 gain 可以各自不同。

仿真中默认按键是 `b` 切换 policy；真机遥控器默认使用 `B`。


## TODO

- [ ] Max torque clip
- [ ] G1 Motion Tracking Policy Support
- [ ] VR teleop device port, for realtime teleoperation
