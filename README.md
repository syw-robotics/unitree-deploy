# Unitree Deploy

这个仓库用于在真机或 MuJoCo 仿真中运行 Unitree 策略，并提供状态可视化。

核心脚本：
- `controller.py`: 订阅 `LowState`，运行 policy，发布 `LowCmd`
- `sim_bridge.py`: MuJoCo 仿真和 Unitree DDS topic 桥接
- `visualizer.py`: 订阅状态并用 viser 可视化 MuJoCo 模型

## 安装

推荐目录结构：
```text
${workspaceFolder}
├── stubs
├── unitree-deploy
└── unitree_sdk2_python
```

安装依赖：
```bash
cd unitree-deploy
uv sync
source .venv/bin/activate

cd ..
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
uv pip install -e .
```

## 启动

仿真控制，先启动仿真桥：
```bash
python sim_bridge.py --robot g1
```

再启动 controller：
```bash
python controller.py --mode sim --ckpt ckpt/g1/loco_flat
```

状态可视化：
```bash
python visualizer.py --mode sim --robot g1
```

真机控制：
```bash
python controller.py --mode real --net <跑策略的网卡名> --ckpt ckpt/g1/loco_flat
```

## 目录约定

机器人 MuJoCo 模型按机器人分目录：
```text
robot_model/<robot>/
├── <robot>.xml
├── visualizer.yaml
└── meshes/
```

地形 MuJoCo 片段统一放在：
```text
robot_model/scene/
├── flat_terrain.xml
└── rough_terrain.xml
```

例如：
```bash
python sim_bridge.py --robot g1
```

会默认加载 G1 机器人本体，并组合默认平地：
```text
robot_model/g1/g1.xml
robot_model/scene/flat_terrain.xml
```

如果 XML 文件名不符合这个约定，可以手动覆盖：
```bash
python sim_bridge.py --robot g1 --model-xml robot_model/g1/g1.xml
```

运行时可以自由选择地形：
```bash
python sim_bridge.py --robot g1 --terrain rough
python sim_bridge.py --robot g1 --terrain robot_model/scene/rough_terrain.xml
```

策略和控制配置按机器人、策略名分目录：
```text
ckpt/<robot>/<policy>/
├── controller.yaml
├── policy.yaml
├── policy.onnx
└── exported-deploy.yaml
```

当前 G1 平地策略位于：
```text
ckpt/g1/loco_flat/
```

`controller.py` 不直接依赖 MuJoCo XML。关节数量、关节顺序和增益都从 `--ckpt` 目录里的 `controller.yaml` / `policy.yaml` 推导。

## 新增机器人

新增一个机器人时，通常需要做三件事：

1. 添加 MuJoCo 模型：
```text
robot_model/<robot>/<robot>.xml
robot_model/<robot>/meshes/
```

2. 添加可视化配置：
```text
robot_model/<robot>/visualizer.yaml
```

最小配置可以只写：
```yaml
lowstate_topic: rt/lowstate
odom_topic: rt/odommodestate
use_odom: true
base_joint: floating_base_joint
enable_cameras: false
state_joint_names:
  - joint_1
  - joint_2
```

如果没有 `visualizer.yaml`，`visualizer.py` 会尝试从 MuJoCo actuator 自动推导关节顺序，并默认关闭相机。

3. 添加策略 checkpoint：
```text
ckpt/<robot>/<policy>/controller.yaml
ckpt/<robot>/<policy>/policy.yaml
ckpt/<robot>/<policy>/<policy>.onnx
```

`controller.yaml` 中需要维护：
- `robot`
- `real_joint_names`
- `mujoco_joint_names`
- `isaac_joint_names_state`
- `kps_real`
- `kds_real`

`policy.yaml` 中需要维护 policy 输入输出、`observations`、action joint、controlled joint、default qpos 和 ONNX 路径。
`observations` 是按顺序拼接的列表，每项包含 `type` 和 `history_len`，可用 type 包括 `command`、`base_angvel`、`projected_gravity`、`joint_pos`、`joint_vel`、`prev_action`。
特殊策略需要新增 observation type 时，继承 `policy.base_policy.BasePolicy` 并扩展 `OBSERVATION_TYPES` 或覆写 `_build_observation()`。

## 新增策略

同一个机器人新增策略时，只需要新增 checkpoint 目录：
```text
ckpt/g1/<new_policy>/
```

然后运行：
```bash
python controller.py --mode sim --ckpt ckpt/g1/<new_policy>
```

## Visualizer 配置

`visualizer.py` 默认读取：
```text
robot_model/<robot>/visualizer.yaml
```

常用字段：
- `lowstate_topic`: 默认 `rt/lowstate`
- `odom_topic`: 默认 `rt/odommodestate`
- `use_odom`: 是否订阅并应用 odom
- `base_joint`: odom 写入的 MuJoCo freejoint 名称
- `initial_base`: 初始 base pose
- `state_joint_names`: `LowState.motor_state` 对应的 MuJoCo joint 顺序
- `enable_cameras`: 是否默认启用 RealSense
- `cameras`: RealSense 相机参数列表

G1 示例：
```yaml
lowstate_topic: rt/lowstate
odom_topic: rt/odommodestate
use_odom: true
base_joint: floating_base_joint
initial_base:
  pos: [0.0, 0.0, 1.0]
  quat: [1.0, 0.0, 0.0, 0.0]

state_joint_names:
  - left_hip_pitch_joint
  - left_hip_roll_joint

enable_cameras: false
cameras:
  - name: d435_head
    pose_camera_name: d435_head
    serial: "140122071098"
    width: 640
    height: 480
    fps: 30
    enable_depth: true
```

默认不会启用 RealSense。需要相机时，可以在 yaml 里设置：
```yaml
enable_cameras: true
```

也可以运行时覆盖：
```bash
python visualizer.py --robot g1 --camera
python visualizer.py --robot g1 --no-camera
```

没有 RealSense 的机器人保持 `enable_cameras: false`，或者不写 `cameras`。

## Sim Bridge

常用参数：
```bash
python sim_bridge.py --robot g1
python sim_bridge.py --robot g1 --terrain rough
python sim_bridge.py --robot g1 --no-band
python sim_bridge.py --robot g1 --band-sites left_gantry_attach_point,right_gantry_attach_point
```

`--no-band` 适合没有吊带 site 的模型。不同机器人如果吊带 site 名不同，用 `--band-sites` 覆盖。

## 控制器状态机

`controller.py` 有 4 个状态：
- `zero_torque_state`
- `move_to_default_qpos`
- `default_qpos_state`
- `run`

切换规则：
- `A`: 从零力矩进入默认姿态过渡
- `Start`: 从默认姿态进入 `run`
- `X`: 返回零力矩

## 仿真按键

`sim_bridge.py` 同时负责遥控器映射和仿真复位：
- `r`: 重置仿真到初始状态，同时发送遥控器 `X`
- `b`: 发送遥控器 `A`
- `m`: 发送遥控器 `Start`
- `up` / `down`: 调整吊带高度
- `n`: 解除吊带
- `w/s/a/d/q/e`: 控制前后、侧移、转向
- `esc`: 退出

## IDE Stub

用于让 IDE 更好识别 MuJoCo、pyrealsense 等 C 扩展接口：
```bash
cd ${workspaceFolder}
uv pip install mypy
uv run stubgen -m mujoco -o stubs
uv run stubgen -p mujoco -o stubs
```
