# Install
整体目录
```
${workspaceFolder}
├── stubs
├── unitree-deploy
└── unitree_sdk2_python
```
```
cd unitree-deploy
uv sync
source .venv/bin/activate
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd ../unitree_sdk2_python
uv pip install -e .
```

# 启动
真机控制：
`python controller.py --mode real --net <跑策略的网卡名> --ckpt ckpt/g1/loco_flat`

仿真控制：
`python sim_bridge.py --robot g1`
`python controller.py --mode sim --ckpt ckpt/g1/loco_flat`

状态可视化：
`python visualizer.py --mode sim --robot g1`

## 多机器人模型
MuJoCo 模型按机器人分目录放置：
```
unitree-deploy/robot_model/<robot>/<robot>.xml
```

例如 `--robot g1` 会加载 `robot_model/g1/g1.xml`。如果 XML 文件名不符合这个约定，可以用 `--model-xml <path>` 覆盖：
`python sim_bridge.py --robot g1 --model-xml robot_model/g1/g1.xml`

## Checkpoint 目录
策略和控制配置按机器人、策略名分目录放置：
```
unitree-deploy/ckpt/<robot>/<policy>/
├── controller.yaml
├── policy.yaml
├── policy.onnx
└── exported-deploy.yaml
```

当前 G1 平地策略位于 `ckpt/g1/loco_flat/`。

`controller.py` 不直接依赖 MuJoCo XML，关节数量和顺序从 `--ckpt` 目录里的 `controller.yaml` / `policy.yaml` 推导。新机器人或新策略需要准备对应 checkpoint 目录，并在 `controller.yaml` 中维护 `robot`、`real_joint_names`、`mujoco_joint_names`、`isaac_joint_names_state` 和增益。

## 控制器状态机
`controller.py` 有 4 个状态：
- `zero_torque_state`
- `move_to_default_qpos`
- `default_qpos_state`
- `run`

切换规则：
- `A` 从零力矩进入默认姿态过渡
- `Start` 从默认姿态进入 `run`
- `X` 返回零力矩

## 仿真按键
`sim_bridge.py` 同时负责遥控器映射和仿真复位：
- `r` 重置仿真到初始状态
- `b` 发送遥控器 `A`
- `m` 发送遥控器 `Start`
- `up` / `down` 调整吊带高度
- `n` 解除吊带
- `w/s/a/d/q/e` 分别控制前后、侧移、转向
- `esc` 退出

## 说明
- `controller.py` 只负责状态订阅和控制输出。
- `sim_bridge.py` 负责仿真状态发布和键盘输入。
- `visualizer.py` 负责纯可视化。

<video controls src="可视化.webm" title="Title"></video>

## IDE中智能显示mujoco、pyrealsense等（C编写的py接口）的子类
```
cd ${workspaceFolder}
uv pip install mypy
uv run stubgen -m mujoco -o stubs
uv run stubgen -p mujoco -o stubs
```
