# Go2 Perceptive Locomotion Policy

深度视觉感知的四足运动策略部署配置。

## 概述

该policy使用深度相机观测环境，实现复杂地形的鲁棒行走。深度图以10Hz更新，policy以50Hz推理。

## 配置说明

### 深度相机参数 (`sensor_depth_camera.yaml`)

- **source**: 相机数据源，仿真中使用 `"mujoco"`
- **name**: MuJoCo camera 名称，默认 `"depth_camera"`
- **attach_body**: 相机挂载的 MuJoCo body，例如 `"base_link"`
- **transform**: 相机相对于base_link的位置和姿态
  - `position`: [x, y, z] 米
  - `rpy`: [roll, pitch, yaw] 欧拉角，默认弧度
  - `degrees`: 设为 `true` 时 `rpy` 使用角度
  
- **intrinsics**: 相机内参
  - `width`, `height`: 深度图分辨率（policy输入尺寸）
  - `fovy`: 垂直视场角（度）。MuJoCo camera 原生使用 `fovy`，水平视场角由 `width / height` 自动推导
  - `near`, `far`: 深度范围（米）

- **preprocessing**: 深度图预处理
  - `crop`: 裁剪像素数，按 `top`, `bottom`, `left`, `right` 配置；裁剪发生在 clip/normalize 之前
  - `clip_range`: 深度值裁剪范围
  - `normalize_mode`: 归一化方式
    - `"clip_scale"`: (depth - near) / (far - near)
    - `"standard"`: z-score标准化
    - `"none"`: 不归一化
  - `fill_invalid`: 无效像素填充值

- **update_rate**: 深度图更新频率（Hz）
- **preview**: 可选的 sim bridge 预览窗口配置
  - `enabled`: 是否显示深度图，默认 `true`
  - `scale`: 预览窗口放大倍数，默认 `4`
  - `title`: 预览窗口标题

### 深度观测 (`policy.yaml` 的 `depth` observation)

- **history_len**: 历史帧堆叠数（通常为1）
- `height`, `width`: policy 实际输入的深度图尺寸，本例为原始 `87x58`
  减去四边各 2 像素 crop 后的 `83x54`
- `shared_memory_name`: 必须与 sensor yaml 里的相机配置一致

### 高程扫描参数 (`sensor_height_scan.yaml`)

- **source**: 仿真中使用 `"mujoco_raycast"`
- **attach_body**: scan grid 跟随的 MuJoCo body，例如 `"base_link"`
- **grid**: 高程图采样网格，`shape` 为 `[height, width]`
- **ray**: ray 起点高度、最大距离、方向和 geom group mask
- **preprocessing.mode**:
  - `"base_relative_height"`: 命中点高度减去 base 高度
  - `"clearance"`: ray 起点到命中点的距离
  - `"world_height"`: 命中点世界高度
  - `"distance"`: MuJoCo raycast 距离
- **visualization**: MuJoCo viewer 中显示有效 raycast 命中点；仅仿真生效

如果 policy 使用高程图观测，在 `policy.yaml` 的 `observations` 中添加：

```yaml
  - type: height_scan
    history_len: 1
    params:
      height: 15
      width: 11
      shared_memory_name: unitree_height_scan_go2_perceptive_locomotion
```

`height`, `width`, `shared_memory_name` 必须与 sensor yaml 保持一致。

## 部署

### Sim环境（Mujoco）

1. 启动 sim bridge，并传入 sensor 配置。bridge 会读取 sensor yaml 的
   `camera` 块，在运行时生成带深度相机的临时 MuJoCo XML：

```bash
unitree-sim-bridge --robot go2 --terrain rough --sensor ckpt/go2/perceptive_locomotion/sensor_depth_camera.yaml
```

如果不需要深度图窗口，可以加 `--no-depth-preview`。

2. 使用同一个 ckpt 启动 controller：

```bash
unitree-controller --mode sim --robot go2 --ckpt ckpt/go2/perceptive_locomotion/policy.yaml
```

3. `MujocoDepthCamera` 会渲染深度图并写入 sensor yaml 指定的 shared
   memory，`DepthObservation` 会按 `policy.yaml` 中的 depth params 读取。

### Real环境（RealSense）

1. 连接RealSense相机（推荐D435i或D455）
2. 标定相机位置，更新`camera.transform`参数
3. 使用`RealSenseDepthCamera`采集深度图

高程扫描的真机 producer 目前预留为空；policy 侧可以继续通过
`HeightScanObservation` 读取同名 shared memory，后续由真实高程图模块写入。

## 实现架构

```
Policy (50Hz)
    ↓ 读取最新深度
DepthObservationBuffer (共享缓存)
    ↑ 异步更新 (10Hz)
DepthCamera (Mujoco/RealSense)
```

## 注意事项

1. **延迟**: 深度图最多有100ms延迟（10Hz更新）
2. **线程安全**: DepthObservationBuffer使用锁保护
3. **性能**: RealSense推理可能需要GPU加速（CUDA provider）
4. **标定**: Real部署需要精确标定相机位置
