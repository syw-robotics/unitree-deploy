# Go2 Perceptive Locomotion Policy

深度视觉感知的四足运动策略部署配置。

## 概述

该policy使用深度相机观测环境，实现复杂地形的鲁棒行走。深度图以10Hz更新，policy以50Hz推理。

## 配置说明

### 相机参数 (`camera` 块)

- **transform**: 相机相对于base_link的位置和姿态
  - `position`: [x, y, z] 米
  - `quaternion`: [w, x, y, z] 四元数
  
- **intrinsics**: 相机内参
  - `width`, `height`: 深度图分辨率（policy输入尺寸）
  - `fov`: 视场角（度）
  - `near`, `far`: 深度范围（米）

- **preprocessing**: 深度图预处理
  - `clip_range`: 深度值裁剪范围
  - `normalize_mode`: 归一化方式
    - `"clip_scale"`: (depth - near) / (far - near)
    - `"standard"`: z-score标准化
    - `"none"`: 不归一化
  - `fill_invalid`: 无效像素填充值

- **update_rate**: 深度图更新频率（Hz）

### 深度观测 (`depth` observation)

- **history_len**: 历史帧堆叠数（通常为1）
- **params**:
  - `height`, `width`: 必须与camera.intrinsics一致

## 部署

### Sim环境（Mujoco）

1. 在Mujoco XML中添加深度相机：
```xml
<camera name="depth_camera" 
        pos="0.28 0.0 0.05" 
        quat="1.0 0.0 0.0 0.0"
        fovy="87" 
        mode="fixed"/>
```

2. 使用`MujocoDepthCamera`渲染深度图

### Real环境（RealSense）

1. 连接RealSense相机（推荐D435i或D455）
2. 标定相机位置，更新`camera.transform`参数
3. 使用`RealSenseDepthCamera`采集深度图

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
