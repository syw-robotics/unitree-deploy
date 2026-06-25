# Go2 Perceptive Locomotion Policy 部署设计方案

## 概述

支持深度视觉的policy部署需要解决以下核心问题：
1. **异步观测更新**：深度图更新频率(~10Hz)与policy推理频率(~50Hz)不一致
2. **相机参数配置**：sim和real环境中的相机配置统一管理
3. **相机源抽象**：Mujoco深度相机 vs RealSense相机的统一接口
4. **深度观测处理**：深度图预处理、编码、历史堆叠

## 架构组件

```
┌─────────────────────────────────────────────────────────────┐
│                    Policy (50Hz)                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ObservationContext + DepthObservation                │   │
│  │  - 本体感知: IMU, joint states                       │   │
│  │  - 视觉: 缓存的深度图 (异步更新)                     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │ 每次policy step获取最新缓存的depth
                           │
┌─────────────────────────────────────────────────────────────┐
│              DepthObservationBuffer (共享状态)              │
│  - 存储最新的深度图                                         │
│  - 线程安全访问                                              │
│  - 时间戳管理                                                │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │ 异步更新 (~10Hz)
                           │
┌─────────────────────────────────────────────────────────────┐
│                  DepthCameraSource                          │
│                                                              │
│  ┌─────────────────┐              ┌──────────────────┐     │
│  │ MujocoDepthCam  │              │ RealSenseDepthCam│     │
│  │  - 读取mjData   │              │  - pyrealsense2  │     │
│  │  - 相机矩阵配置 │              │  - USB设备       │     │
│  └─────────────────┘              └──────────────────┘     │
│           Sim环境                        Real环境           │
└─────────────────────────────────────────────────────────────┘
```

## 关键设计决策

### 1. 异步观测更新策略

**方案A：Latest-Value缓存（推荐）**
- Policy每次推理时读取最新的深度图缓存
- 深度相机独立线程以10Hz更新缓存
- 优点：简单、无阻塞、policy运行流畅
- 缺点：可能使用"旧"的深度图（最多100ms延迟）

**方案B：基于时间戳的插值**
- 记录每帧深度图的时间戳
- Policy根据当前时间插值或选择最近帧
- 优点：时间同步更精确
- 缺点：增加复杂度，插值深度图意义不大

**推荐：方案A**，因为训练时的深度观测也是有延迟的，模拟训练条件。

### 2. DepthObservation实现

继承`ObservationBase`，添加以下特性：
- 接收预处理后的深度图（已resize、归一化）
- 支持历史堆叠（如果policy需要时序信息）
- 可配置的预处理参数（crop、resize、normalization）

### 3. 相机参数配置（policy.yaml扩展）

```yaml
# 新增camera配置块
camera:
  # 相机类型
  type: depth  # depth | rgb | rgbd
  
  # 相机相对于base_link的位置和姿态 (x, y, z, qw, qx, qy, qz)
  transform:
    position: [0.3, 0.0, 0.05]  # 相机在base前方30cm
    quaternion: [1.0, 0.0, 0.0, 0.0]  # 无旋转
  
  # 相机内参
  intrinsics:
    width: 58  # policy输入的宽度
    height: 87  # policy输入的高度
    fov: 87.0  # 视场角（度）
    near: 0.1  # 近裁剪面（米）
    far: 3.0   # 远裁剪面（米）
  
  # 深度图预处理
  preprocessing:
    clip_range: [0.1, 3.0]  # 裁剪范围（米）
    normalize_mode: "clip_scale"  # "clip_scale" | "standard" | "none"
    fill_invalid: 3.0  # 无效像素填充值
    
  # 更新频率
  update_rate: 10.0  # Hz
```

### 4. Sim环境相机配置（Mujoco）

在Mujoco XML中配置深度相机：
```xml
<camera name="depth_camera" 
        pos="0.3 0.0 0.05" 
        quat="1.0 0.0 0.0 0.0"
        fovy="87" 
        mode="fixed"/>
```

`sim_bridge.py`需要：
- 读取policy.yaml的camera配置
- 在Mujoco中渲染深度图
- 应用预处理并更新DepthObservationBuffer

### 5. Real环境相机配置（RealSense）

RealSense设置：
- 使用D435i或D455等型号
- 配置对齐到深度流
- 应用相同的预处理pipeline
- 相机标定：确保transform参数准确

## 文件结构

```
ckpt/go2/perceptive_locomotion/
├── policy.yaml              # 扩展的配置，包含camera块
├── policy_xxxx.onnx         # ONNX模型文件
└── README.md                # 说明文档

src/unitree_deploy/obs/
├── observation.py           # 现有
└── depth_observation.py     # 新增：DepthObservation类

src/unitree_deploy/runtime/
├── depth_camera.py          # 新增：相机源抽象
│   ├── DepthCameraBase
│   ├── MujocoDepthCamera
│   └── RealSenseDepthCamera
└── depth_buffer.py          # 新增：线程安全的深度图缓存
```

## 实现优先级

1. **Phase 1：基础架构**
   - DepthObservationBuffer（共享缓存）
   - DepthObservation类
   - policy.yaml的camera配置解析

2. **Phase 2：Sim支持**
   - MujocoDepthCamera实现
   - sim_bridge.py集成

3. **Phase 3：Real支持**
   - RealSenseDepthCamera实现
   - Real runtime集成

4. **Phase 4：测试与优化**
   - 延迟测量
   - 性能优化
   - 可视化工具
