import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import viser
from unitree_deploy.config.defaults import DEFAULT_MODE, LOWSTATE_TOPIC, ODOM_TOPIC
from unitree_deploy.robot_model.robot_config import DEFAULT_ROBOT, DEFAULT_TERRAIN, RobotModel, load_robot_model
from unitree_deploy.visualization.scene_config import RealSenseCameraConfig, StandaloneMujocoScene
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.thread import RecurrentThread

from ..utils.yaml_utils import load_yaml


def log(message: str) -> None:
    print(f"[visualizer] {message}", flush=True)


@dataclass(frozen=True)
class RuntimeConfig:
    robot: RobotModel
    visualizer_yaml: Path | None
    mode: str
    net: str | None
    camera: bool | None


class RobotStateVisualizer:
    """Render LowState/Odom DDS messages on top of a MuJoCo model in Viser.

    Data flow:
      LowState motor_state -> MuJoCo qpos joints
      optional Odom -> MuJoCo freejoint base pose
      MuJoCo forward kinematics -> Viser scene
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.cfg = self.load_config()
        self.lowstate_topic = self.cfg.get("lowstate_topic", LOWSTATE_TOPIC)
        self.odom_topic = self.cfg.get("odom_topic", ODOM_TOPIC)
        self.use_odom = bool(self.cfg.get("use_odom", True))
        self.show_camera_frustums = bool(self.cfg.get("show_camera_frustums", True))

        self.model = None
        self.data = None
        self.lock = threading.Lock()
        self.joint_map: list[tuple[int, int]] = []
        self.base_qpos_adr: int | None = None

        self.lowstate_subscriber = None
        self.odomstate_subscriber = None
        self.viser_server = None
        self.viser_scene = None
        self.timer = None
        self.closed = False

    # ----- Config and scene setup -----

    def load_config(self) -> dict:
        path = self.config.visualizer_yaml or self.config.robot.config_dir / "visualizer.yaml"
        if not path.exists():
            log(f"no visualizer yaml found at {path}; using automatic joint mapping")
            return {}
        return load_yaml(path)

    def Init(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(self.config.robot.xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        self.base_qpos_adr = self.resolve_base_qpos_adr()
        self.apply_initial_base_pose()
        self.joint_map = self.build_joint_map()

        self.viser_server = viser.ViserServer()
        self.viser_scene = StandaloneMujocoScene.create(
            self.viser_server,
            self.model,
            show_camera_frustums=self.show_camera_frustums,
            real_sense_configs=self.build_camera_configs(),
        )

        self.lowstate_subscriber = ChannelSubscriber(self.lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        if self.use_odom and self.base_qpos_adr is not None:
            self.odomstate_subscriber = ChannelSubscriber(self.odom_topic, SportModeState_)
            self.odomstate_subscriber.Init(self.OdomStateHandler, 10)

        log(
            f"robot={self.config.robot.name} joints={len(self.joint_map)} "
            f"lowstate={self.lowstate_topic} odom={'on' if self.odomstate_subscriber else 'off'}"
        )

    # ----- MuJoCo model mapping -----

    def resolve_base_qpos_adr(self) -> int | None:
        base_joint = self.cfg.get("base_joint")
        if base_joint:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, base_joint)
            if joint_id < 0:
                log(f"base joint '{base_joint}' not found; odom disabled")
                return None
            if int(self.model.jnt_type[joint_id]) != mujoco.mjtJoint.mjJNT_FREE:
                log(f"base joint '{base_joint}' is not a freejoint; odom disabled")
                return None
            return int(self.model.jnt_qposadr[joint_id])

        for joint_id in range(self.model.njnt):
            if int(self.model.jnt_type[joint_id]) == mujoco.mjtJoint.mjJNT_FREE:
                return int(self.model.jnt_qposadr[joint_id])
        return None

    def apply_initial_base_pose(self) -> None:
        if self.base_qpos_adr is None:
            return

        initial_base = self.cfg.get("initial_base", {})
        pos = initial_base.get("pos")
        quat = initial_base.get("quat")
        if pos is not None:
            self.data.qpos[self.base_qpos_adr : self.base_qpos_adr + 3] = np.asarray(
                pos,
                dtype=np.float64,
            )
        if quat is not None:
            self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7] = np.asarray(
                quat,
                dtype=np.float64,
            )

    def actuator_joint_names(self) -> list[str]:
        names = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name:
                names.append(name)
        return names

    def build_joint_map(self) -> list[tuple[int, int]]:
        joint_names = self.cfg.get("state_joint_names") or self.actuator_joint_names()
        joint_map = []
        for msg_i, name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                log(f"joint '{name}' not found; skipping")
                continue
            if int(self.model.jnt_type[joint_id]) not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                log(f"joint '{name}' is not scalar; skipping")
                continue
            joint_map.append((msg_i, int(self.model.jnt_qposadr[joint_id])))
        return joint_map

    # ----- Optional RealSense visualization -----

    def build_camera_configs(self) -> list[RealSenseCameraConfig] | None:
        enable = bool(self.cfg.get("enable_cameras", False))
        if self.config.camera is not None:
            enable = self.config.camera
        if not enable:
            return None

        configs = []
        for item in self.cfg.get("cameras", []):
            width = int(item.get("width", item.get("color_width", 640)))
            height = int(item.get("height", item.get("color_height", 480)))
            configs.append(
                RealSenseCameraConfig(
                    camera_name=item.get("camera_name", item.get("name", "realsense")),
                    pose_camera_name=item.get("pose_camera_name"),
                    serial_number=item.get("serial_number", item.get("serial")),
                    color_width=int(item.get("color_width", width)),
                    color_height=int(item.get("color_height", height)),
                    depth_width=int(item.get("depth_width", width)),
                    depth_height=int(item.get("depth_height", height)),
                    fps=int(item.get("fps", 30)),
                    enable_depth=bool(item.get("enable_depth", True)),
                    align_depth_to_color=bool(item.get("align_depth_to_color", True)),
                    depth_visualization_min_m=float(item.get("depth_visualization_min_m", 0.1)),
                    depth_visualization_max_m=float(item.get("depth_visualization_max_m", 3.0)),
                    frustum_scale=float(item.get("frustum_scale", 0.15)),
                    jpeg_quality=item.get("jpeg_quality", 80),
                )
            )

        if not configs:
            log("camera enabled but no cameras configured")
            return None
        return configs

    # ----- DDS callbacks and render loop -----

    def Start(self) -> None:
        self.timer = RecurrentThread(interval=0.05, target=self.Visualize, name="visualize")
        self.timer.Start()

    def LowStateHandler(self, msg: LowState_) -> None:
        if self.data is None:
            return

        with self.lock:
            for msg_i, qpos_adr in self.joint_map:
                if msg_i < len(msg.motor_state):
                    self.data.qpos[qpos_adr] = float(msg.motor_state[msg_i].q)

    def OdomStateHandler(self, msg: SportModeState_) -> None:
        if self.data is None or self.base_qpos_adr is None:
            return

        with self.lock:
            # Odom updates only the floating base; joint qpos continues to come from LowState.
            self.data.qpos[self.base_qpos_adr : self.base_qpos_adr + 3] = np.asarray(
                msg.position[:3],
                dtype=np.float64,
            )
            self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7] = np.asarray(
                msg.imu_state.quaternion[:4],
                dtype=np.float64,
            )

    def Visualize(self) -> None:
        if self.viser_scene is None or self.data is None:
            return

        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            self.viser_scene.update_from_mjdata(self.data)

    # ----- Cleanup -----

    def Close(self) -> None:
        if self.closed:
            return
        self.closed = True

        if self.timer is not None:
            self.timer.Wait(1.0)
        if self.lowstate_subscriber is not None:
            self.lowstate_subscriber.Close()
        if self.odomstate_subscriber is not None:
            self.odomstate_subscriber.Close()
        if self.viser_scene is not None:
            self.viser_scene.close()
        if self.viser_server is not None:
            self.viser_server.stop()


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Robot state visualizer for real robot or simulation.")
    parser.add_argument("--robot", default=DEFAULT_ROBOT, help="Robot folder under robot_model/.")
    parser.add_argument("--model-xml", help="Override robot XML path.")
    parser.add_argument(
        "--terrain",
        default=DEFAULT_TERRAIN,
        help="Terrain name under robot_model/scene or XML path.",
    )
    parser.add_argument("--visualizer-yaml", type=Path, help="Override visualizer yaml path.")
    parser.add_argument("--net", default="lo", help="Optional DDS network interface.")
    parser.add_argument(
        "--mode",
        choices=("real", "sim"),
        default=DEFAULT_MODE,
        help="Run against a real robot or the sim bridge.",
    )
    parser.add_argument(
        "--camera",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override camera enable setting from visualizer.yaml.",
    )
    args = parser.parse_args()
    return RuntimeConfig(
        robot=load_robot_model(args.robot, args.model_xml, args.terrain),
        visualizer_yaml=args.visualizer_yaml,
        mode=args.mode,
        net=args.net,
        camera=args.camera,
    )


def main() -> None:
    config = parse_args()

    if config.net:
        ChannelFactoryInitialize(0, config.net)
    else:
        ChannelFactoryInitialize(0)

    visualizer = RobotStateVisualizer(config)
    try:
        visualizer.Init()
        visualizer.Start()

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.Close()


if __name__ == "__main__":
    main()
