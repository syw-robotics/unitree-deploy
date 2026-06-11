import argparse
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import viser
from robot_config import DEFAULT_ROBOT, RobotModel, load_robot_model
from scene_config import RealSenseCameraConfig, StandaloneMujocoScene
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.thread import RecurrentThread

DEFAULT_MODE = "sim"
DEFAULT_ENABLE_CAMERA = False
DEFAULT_REALSENSE_SERIAL = "140122071098"
DEFAULT_REALSENSE_CAMERA_NAME = "d435_head"
DEFAULT_REALSENSE_POSE_CAMERA_NAME = "d435_head"
DEFAULT_REALSENSE_WIDTH = 640
DEFAULT_REALSENSE_HEIGHT = 480
DEFAULT_REALSENSE_FPS = 30
DEFAULT_REALSENSE_ENABLE_DEPTH = True
DEFAULT_LOWSTATE_TOPIC = "rt/lowstate"
DEFAULT_ODOMSTATE_TOPIC = "rt/odommodestate"


@dataclass(frozen=True)
class RuntimeConfig:
    robot: RobotModel
    mode: str
    net: str | None
    enable_camera: bool


class RobotStateVisualizer:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.low_state = None
        self.odom_state = None
        self.lowstate_subscriber = None
        self.odomstate_subscriber = None
        self.model = None
        self.data = None
        self.viser_server = None
        self.viser_scene = None
        self.timerPtr = None
        self._visualize_thread_started = False
        self._closed = False

    def Init(self):
        self.model = mujoco.MjModel.from_xml_path(str(self.config.robot.xml_path))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = np.zeros(self.model.nq)
        self.data.qpos[2] = 1.0
        self.data.qpos[3] = 1.0

        self.viser_server = viser.ViserServer()
        real_sense_configs = None
        if self.config.enable_camera:
            real_sense_configs = [
                RealSenseCameraConfig(
                    camera_name=DEFAULT_REALSENSE_CAMERA_NAME,
                    pose_camera_name=DEFAULT_REALSENSE_POSE_CAMERA_NAME,
                    serial_number=DEFAULT_REALSENSE_SERIAL,
                    color_width=DEFAULT_REALSENSE_WIDTH,
                    color_height=DEFAULT_REALSENSE_HEIGHT,
                    depth_width=DEFAULT_REALSENSE_WIDTH,
                    depth_height=DEFAULT_REALSENSE_HEIGHT,
                    fps=DEFAULT_REALSENSE_FPS,
                    enable_depth=DEFAULT_REALSENSE_ENABLE_DEPTH,
                )
            ]
        self.viser_scene = StandaloneMujocoScene.create(
            self.viser_server,
            self.model,
            show_camera_frustums=True,
            real_sense_configs=real_sense_configs,
        )

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber(DEFAULT_LOWSTATE_TOPIC, LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)
        self.odomstate_subscriber = ChannelSubscriber(DEFAULT_ODOMSTATE_TOPIC, SportModeState_)
        self.odomstate_subscriber.Init(self.OdomStateHandler, 10)

    def Start(self):
        self.timerPtr = RecurrentThread(interval=0.05, target=self.Visualize, name="visualize")
        self.timerPtr.Start()
        self._visualize_thread_started = True

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.data is None:
            return

        num_motor = min(self.model.nu, len(self.low_state.motor_state))
        self.data.qpos[7 : 7 + num_motor] = [
            self.low_state.motor_state[i].q for i in range(num_motor)
        ]

    def OdomStateHandler(self, msg: SportModeState_):
        self.odom_state = msg

        if self.data is None:
            return

        self.data.qpos[:3] = np.asarray(self.odom_state.position[:3], dtype=np.float64)
        self.data.qpos[3:7] = np.asarray(
            self.odom_state.imu_state.quaternion[:4], dtype=np.float64
        )

    def Visualize(self):
        if self.viser_scene is None or self.data is None:
            return

        mujoco.mj_forward(self.model, self.data)
        self.viser_scene.update_from_mjdata(self.data)

    def Close(self):
        if self._closed:
            return
        self._closed = True

        if self._visualize_thread_started and self.timerPtr is not None:
            self.timerPtr.Wait(1.0)

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
        default=DEFAULT_ENABLE_CAMERA,
        help="Enable or disable camera-related visualization and capture.",
    )
    args = parser.parse_args()
    return RuntimeConfig(
        robot=load_robot_model(args.robot, args.model_xml),
        mode=args.mode,
        net=args.net,
        enable_camera=bool(args.camera),
    )


if __name__ == "__main__":
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
