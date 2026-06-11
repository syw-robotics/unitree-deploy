import argparse
import math
import signal
import struct
import threading
import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np
from robot_config import DEFAULT_ROBOT, RobotModel, load_robot_model
from sshkeyboard import listen_keyboard, stop_listening
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__SportModeState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


NET = "lo"
LOWCMD_TOPIC = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"
ODOM_TOPIC = "rt/odommodestate"

SIM_HZ = 500
STATE_HZ = 200
RENDER_HZ = 30
BASE_HEIGHT = 1.0
BASE_QUAT = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)

BAND_SITES = ("left_gantry_attach_point", "right_gantry_attach_point")
BAND_CLEARANCE = 0.35
BAND_STIFFNESS = 550.0
BAND_DAMPING = 45.0
BAND_STEP = 0.1
BAND_MIN_Z = 0.8
BAND_MAX_Z = 2.2
BAND_MAX_FORCE = 400.0

STICK = 1.0
REMOTE_BUTTONS = {
    "b": (3, 0),  # A
    "r": (3, 2),  # X
    "m": (2, 2),  # Start
}


def log(message: str) -> None:
    print(f"[sim_bridge] {message}", flush=True)


@dataclass(frozen=True)
class RuntimeConfig:
    robot: RobotModel
    net: str
    band_sites: tuple[str, ...]
    band_enabled: bool


class LoopTimer:
    def __init__(self, hz: int):
        self.dt = 1.0 / hz
        self.next_t = time.perf_counter() + self.dt

    def sleep(self) -> None:
        now = time.perf_counter()
        if self.next_t > now:
            time.sleep(self.next_t - now)
            self.next_t += self.dt
        else:
            self.next_t = now + self.dt


def quat_to_rpy(q: np.ndarray) -> list[float]:
    w, x, y, z = [float(v) for v in q]
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [roll, pitch, yaw]


class SimBridge:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.alive = True
        self.command_received = False
        self.tick = 1
        self.mode_machine = 0
        self.mode_pr = 0
        self.viewer = None
        self.viewer_tick = 0
        self.viewer_decim = max(1, SIM_HZ // RENDER_HZ)
        self.keys: set[str] = set()
        self.lock = threading.Lock()
        self.cmd_lock = threading.Lock()
        self.keys_lock = threading.Lock()

        self.model = mujoco.MjModel.from_xml_path(str(self.config.robot.xml_path))
        self.model.opt.timestep = 1.0 / SIM_HZ
        self.data = mujoco.MjData(self.model)
        self.num_motor = int(self.model.nu)
        self.ctrl_lower = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_upper = self.model.actuator_ctrlrange[:, 1].copy()

        self.base_qpos = np.array([0.0, 0.0, BASE_HEIGHT, *BASE_QUAT], dtype=np.float64)
        self.initial_joint_qpos = np.zeros(self.num_motor, dtype=np.float64)
        self.reset_sim(print_log=False)

        self.target_q = self.initial_joint_qpos.copy()
        self.target_dq = np.zeros(self.num_motor, dtype=np.float64)
        self.kp = np.zeros(self.num_motor, dtype=np.float64)
        self.kd = np.zeros(self.num_motor, dtype=np.float64)
        self.tau_ff = np.zeros(self.num_motor, dtype=np.float64)
        self.motor_enable = np.zeros(self.num_motor, dtype=bool)
        self.ctrl = np.zeros(self.num_motor, dtype=np.float64)

        self.imu_gyro = self.sensor_slice("imu_ang_vel", required=False)
        self.imu_acc = self.sensor_slice("imu_lin_acc", required=True)
        self.crc = CRC()

        self.band_on = self.config.band_enabled and bool(self.config.band_sites)
        self.band_site_ids = [self.site_id(name) for name in self.config.band_sites] if self.band_on else []
        self.band_jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.band_zero = np.zeros(3, dtype=np.float64)
        self.band_anchors = self.make_band_anchors()
        self.band_start_anchors = self.band_anchors.copy()
        self.band_z = float(np.mean(self.band_anchors[:, 2])) if self.band_on else 0.0
        self.band_start_z = self.band_z

        self.lowstate_pub = ChannelPublisher(LOWSTATE_TOPIC, LowState_)
        self.lowstate_pub.Init()
        self.odom_pub = ChannelPublisher(ODOM_TOPIC, SportModeState_)
        self.odom_pub.Init()
        self.lowcmd_sub = ChannelSubscriber(LOWCMD_TOPIC, LowCmd_)
        self.lowcmd_sub.Init(self.on_lowcmd)

        self.state_thread = threading.Thread(target=self.publish_state_loop, daemon=False)
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop, daemon=True)

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    def sensor_slice(self, name: str, required: bool):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            if required:
                raise ValueError(f"MuJoCo XML missing required sensor: {name}")
            return None
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        if required and dim < 3:
            raise ValueError(f"MuJoCo sensor {name} must have dim >= 3")
        return adr, dim

    def site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"MuJoCo XML missing required site: {name}")
        return int(site_id)

    def make_band_anchors(self) -> np.ndarray:
        anchors = []
        mujoco.mj_forward(self.model, self.data)
        for site_id in self.band_site_ids:
            anchor = self.data.site_xpos[site_id].copy()
            anchor[2] += BAND_CLEARANCE
            anchors.append(anchor)
        return np.asarray(anchors, dtype=np.float64)

    def reset_sim(self, print_log: bool = True) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.base_qpos
        self.data.qpos[7 : 7 + self.num_motor] = self.initial_joint_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        if hasattr(self, "target_q"):
            with self.cmd_lock:
                self.target_q[:] = self.initial_joint_qpos
                self.target_dq[:] = 0.0
                self.kp[:] = 0.0
                self.kd[:] = 0.0
                self.tau_ff[:] = 0.0
                self.motor_enable[:] = False
            self.band_on = self.config.band_enabled and bool(self.band_site_ids)
            if self.band_on:
                self.band_z = self.band_start_z
                self.band_anchors[:] = self.band_start_anchors

        if print_log:
            log("reset robot state to initial pose")

    def on_lowcmd(self, msg: LowCmd_) -> None:
        if msg is None:
            return

        with self.cmd_lock:
            self.mode_machine = int(getattr(msg, "mode_machine", self.mode_machine))
            self.mode_pr = int(getattr(msg, "mode_pr", self.mode_pr))
            motor_count = min(self.num_motor, len(msg.motor_cmd))
            for i in range(motor_count):
                cmd = msg.motor_cmd[i]
                self.target_q[i] = float(cmd.q)
                self.target_dq[i] = float(cmd.dq)
                self.kp[i] = float(cmd.kp)
                self.kd[i] = float(cmd.kd)
                self.tau_ff[i] = float(cmd.tau)
                self.motor_enable[i] = int(getattr(cmd, "mode", 1)) != 0
            self.command_received = True

    def keyboard_loop(self) -> None:
        log(
            "keyboard: Up/Down band, n release band, r reset/X, b A, m Start, "
            "wsad left stick, qe right stick x, esc quit"
        )
        try:
            listen_keyboard(
                on_press=self.on_key_press,
                on_release=self.on_key_release,
                until=None,
                sequential=True,
            )
        except Exception as exc:
            if self.alive:
                log(f"keyboard stopped: {exc}")

    def on_key_press(self, key: str) -> None:
        key = key.lower()
        with self.keys_lock:
            first_press = key not in self.keys
            self.keys.add(key)
        if not first_press:
            return

        if key == "up":
            self.move_band(BAND_STEP)
        elif key == "down":
            self.move_band(-BAND_STEP)
        elif key == "n":
            self.band_on = False
            log("suspension bands released")
        elif key == "r":
            with self.lock:
                self.reset_sim()
        elif key == "esc":
            self.close()

    def on_key_release(self, key: str) -> None:
        with self.keys_lock:
            self.keys.discard(key.lower())

    def move_band(self, dz: float) -> None:
        if not self.band_on:
            return
        self.band_z = float(np.clip(self.band_z + dz, BAND_MIN_Z, BAND_MAX_Z))
        self.band_anchors[:, 2] = self.band_z
        log(f"band height -> {self.band_z:.3f} m")

    def remote_bytes(self) -> list[int]:
        remote = [0] * 40
        with self.keys_lock:
            keys = set(self.keys)

        sticks = (
            STICK * ("a" in keys) - STICK * ("d" in keys),
            STICK * ("q" in keys) - STICK * ("e" in keys),
            0.0,
            STICK * ("w" in keys) - STICK * ("s" in keys),
        )
        for offset, value in zip((4, 8, 12, 20), sticks):
            remote[offset : offset + 4] = struct.pack("<f", value)

        for key, (byte_i, bit_i) in REMOTE_BUTTONS.items():
            if key in keys:
                remote[byte_i] |= 1 << bit_i
        return remote

    def apply_band(self) -> None:
        if not self.band_on:
            return

        for site_id, anchor in zip(self.band_site_ids, self.band_anchors):
            pos = self.data.site_xpos[site_id]
            self.band_jacp[:] = 0.0
            mujoco.mj_jacSite(self.model, self.data, self.band_jacp, None, site_id)
            vel = self.band_jacp @ self.data.qvel
            force = BAND_STIFFNESS * (anchor - pos) - BAND_DAMPING * vel
            force = np.clip(force, -BAND_MAX_FORCE, BAND_MAX_FORCE)
            body_id = int(self.model.site_bodyid[site_id])
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force,
                self.band_zero,
                pos,
                body_id,
                self.data.qfrc_applied,
            )

    def compute_ctrl(self) -> np.ndarray:
        q = self.data.qpos[7 : 7 + self.num_motor]
        dq = self.data.qvel[6 : 6 + self.num_motor]
        with self.cmd_lock:
            self.ctrl[:] = (
                self.kp * (self.target_q - q)
                + self.kd * (self.target_dq - dq)
                + self.tau_ff
            )
            self.ctrl[~self.motor_enable] = 0.0
        np.clip(self.ctrl, self.ctrl_lower, self.ctrl_upper, out=self.ctrl)
        return self.ctrl

    def snapshot(self):
        with self.lock:
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            ctrl = self.data.ctrl[: self.num_motor].copy()
            sensordata = self.data.sensordata.copy()

        gyro = qvel[3:6]
        if self.imu_gyro is not None:
            adr, dim = self.imu_gyro
            if dim >= 3:
                gyro = sensordata[adr : adr + 3]
        acc_adr, _ = self.imu_acc
        acc = sensordata[acc_adr : acc_adr + 3]
        return qpos, qvel, ctrl, gyro, acc

    def make_lowstate(self, qpos, qvel, ctrl, gyro, acc) -> LowState_:
        msg = unitree_hg_msg_dds__LowState_()
        msg.mode_pr = int(self.mode_pr)
        msg.mode_machine = int(self.mode_machine)
        msg.tick = int(self.tick)

        for i in range(self.num_motor):
            msg.motor_state[i].q = float(qpos[7 + i])
            msg.motor_state[i].dq = float(qvel[6 + i])
            msg.motor_state[i].tau_est = float(ctrl[i])

        quat = qpos[3:7]
        msg.imu_state.quaternion = quat.tolist()
        msg.imu_state.gyroscope = gyro.tolist()
        msg.imu_state.accelerometer = acc.tolist()
        if hasattr(msg.imu_state, "rpy"):
            msg.imu_state.rpy = quat_to_rpy(quat)
        msg.wireless_remote = self.remote_bytes()
        msg.crc = self.crc.Crc(msg)

        self.tick += 1
        return msg

    def make_odom(self, qpos, qvel, gyro, acc) -> SportModeState_:
        msg = unitree_go_msg_dds__SportModeState_()
        quat = qpos[3:7]
        msg.position = qpos[:3].tolist()
        msg.velocity = qvel[:3].tolist()
        msg.body_height = float(qpos[2])
        msg.yaw_speed = float(gyro[2])
        msg.imu_state.quaternion = quat.tolist()
        msg.imu_state.gyroscope = gyro.tolist()
        msg.imu_state.accelerometer = acc.tolist()
        if hasattr(msg.imu_state, "rpy"):
            msg.imu_state.rpy = quat_to_rpy(quat)
        return msg

    def publish_state_loop(self) -> None:
        timer = LoopTimer(STATE_HZ)
        while self.alive:
            qpos, qvel, ctrl, gyro, acc = self.snapshot()
            self.lowstate_pub.Write(self.make_lowstate(qpos, qvel, ctrl, gyro, acc))
            self.odom_pub.Write(self.make_odom(qpos, qvel, gyro, acc))
            timer.sleep()

    def sync_viewer(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            self.alive = False
            return False
        self.viewer_tick += 1
        if self.viewer_tick % self.viewer_decim == 0:
            self.viewer.sync()
        return True

    def simulate(self) -> None:
        timer = LoopTimer(SIM_HZ)
        last_log = time.perf_counter()
        steps = 0

        while self.alive:
            with self.lock:
                self.data.qfrc_applied[:] = 0.0
                self.apply_band()
                self.data.ctrl[:] = self.compute_ctrl()
                mujoco.mj_step(self.model, self.data)

            if not self.sync_viewer():
                break

            steps += 1
            now = time.perf_counter()
            if now - last_log >= 1.0:
                log(
                    f"t={steps / SIM_HZ:6.2f}s height={self.data.qpos[2]:.3f} "
                    f"cmd={'yes' if self.command_received else 'no'} "
                    f"band={'on' if self.band_on else 'off'}"
                )
                last_log = now
            timer.sleep()

    def run(self) -> None:
        log(f"robot={self.config.robot.name} model={self.config.robot.xml_path}")
        log(f"topics: lowcmd={LOWCMD_TOPIC}, lowstate={LOWSTATE_TOPIC}, odom={ODOM_TOPIC}")
        log(f"sim={SIM_HZ}Hz state_pub={STATE_HZ}Hz")
        if self.band_on:
            log(f"suspension bands enabled at z={self.band_z:.3f} m")

        self.state_thread.start()
        self.keyboard_thread.start()
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            self.viewer = viewer
            self.simulate()
            self.viewer = None

    def close(self, *_args) -> None:
        if self.alive:
            log("shutting down...")
        self.alive = False

        try:
            stop_listening()
        except Exception:
            pass

        for thread in (self.keyboard_thread, self.state_thread):
            if thread.is_alive() and threading.current_thread() is not thread:
                thread.join(timeout=1.0)

    def cleanup(self) -> None:
        self.alive = False
        for obj in (self.lowcmd_sub, self.lowstate_pub, self.odom_pub):
            try:
                obj.Close()
            except Exception:
                pass


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="MuJoCo to Unitree DDS bridge.")
    parser.add_argument("--robot", default=DEFAULT_ROBOT, help="Robot folder under robot_model/.")
    parser.add_argument("--model-xml", help="Override robot XML path.")
    parser.add_argument("--net", default=NET, help="DDS network interface.")
    parser.add_argument(
        "--band-sites",
        default=",".join(BAND_SITES),
        help="Comma-separated MuJoCo site names used by the suspension bands.",
    )
    parser.add_argument(
        "--band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable suspension bands.",
    )
    args = parser.parse_args()
    band_sites = tuple(site.strip() for site in args.band_sites.split(",") if site.strip())
    return RuntimeConfig(
        robot=load_robot_model(args.robot, args.model_xml),
        net=args.net,
        band_sites=band_sites,
        band_enabled=bool(args.band),
    )


if __name__ == "__main__":
    config = parse_args()
    ChannelFactoryInitialize(0, config.net)
    bridge = SimBridge(config)
    try:
        bridge.run()
    finally:
        bridge.close()
        bridge.cleanup()
