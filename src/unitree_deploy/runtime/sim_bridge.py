import argparse
import math
import signal
import struct
import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np
from unitree_deploy.config.defaults import (
    ACC_SENSOR_NAMES,
    BAND_CLEARANCE,
    BAND_DAMPING,
    BAND_MAX_FORCE,
    BAND_MAX_Z,
    BAND_MIN_Z,
    BAND_SITES,
    BAND_STEP,
    BAND_STIFFNESS,
    BASE_HEIGHT,
    BASE_QUAT,
    DEFAULT_NET,
    GYRO_SENSOR_NAMES,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
    ODOM_TOPIC,
    REMOTE_STICK_SCALE,
    RENDER_HZ,
    SIM_HZ,
    SIM_REMOTE_BUTTON_KEYS,
    STATE_HZ,
    sim_key_for_button,
)
from unitree_deploy.robot_model.robot_config import (
    DEFAULT_ROBOT,
    DEFAULT_TERRAIN,
    DEFAULT_VIEWER,
    VIEWER_CHOICES,
    RobotModel,
    load_robot_model,
)
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
from unitree_deploy.utils.viewer_backend import create_viewer_backend


def log(message: str) -> None:
    print(f"[sim_bridge] {message}", flush=True)


@dataclass(frozen=True)
class RuntimeConfig:
    robot: RobotModel
    net: str
    viewer: str
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
    """MuJoCo <-> Unitree DDS bridge used for sim2sim validation.

    Data flow:
      LowCmd -> MuJoCo PD control -> physics step -> LowState/Odom DDS topics
      keyboard -> wireless_remote bytes and optional suspension-band controls
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.alive = True
        self.command_received = False
        self.simulation_paused = True
        self.tick = 1
        self.mode_machine = 0
        self.mode_pr = 0
        self.keys: set[str] = set()
        self.lock = threading.Lock()
        self.cmd_lock = threading.Lock()
        self.keys_lock = threading.Lock()

        self.model = mujoco.MjModel.from_xml_path(str(self.config.robot.xml_path))
        self.model.opt.timestep = 1.0 / SIM_HZ
        self.data = mujoco.MjData(self.model)
        # Viewer backend owns GUI/server lifecycle; the physics loop only calls sync().
        self.viewer_backend = create_viewer_backend(
            self.config.viewer,
            self.config.robot,
            self.model,
            self.data,
            sim_hz=SIM_HZ,
            render_hz=RENDER_HZ,
            log=log,
        )
        self.num_motor = int(self.model.nu)
        # Actuator ctrlrange is the final safety clamp before writing data.ctrl.
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

        self.imu_gyro = self.sensor_slice(
            GYRO_SENSOR_NAMES,
            mujoco.mjtSensor.mjSENS_GYRO,
            label="IMU gyro",
            required=False,
        )
        self.imu_acc = self.sensor_slice(
            ACC_SENSOR_NAMES,
            mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            label="IMU accelerometer",
            required=False,
        )
        self.crc = CRC()

        self.band_site_ids = (
            self.site_ids(self.config.band_sites) if self.config.band_enabled else []
        )
        self.band_on = self.config.band_enabled and bool(self.band_site_ids)
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

    # ----- MuJoCo model lookup helpers -----

    def sensor_slice_by_id(self, sid: int, label: str, required: bool):
        adr = int(self.model.sensor_adr[sid])
        dim = int(self.model.sensor_dim[sid])
        if required and dim < 3:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            raise ValueError(f"MuJoCo {label} sensor {name} must have dim >= 3")
        if dim < 3:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            log(f"ignoring {label} sensor {name}: dim={dim} < 3")
            return None
        return adr, dim

    def sensor_slice(
        self,
        names: tuple[str, ...],
        sensor_type: mujoco.mjtSensor,
        label: str,
        required: bool,
    ):
        for name in names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid >= 0:
                return self.sensor_slice_by_id(int(sid), label, required)

        for sid in range(self.model.nsensor):
            if int(self.model.sensor_type[sid]) == int(sensor_type):
                return self.sensor_slice_by_id(int(sid), label, required)

        if required:
            raise ValueError(
                f"MuJoCo XML missing required {label} sensor; tried names: {', '.join(names)}"
            )
        log(f"MuJoCo XML has no {label} sensor; using fallback data")
        return None

    def site_ids(self, names: tuple[str, ...]) -> list[int]:
        site_ids = []
        missing = []
        for name in names:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id < 0:
                missing.append(name)
            else:
                site_ids.append(int(site_id))

        if missing:
            log(f"MuJoCo XML missing suspension band site(s), skipping: {', '.join(missing)}")
        if names and not site_ids:
            log("suspension bands disabled for this XML")
        return site_ids

    # ----- Reset and suspension-band setup -----

    def make_band_anchors(self) -> np.ndarray:
        if not self.band_site_ids:
            return np.zeros((0, 3), dtype=np.float64)

        anchors = np.zeros((len(self.band_site_ids), 3), dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        for i, site_id in enumerate(self.band_site_ids):
            anchors[i] = self.data.site_xpos[site_id]
            anchors[i, 2] += BAND_CLEARANCE
        return anchors

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

    # ----- DDS input: LowCmd from controller.py -----

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

    # ----- Keyboard controls and virtual wireless remote -----

    def keyboard_loop(self) -> None:
        log(
            f"keyboard: Up/Down band, n release band, {sim_key_for_button('X')} reset/X, "
            f"{sim_key_for_button('A')} A, {sim_key_for_button('Start')} Start, "
            "space pause/resume, wsad left stick, qe right stick x, esc quit"
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
        if key == sim_key_for_button("A"):
            self.resume_simulation()

        with self.keys_lock:
            first_press = key not in self.keys
            self.keys.add(key)
        if not first_press:
            return

        if key == "space":
            self.toggle_simulation_pause()
        elif key == "up":
            self.move_band(BAND_STEP)
        elif key == "down":
            self.move_band(-BAND_STEP)
        elif key == "n":
            self.band_on = False
            log("suspension bands released")
        elif key == sim_key_for_button("X"):
            with self.lock:
                self.reset_sim()
        elif key == "esc":
            self.close()

    def on_key_release(self, key: str) -> None:
        with self.keys_lock:
            self.keys.discard(key.lower())

    def resume_simulation(self) -> None:
        if not self.simulation_paused:
            return
        self.simulation_paused = False
        log("simulation resumed by A")

    def toggle_simulation_pause(self) -> None:
        self.simulation_paused = not self.simulation_paused
        state = "paused" if self.simulation_paused else "resumed"
        log(f"simulation {state} by space")

    def move_band(self, dz: float) -> None:
        if not self.band_on:
            return
        self.band_z = float(np.clip(self.band_z + dz, BAND_MIN_Z, BAND_MAX_Z))
        self.band_anchors[:, 2] = self.band_z
        log(f"band height -> {self.band_z:.3f} m")

    def remote_bytes(self) -> list[int]:
        remote = bytearray(40)
        with self.keys_lock:
            keys = set(self.keys)
        stick_keys = keys - set(SIM_REMOTE_BUTTON_KEYS)

        # Match Unitree wireless_remote layout consumed by controller.RemoteCommand.
        for offset, value in zip(
            (4, 8, 12, 20),
            (
                REMOTE_STICK_SCALE * ("a" in stick_keys) - REMOTE_STICK_SCALE * ("d" in stick_keys),
                REMOTE_STICK_SCALE * ("q" in stick_keys) - REMOTE_STICK_SCALE * ("e" in stick_keys),
                0.0,
                REMOTE_STICK_SCALE * ("w" in stick_keys) - REMOTE_STICK_SCALE * ("s" in stick_keys),
            ),
        ):
            struct.pack_into("<f", remote, offset, value)

        for key, (byte_i, bit_i) in SIM_REMOTE_BUTTON_KEYS.items():
            if key in keys:
                remote[byte_i] |= 1 << bit_i
        return list(remote)

    # ----- Physics step helpers -----

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
        # PD target arrays are written by the DDS callback; this runs inside the MuJoCo step loop.
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

    # ----- DDS output: simulated robot state -----

    def state_snapshot(self):
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
        acc = np.zeros(3, dtype=np.float64)
        if self.imu_acc is not None:
            acc_adr, dim = self.imu_acc
            if dim >= 3:
                acc = sensordata[acc_adr : acc_adr + 3]
        return qpos, qvel, ctrl, gyro, acc

    @staticmethod
    def fill_imu(msg, quat, gyro, acc) -> None:
        msg.imu_state.quaternion = quat.tolist()
        msg.imu_state.gyroscope = gyro.tolist()
        msg.imu_state.accelerometer = acc.tolist()
        if hasattr(msg.imu_state, "rpy"):
            msg.imu_state.rpy = quat_to_rpy(quat)

    def make_lowstate(self, qpos, qvel, ctrl, gyro, acc) -> LowState_:
        msg = unitree_hg_msg_dds__LowState_()
        msg.mode_pr = int(self.mode_pr)
        msg.mode_machine = int(self.mode_machine)
        msg.tick = int(self.tick)

        for i in range(self.num_motor):
            msg.motor_state[i].q = float(qpos[7 + i])
            msg.motor_state[i].dq = float(qvel[6 + i])
            msg.motor_state[i].tau_est = float(ctrl[i])

        self.fill_imu(msg, qpos[3:7], gyro, acc)
        msg.wireless_remote = self.remote_bytes()
        msg.crc = self.crc.Crc(msg)

        self.tick += 1
        return msg

    def make_odom(self, qpos, qvel, gyro, acc) -> SportModeState_:
        msg = unitree_go_msg_dds__SportModeState_()
        msg.position = qpos[:3].tolist()
        msg.velocity = qvel[:3].tolist()
        msg.body_height = float(qpos[2])
        msg.yaw_speed = float(gyro[2])
        self.fill_imu(msg, qpos[3:7], gyro, acc)
        return msg

    def publish_state_loop(self) -> None:
        timer = LoopTimer(STATE_HZ)
        while self.alive:
            qpos, qvel, ctrl, gyro, acc = self.state_snapshot()
            self.lowstate_pub.Write(self.make_lowstate(qpos, qvel, ctrl, gyro, acc))
            self.odom_pub.Write(self.make_odom(qpos, qvel, gyro, acc))
            timer.sleep()

    # ----- Simulation loop and cleanup -----

    def simulate(self) -> None:
        timer = LoopTimer(SIM_HZ)
        last_log = time.perf_counter()
        steps = 0

        while self.alive:
            if not self.simulation_paused:
                with self.lock:
                    self.data.qfrc_applied[:] = 0.0
                    self.apply_band()
                    self.data.ctrl[:] = self.compute_ctrl()
                    mujoco.mj_step(self.model, self.data)

            if not self.viewer_backend.sync():
                self.alive = False
                break

            if not self.simulation_paused:
                steps += 1
            now = time.perf_counter()
            if now - last_log >= 1.0:
                if self.simulation_paused:
                    log(f"simulation paused; press {sim_key_for_button('A')} (virtual A) to start")
                else:
                    log(
                        f"t={steps / SIM_HZ:6.2f}s height={self.data.qpos[2]:.3f} "
                        f"cmd={'yes' if self.command_received else 'no'} "
                        f"band={'on' if self.band_on else 'off'}"
                    )
                last_log = now
            timer.sleep()

    def run(self) -> None:
        log(
            f"robot={self.config.robot.name} terrain={self.config.robot.terrain} "
            f"model={self.config.robot.xml_path}"
        )
        log(f"topics: lowcmd={LOWCMD_TOPIC}, lowstate={LOWSTATE_TOPIC}, odom={ODOM_TOPIC}")
        log(f"sim={SIM_HZ}Hz state_pub={STATE_HZ}Hz viewer={self.config.viewer}")
        log(f"simulation starts paused; press {sim_key_for_button('A')} (virtual A) to continue")
        if self.band_on:
            log(f"suspension bands enabled at z={self.band_z:.3f} m")

        self.state_thread.start()
        self.keyboard_thread.start()
        # Blocks until the selected viewer or simulation loop exits.
        self.viewer_backend.run(self.simulate)

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
    parser.add_argument(
        "--terrain",
        default=DEFAULT_TERRAIN,
        help="Terrain name under robot_model/scene or XML path.",
    )
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface.")
    parser.add_argument(
        "--viewer",
        choices=VIEWER_CHOICES,
        default=DEFAULT_VIEWER,
        help="Visualization backend: mujoco or mjswan.",
    )
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
        robot=load_robot_model(args.robot, args.model_xml, args.terrain),
        net=args.net,
        viewer=args.viewer,
        band_sites=band_sites,
        band_enabled=bool(args.band),
    )


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize(0, config.net)
    bridge = SimBridge(config)
    try:
        bridge.run()
    finally:
        bridge.close()
        bridge.cleanup()


if __name__ == "__main__":
    main()
