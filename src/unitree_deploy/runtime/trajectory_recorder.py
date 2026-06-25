from __future__ import annotations

import argparse
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from pynput import keyboard
from unitree_deploy.config.defaults import (
    DEFAULT_NET,
    LOWSTATE_TOPIC,
    ODOM_TOPIC,
)
from unitree_deploy.robot_model.robot_config import (
    DEFAULT_ROBOT,
    DEFAULT_TERRAIN,
    RobotModel,
    load_robot_model,
)
from unitree_deploy.trajectory.recorder import TrajectoryRecorder
from unitree_deploy.utils.terminal_status import ComponentConsole
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


console = ComponentConsole("recorder", "magenta")


def log(message: str) -> None:
    console.log(message)


@dataclass(frozen=True)
class RuntimeConfig:
    robot: RobotModel
    out: Path
    net: str
    record_hz: float = 30.0


class KeyboardToggle:
    def __init__(self, on_toggle) -> None:
        self.on_toggle = on_toggle
        self.pressed: set[str] = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def start(self) -> None:
        self.listener.start()

    def stop(self) -> None:
        self.listener.stop()

    def _on_press(self, key) -> None:
        k = key_name(key)
        with self.lock:
            is_new = k not in self.pressed
            self.pressed.add(k)
        if is_new and k == "o":
            self.on_toggle()

    def _on_release(self, key) -> None:
        with self.lock:
            self.pressed.discard(key_name(key))


class DdsTrajectoryRecorder:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.model = mujoco.MjModel.from_xml_path(str(config.robot.xml_path))
        self.data = mujoco.MjData(self.model)
        self.motor_joint_ids = self.actuator_joint_ids()
        self.motor_qposadr = self.model.jnt_qposadr[self.motor_joint_ids].astype(np.int64)
        self.motor_dofadr = self.model.jnt_dofadr[self.motor_joint_ids].astype(np.int64)

        self.lock = threading.Lock()
        self.alive = True
        self.has_lowstate = False
        self.has_odom = False
        self.lowstate: LowState_ | None = None
        self.odom: SportModeState_ | None = None
        self.recorder: TrajectoryRecorder | None = None
        self.recording = False
        self.record_start_wall_t = -float("inf")
        self.last_sample_wall_t = -float("inf")
        self.samples = 0

        self.lowstate_sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        self.lowstate_sub.Init(self.on_lowstate, 1)
        self.odom_sub = ChannelSubscriber(ODOM_TOPIC, SportModeState_)
        self.odom_sub.Init(self.on_odom, 1)
        self.keyboard = KeyboardToggle(self.toggle_recording)

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    def actuator_joint_ids(self) -> np.ndarray:
        joint_ids = np.zeros(int(self.model.nu), dtype=np.int32)
        for i in range(int(self.model.nu)):
            joint_id = int(self.model.actuator_trnid[i, 0])
            joint_ids[i] = joint_id
        return joint_ids

    def on_lowstate(self, msg: LowState_) -> None:
        with self.lock:
            self.lowstate = msg
            self.has_lowstate = True

    def on_odom(self, msg: SportModeState_) -> None:
        with self.lock:
            self.odom = msg
            self.has_odom = True

    def toggle_recording(self) -> None:
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        with self.lock:
            if not self.has_lowstate:
                log("cannot start: waiting for LowState")
                return
            out_dir = make_recording_dir(self.config.out)
            self.recorder = TrajectoryRecorder(self.model, out_dir, metadata=self.metadata(out_dir))
            self.recording = True
            self.samples = 0
            self.record_start_wall_t = time.perf_counter()
            self.last_sample_wall_t = -float("inf")
        log(f"recording started -> {out_dir}")

    def stop_recording(self) -> None:
        with self.lock:
            recorder = self.recorder
            self.recorder = None
            self.recording = False
        if recorder is None:
            return
        path = recorder.save()
        log(f"recording saved -> {path}")

    def metadata(self, out_dir: Path) -> dict:
        return {
            "mode": "dds_trajectory_recorder",
            "robot": self.config.robot.name,
            "terrain": self.config.robot.terrain,
            "model_xml": self.config.robot.xml_path.as_posix(),
            "source_xml": self.config.robot.source_xml_path.as_posix(),
            "terrain_xml": self.config.robot.terrain_xml_path.as_posix(),
            "record_hz": self.config.record_hz,
            "record_root": self.config.out.expanduser().resolve().as_posix(),
            "record_dir": out_dir.as_posix(),
            "lowstate_topic": LOWSTATE_TOPIC,
            "odom_topic": ODOM_TOPIC,
            "actuator_joint_order": self.motor_joint_names(),
        }

    def motor_joint_names(self) -> list[str]:
        names = []
        for joint_id in self.motor_joint_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
            names.append(name or f"joint_{int(joint_id)}")
        return names

    def run(self) -> None:
        self.keyboard.start()
        log(f"robot={self.config.robot.name} terrain={self.config.robot.terrain}")
        log(f"topics: lowstate={LOWSTATE_TOPIC}, odom={ODOM_TOPIC}")
        log(f"press 'o' to start recording; press 'o' again to save")
        while self.alive:
            self.maybe_sample()
            time.sleep(0.001)

    def maybe_sample(self) -> None:
        with self.lock:
            if not self.recording or self.recorder is None or self.lowstate is None:
                return
            now = time.perf_counter()
            interval = 1.0 / max(self.config.record_hz, 1e-9)
            if now - self.last_sample_wall_t < interval - 1e-9:
                return
            lowstate = self.lowstate
            odom = self.odom
            recorder = self.recorder
            record_t = now - self.record_start_wall_t
            self.last_sample_wall_t = now

        self.update_mjdata(lowstate, odom, record_t=record_t)
        recorder.sample(self.data, policy_name="dds")
        self.samples += 1
        if self.samples % max(1, int(round(self.config.record_hz))) == 0:
            log(f"recording samples={self.samples} t={self.data.time:.2f}s")

    def update_mjdata(self, lowstate: LowState_, odom: SportModeState_ | None, *, record_t: float) -> None:
        self.data.time = float(record_t)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if odom is not None:
            self.data.qpos[:3] = np.asarray(odom.position[:3], dtype=np.float64)
            self.data.qvel[:3] = np.asarray(odom.velocity[:3], dtype=np.float64)
            self.data.qpos[3:7] = np.asarray(odom.imu_state.quaternion[:4], dtype=np.float64)
            self.data.qvel[3:6] = np.asarray(odom.imu_state.gyroscope[:3], dtype=np.float64)
        else:
            self.data.qpos[3] = 1.0
            self.data.qpos[2] = 1.0

        motor_count = min(int(self.model.nu), len(lowstate.motor_state))
        ctrl = np.zeros(int(self.model.nu), dtype=np.float64)
        for i in range(motor_count):
            state = lowstate.motor_state[i]
            self.data.qpos[self.motor_qposadr[i]] = float(state.q)
            self.data.qvel[self.motor_dofadr[i]] = float(state.dq)
            ctrl[i] = float(getattr(state, "tau_est", 0.0))
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)

    def close(self, *_args) -> None:
        if self.recording:
            self.stop_recording()
        self.alive = False
        console.stop()
        self.keyboard.stop()

    def cleanup(self) -> None:
        for obj in (self.lowstate_sub, self.odom_sub):
            try:
                obj.Close()
            except Exception:
                pass


def make_recording_dir(root: Path) -> Path:
    base = root.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in ("", *(f"-{i:03d}" for i in range(1, 1000))):
        candidate = base / f"{stamp}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a unique recording directory under {base}")


def key_name(key) -> str:
    try:
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        return str(key).replace("Key.", "").lower()
    except AttributeError:
        return str(key).replace("Key.", "").lower()


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Record Unitree DDS state into trajectory.npz for rendering.")
    parser.add_argument("--robot", default=DEFAULT_ROBOT)
    parser.add_argument("--model-xml", type=Path, help="Optional MuJoCo XML override.")
    parser.add_argument("--terrain", default=DEFAULT_TERRAIN)
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface.")
    parser.add_argument("--out", type=Path, required=True, help="Recording root directory.")
    parser.add_argument("--record-hz", type=float, default=60.0)
    args = parser.parse_args()
    if args.record_hz <= 0.0:
        parser.error("--record-hz must be positive")
    return RuntimeConfig(
        robot=load_robot_model(args.robot, args.model_xml, args.terrain),
        out=args.out,
        net=args.net,
        record_hz=float(args.record_hz),
    )


def main() -> None:
    config = parse_args()
    ChannelFactoryInitialize(0, config.net)
    recorder = DdsTrajectoryRecorder(config)
    try:
        recorder.run()
    finally:
        recorder.close()
        recorder.cleanup()


if __name__ == "__main__":
    main()
