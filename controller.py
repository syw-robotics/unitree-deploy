import argparse
import signal
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from policy.observation import ObservationContext
from policy.base_policy import BasePolicy
from robot_config import DEFAULT_ROBOT
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


LOWCMD_TOPIC = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"
LOCAL_NET = "lo"
MODE = "sim"
MOVE_TO_DEFAULT_TIME = 2.0

ZERO_TORQUE = "zero_torque_state"
MOVE_TO_DEFAULT = "move_to_default_qpos"
DEFAULT_QPOS = "default_qpos_state"
RUN_POLICY = "run_policy"

BUTTON_BITS = {
    "Start": (2, 2),
    "A": (3, 0),
    "X": (3, 2),
}


def log(message: str) -> None:
    print(f"[controller] {message}", flush=True)


class LoopTimer:
    def __init__(self, dt: float):
        self.dt = float(dt)
        self.next_t = time.perf_counter() + self.dt

    def sleep(self) -> None:
        now = time.perf_counter()
        if self.next_t > now:
            time.sleep(self.next_t - now)
            self.next_t += self.dt
        else:
            self.next_t = now + self.dt


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    net: str | None
    ckpt_dir: Path
    robot: str | None


class RemoteCommand:
    def __init__(self) -> None:
        self.lx = self.ly = self.rx = self.ry = 0.0
        self.buttons = {name: False for name in BUTTON_BITS}
        self.pressed_edges: set[str] = set()

    def set(self, wireless_remote) -> None:
        payload = bytes(wireless_remote)
        self.lx = struct.unpack("<f", payload[4:8])[0]
        self.rx = struct.unpack("<f", payload[8:12])[0]
        self.ry = struct.unpack("<f", payload[12:16])[0]
        self.ly = struct.unpack("<f", payload[20:24])[0]

        for name, (byte_i, bit_i) in BUTTON_BITS.items():
            pressed = bool((int(wireless_remote[byte_i]) >> bit_i) & 1)
            if pressed and not self.buttons[name]:
                self.pressed_edges.add(name)
            self.buttons[name] = pressed

    def consume(self, name: str) -> bool:
        if name not in self.pressed_edges:
            return False
        self.pressed_edges.remove(name)
        return True


class Controller:
    """Real-time Unitree controller shared by sim and real deployment.

    Data flow:
      LowState -> policy joint order -> observation -> ONNX policy -> raw joint order -> LowCmd
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.ckpt_dir = config.ckpt_dir.resolve()
        self.cfg = self.load_yaml(self.ckpt_dir / "controller.yaml")
        self.policy = BasePolicy(self.ckpt_dir / "policy.yaml")
        self.robot = config.robot or self.cfg.get("robot", DEFAULT_ROBOT)

        self.lowcmd_topic = self.cfg.get("lowcmd_topic", LOWCMD_TOPIC)
        self.lowstate_topic = self.cfg.get("lowstate_topic", LOWSTATE_TOPIC)
        self.real_joint_names = list(self.cfg["real_joint_names"])
        self.policy_joint_names = list(self.cfg["isaac_joint_names_state"])
        # "raw" is the order used by the active backend: MuJoCo in sim, DDS motor order on real hardware.
        # The policy always sees the Isaac training order.
        raw_key = "mujoco_joint_names" if config.mode == "sim" else "real_joint_names"
        self.raw_joint_names = list(self.cfg[raw_key])
        self.num_joints = len(self.raw_joint_names)
        self.check_joint_config()

        self.raw_to_policy = self.reorder_indices(self.raw_joint_names, self.policy_joint_names)
        self.policy_to_raw = self.reorder_indices(self.policy_joint_names, self.raw_joint_names)
        self.kp = self.gain_array("kps_real")
        self.kd = self.gain_array("kds_real")
        self.zero = np.zeros(self.num_joints, dtype=np.float64)
        self.zero_torque_kd = np.ones(self.num_joints, dtype=np.float64)
        self.default_q_policy = self.policy.default_joint_pos
        self.default_q_raw = self.default_q_policy[self.policy_to_raw]
        self.target_policy = np.zeros(self.num_joints, dtype=np.float64)
        self.target_raw = np.zeros(self.num_joints, dtype=np.float64)

        self.lock = threading.Lock()
        self.alive = True
        self.cleanup_done = False
        self.has_low_state = False
        self.mode_machine = 0
        self.mode_pr = 0
        self.state = ZERO_TORQUE
        self.state_enter_t = time.perf_counter()

        self.q = np.zeros(self.num_joints, dtype=np.float64)
        self.dq = np.zeros(self.num_joints, dtype=np.float64)
        self.q_policy = np.zeros(self.num_joints, dtype=np.float64)
        self.dq_policy = np.zeros(self.num_joints, dtype=np.float64)
        self.move_start_q = np.zeros(self.num_joints, dtype=np.float64)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.gyro = np.zeros(3, dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float64)
        self.remote = RemoteCommand()

        if config.mode == "real":
            self.enter_debug_mode()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.lowstate_sub = ChannelSubscriber(self.lowstate_topic, LowState_)
        self.lowstate_sub.Init(self.on_lowstate, 1)
        self.lowcmd_pub = ChannelPublisher(self.lowcmd_topic, LowCmd_)
        self.lowcmd_pub.Init()

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

    # ----- Config and joint-order helpers -----

    @staticmethod
    def load_yaml(path: Path):
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def reorder_indices(source: list[str], target: list[str]) -> np.ndarray:
        source_index = {name: i for i, name in enumerate(source)}
        return np.asarray([source_index[name] for name in target], dtype=np.int64)

    def check_joint_config(self) -> None:
        if len(self.policy_joint_names) != self.num_joints:
            raise ValueError(
                f"joint count mismatch: {len(self.policy_joint_names)} policy joints, "
                f"{self.num_joints} raw joints"
            )
        if len(self.policy.default_joint_pos) != self.num_joints:
            raise ValueError(
                f"policy default_qpos has {len(self.policy.default_joint_pos)} joints, "
                f"controller has {self.num_joints}"
            )

    def gain_array(self, key: str) -> np.ndarray:
        values = np.asarray(self.cfg[key], dtype=np.float64).reshape(-1)
        by_name = dict(zip(self.real_joint_names, values))
        return np.asarray([by_name[name] for name in self.raw_joint_names], dtype=np.float64)

    def write_policy_to_raw(self, value: np.ndarray) -> np.ndarray:
        np.take(value, self.policy_to_raw, out=self.target_raw)
        return self.target_raw

    # ----- Real-robot setup -----

    def enter_debug_mode(self) -> None:
        log("real mode: releasing current motion mode...")
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()

        _, result = msc.CheckMode()
        while result["name"]:
            msc.ReleaseMode()
            _, result = msc.CheckMode()
            time.sleep(1.0)
        log("real mode: motion mode released")

    # ----- DDS input and controller state snapshot -----

    def on_lowstate(self, msg: LowState_) -> None:
        with self.lock:
            self.mode_machine = int(msg.mode_machine)
            self.mode_pr = int(msg.mode_pr)

            for i in range(self.num_joints):
                state = msg.motor_state[i]
                self.q[i] = float(state.q)
                self.dq[i] = float(state.dq)

            np.take(self.q, self.raw_to_policy, out=self.q_policy)
            np.take(self.dq, self.raw_to_policy, out=self.dq_policy)
            self.quat[:] = np.asarray(msg.imu_state.quaternion[:4], dtype=np.float64)
            self.gyro[:] = np.asarray(msg.imu_state.gyroscope[:3], dtype=np.float64)
            self.remote.set(msg.wireless_remote)
            self.command[:] = [self.remote.ly, -self.remote.lx, -self.remote.rx]
            self.has_low_state = True

    def observation(self) -> ObservationContext:
        with self.lock:
            return ObservationContext(
                q=self.q_policy.copy(),
                dq=self.dq_policy.copy(),
                quat=self.quat.copy(),
                gyro=self.gyro.copy(),
                command=self.command.copy(),
            )

    # ----- State-machine controls -----

    def consume_button(self, name: str) -> bool:
        with self.lock:
            return self.remote.consume(name)

    def transition(self, state: str) -> None:
        if self.state == state:
            return
        self.state = state
        self.state_enter_t = time.perf_counter()
        self.policy.reset()
        if state == MOVE_TO_DEFAULT:
            with self.lock:
                self.move_start_q[:] = self.q_policy
        log(f"state -> {state}")

    # ----- DDS output -----

    def send_joint_cmd(
        self,
        target_q: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        *,
        enable: bool = True,
        target_dq: np.ndarray | None = None,
        tau_ff: np.ndarray | None = None,
    ) -> None:
        target_dq = self.zero if target_dq is None else target_dq
        tau_ff = self.zero if tau_ff is None else tau_ff
        with self.lock:
            self.low_cmd.mode_pr = int(self.mode_pr)
            self.low_cmd.mode_machine = int(self.mode_machine)

        # Clear all motors first so any joints outside num_joints stay disabled.
        for cmd in self.low_cmd.motor_cmd:
            cmd.mode = 0
            cmd.q = cmd.dq = cmd.tau = cmd.kp = cmd.kd = 0.0

        for i in range(self.num_joints):
            cmd = self.low_cmd.motor_cmd[i]
            cmd.mode = 1 if enable else 0
            cmd.q = float(target_q[i])
            cmd.dq = float(target_dq[i])
            cmd.tau = float(tau_ff[i])
            cmd.kp = float(kp[i])
            cmd.kd = float(kd[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_pub.Write(self.low_cmd)

    # ----- Per-state control steps -----

    def step_zero_torque(self) -> None:
        if self.consume_button("A"):
            self.transition(MOVE_TO_DEFAULT)
            return

        with self.lock:
            q = self.q.copy()
        self.send_joint_cmd(q, self.zero, self.zero_torque_kd)

    def step_move_to_default(self) -> None:
        if self.consume_button("X"):
            self.transition(ZERO_TORQUE)
            return

        ratio = np.clip(
            (time.perf_counter() - self.state_enter_t) / MOVE_TO_DEFAULT_TIME,
            0.0,
            1.0,
        )
        self.target_policy[:] = self.default_q_policy
        self.target_policy -= self.move_start_q
        self.target_policy *= ratio
        self.target_policy += self.move_start_q
        self.send_joint_cmd(self.write_policy_to_raw(self.target_policy), self.kp, self.kd)

        if ratio >= 1.0:
            self.transition(DEFAULT_QPOS)

    def step_default_qpos(self) -> None:
        if self.consume_button("X"):
            self.transition(ZERO_TORQUE)
            return
        if self.consume_button("Start"):
            self.transition(RUN_POLICY)
            return

        self.send_joint_cmd(self.default_q_raw, self.kp, self.kd)

    def step_policy(self) -> None:
        if self.consume_button("X"):
            self.transition(ZERO_TORQUE)
            return

        # Policy returns targets in policy order; LowCmd must be written in raw motor order.
        target_policy = self.policy.compute_target_q(self.observation())
        self.send_joint_cmd(self.write_policy_to_raw(target_policy), self.kp, self.kd)

    def step(self) -> None:
        if self.state == ZERO_TORQUE:
            self.step_zero_torque()
        elif self.state == MOVE_TO_DEFAULT:
            self.step_move_to_default()
        elif self.state == DEFAULT_QPOS:
            self.step_default_qpos()
        elif self.state == RUN_POLICY:
            self.step_policy()

    # ----- Main loop and cleanup -----

    def spin(self) -> None:
        log(
            f"robot={self.robot} mode={self.config.mode} joints={self.num_joints} "
            f"topics: lowstate={self.lowstate_topic}, lowcmd={self.lowcmd_topic}"
        )
        if self.config.mode == "sim":
            log("sim keymap: b->A, m->Start, r->X + reset sim state")
        log("A: zero torque -> default pose, Start: default pose -> run policy, X: back to zero torque")
        log("waiting for lowstate...")

        timer = LoopTimer(float(self.policy.control_step_dt))
        last_log = time.perf_counter()

        while self.alive:
            with self.lock:
                ready = self.has_low_state
                command = self.command.copy()
            if ready:
                self.step()

            now = time.perf_counter()
            if now - last_log >= 1.0:
                log(
                    f"state={self.state} "
                    f"cmd=({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f})"
                )
                last_log = now
            timer.sleep()

    def close(self, *_args) -> None:
        self.alive = False

    def cleanup(self) -> None:
        if self.cleanup_done:
            return
        self.cleanup_done = True
        self.alive = False
        self.lowstate_sub.Close()
        self.lowcmd_pub.Close()


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Reusable Unitree controller for sim or real robot.")
    parser.add_argument("--mode", choices=("real", "sim"), default=MODE)
    parser.add_argument("--net", default=LOCAL_NET, help="DDS network interface. Use lo for local sim.")
    parser.add_argument("--robot", help="Robot name for logs. Defaults to controller.yaml robot.")
    parser.add_argument("--ckpt", type=Path, help="Checkpoint directory containing controller.yaml and policy.yaml.")
    parser.add_argument("--deploy-yaml", type=Path, help="Compatibility alias: use the parent directory as --ckpt.")
    args = parser.parse_args()
    if args.ckpt is None and args.deploy_yaml is None:
        parser.error("one of --ckpt or --deploy-yaml is required")

    ckpt_dir = args.ckpt or args.deploy_yaml.resolve().parent
    return RuntimeConfig(
        mode=args.mode,
        net=args.net,
        ckpt_dir=ckpt_dir,
        robot=args.robot,
    )


if __name__ == "__main__":
    config = parse_args()

    if config.mode == "real":
        print("WARNING: Please ensure there are no obstacles around the robot while running controller.py.")
        input("Press Enter to continue...")

    if config.net:
        ChannelFactoryInitialize(0, config.net)
    else:
        ChannelFactoryInitialize(0)

    controller = Controller(config)
    try:
        controller.spin()
    finally:
        controller.cleanup()
