import argparse
import signal
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from unitree_deploy.config.defaults import (
    DEFAULT_MODE,
    DEFAULT_NET,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
    MOVE_TO_DEFAULT_STATE,
    MOVE_TO_DEFAULT_TIME,
    RUN_POLICY_STATE,
    WIRELESS_REMOTE_BUTTON_BITS,
    DAMPING_STATE,
    sim_key_for_button,
)
from unitree_deploy.obs.observation import ObservationContext
from unitree_deploy.policy.base_policy import load_policy
from unitree_deploy.robot_model.robot_config import DEFAULT_ROBOT
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


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
        self.buttons = {name: False for name in WIRELESS_REMOTE_BUTTON_BITS}
        self.pressed_edges: set[str] = set()

    def set(self, wireless_remote) -> None:
        payload = bytes(wireless_remote)
        self.lx = struct.unpack("<f", payload[4:8])[0]
        self.rx = struct.unpack("<f", payload[8:12])[0]
        self.ry = struct.unpack("<f", payload[12:16])[0]
        self.ly = struct.unpack("<f", payload[20:24])[0]

        for name, (byte_i, bit_i) in WIRELESS_REMOTE_BUTTON_BITS.items():
            pressed = bool((int(wireless_remote[byte_i]) >> bit_i) & 1)
            if pressed and not self.buttons[name]:
                self.pressed_edges.add(name)
            self.buttons[name] = pressed

    def button_pressed(self, name: str) -> bool:
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
        self.policy = load_policy(self.ckpt_dir / "policy.yaml")
        self.robot = config.robot or self.policy.config.get("robot", DEFAULT_ROBOT)

        self.lowcmd_topic = LOWCMD_TOPIC
        self.lowstate_topic = LOWSTATE_TOPIC
        self.sdk_joint_order = list(self.policy.sdk_joint_order)
        self.obs_joint_order = list(self.policy.obs_joint_order)
        self.num_joints = len(self.sdk_joint_order)
        self.check_joint_config()

        self.sdk_to_obs = self.reorder_indices(self.sdk_joint_order, self.obs_joint_order)
        self.obs_to_sdk = self.reorder_indices(self.obs_joint_order, self.sdk_joint_order)
        self.kp_policy = self.gain_array("kp_policy", legacy_keys=("kp", "kps_real"))
        self.kd_policy = self.gain_array("kd_policy", legacy_keys=("kd", "kds_real"))
        self.kp_fixed_stand = self.gain_array("kp_fixed_stand", fallback=self.kp_policy)
        self.kd_fixed_stand = self.gain_array("kd_fixed_stand", fallback=self.kd_policy)
        self.kd_damping = self.gain_array(
            "kd_damping",
            fallback=np.ones(self.num_joints, dtype=np.float64),
        )
        self.command_min, self.command_max = self.command_range()
        self.raw_command = np.zeros(3, dtype=np.float64)
        self.zero = np.zeros(self.num_joints, dtype=np.float64)
        self.default_q_obs = self.policy.default_joint_pos
        self.default_q_sdk = self.default_q_obs[self.obs_to_sdk]
        self.target_obs = np.zeros(self.num_joints, dtype=np.float32)
        self.target_sdk = np.zeros(self.num_joints, dtype=np.float32)

        self.lock = threading.Lock()
        self.alive = True
        self.cleanup_done = False
        self.has_low_state = False
        self.mode_machine = 0
        self.mode_pr = 0
        self.state = DAMPING_STATE
        self.state_enter_t = time.perf_counter()

        self.q = np.zeros(self.num_joints, dtype=np.float64)
        self.dq = np.zeros(self.num_joints, dtype=np.float64)
        self.q_obs = np.zeros(self.num_joints, dtype=np.float64)
        self.dq_obs = np.zeros(self.num_joints, dtype=np.float64)
        self.move_start_q = np.zeros(self.num_joints, dtype=np.float32)
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

        self._init_handlers()

    # ----- Config and joint-order helpers -----

    @staticmethod
    def reorder_indices(source: list[str], target: list[str]) -> np.ndarray:
        source_index = {name: i for i, name in enumerate(source)}
        return np.asarray([source_index[name] for name in target], dtype=np.int64)

    def check_joint_config(self) -> None:
        if len(self.obs_joint_order) != self.num_joints:
            raise ValueError(
                f"joint count mismatch: {len(self.obs_joint_order)} obs joints, "
                f"{self.num_joints} sdk joints"
            )
        if set(self.obs_joint_order) != set(self.sdk_joint_order):
            missing_in_sdk = sorted(set(self.obs_joint_order) - set(self.sdk_joint_order))
            missing_in_obs = sorted(set(self.sdk_joint_order) - set(self.obs_joint_order))
            raise ValueError(
                f"obs_joint_order and sdk_joint_order must contain the same joints; "
                f"missing_in_sdk={missing_in_sdk}, missing_in_obs={missing_in_obs}"
            )
        if len(self.policy.default_joint_pos) != self.num_joints:
            raise ValueError(
                f"policy default_qpos has {len(self.policy.default_joint_pos)} joints, "
                f"controller has {self.num_joints}"
            )

    def gain_array(
        self,
        key: str,
        *,
        legacy_keys: tuple[str, ...] = (),
        fallback: np.ndarray | None = None,
    ) -> np.ndarray:
        config_key = next(
            (candidate for candidate in (key, *legacy_keys) if candidate in self.policy.config),
            None,
        )
        if config_key is None:
            if fallback is None:
                names = ", ".join(repr(name) for name in (key, *legacy_keys))
                raise KeyError(f"policy.yaml must define one of: {names}")
            values = np.asarray(fallback, dtype=np.float64).reshape(-1).copy()
            config_key = key
        else:
            values = np.asarray(self.policy.config[config_key], dtype=np.float64).reshape(-1)

        if values.size != self.num_joints:
            raise ValueError(f"{config_key} has {values.size} values, expected {self.num_joints}")
        return values

    def command_range(self) -> tuple[np.ndarray, np.ndarray]:
        for observation_spec in self.policy.config["observations"]:
            if observation_spec["type"] != "command":
                continue
            command_range = np.asarray(
                observation_spec["command_range"],
                dtype=np.float64,
            )
            if command_range.shape != (3, 2):
                raise ValueError(
                    f"command_range must have shape (3, 2), got {command_range.shape}"
                )
            return command_range[:, 0], command_range[:, 1]
        raise KeyError("policy.yaml observations must include type: command")

    def write_obs_to_sdk(self, value: np.ndarray) -> np.ndarray:
        np.take(value, self.obs_to_sdk, out=self.target_sdk)
        return self.target_sdk

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

            np.take(self.q, self.sdk_to_obs, out=self.q_obs)
            np.take(self.dq, self.sdk_to_obs, out=self.dq_obs)
            self.quat[:] = np.asarray(msg.imu_state.quaternion[:4], dtype=np.float64)
            self.gyro[:] = np.asarray(msg.imu_state.gyroscope[:3], dtype=np.float64)
            self.remote.set(msg.wireless_remote)
            self.raw_command[:] = [self.remote.ly, -self.remote.lx, -self.remote.rx]
            np.clip(self.raw_command, self.command_min, self.command_max, out=self.command)
            self.has_low_state = True

    def observation(self) -> ObservationContext:
        with self.lock:
            return ObservationContext(
                q=self.q_obs.copy(),
                dq=self.dq_obs.copy(),
                quat=self.quat.copy(),
                gyro=self.gyro.copy(),
                command=self.command.copy(),
            )

    # ----- State-machine controls -----

    def button_pressed(self, name: str) -> bool:
        with self.lock:
            return self.remote.button_pressed(name)

    def transition(self, state: str) -> None:
        if self.state == state:
            return
        self.state = state
        self.state_enter_t = time.perf_counter()
        self.policy.reset()
        if state == MOVE_TO_DEFAULT_STATE:
            with self.lock:
                self.move_start_q[:] = self.q_obs
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

    def step_damping(self) -> None:
        if self.button_pressed("A"):
            self.transition(MOVE_TO_DEFAULT_STATE)
            return

        with self.lock:
            q = self.q.copy()
        self.send_joint_cmd(q, self.zero, self.kd_damping)

    def step_move_to_default(self) -> None:
        if self.button_pressed("Start"):
            if (time.perf_counter() - self.state_enter_t) >= MOVE_TO_DEFAULT_TIME:
                self.transition(RUN_POLICY_STATE)
                return

        ratio = np.clip(
            (time.perf_counter() - self.state_enter_t) / MOVE_TO_DEFAULT_TIME,
            0.0,
            1.0,
        )
        self.target_obs[:] = self.default_q_obs
        self.target_obs -= self.move_start_q
        self.target_obs *= ratio
        self.target_obs += self.move_start_q
        self.send_joint_cmd(
            self.write_obs_to_sdk(self.target_obs),
            self.kp_fixed_stand,
            self.kd_fixed_stand,
        )

    def step_policy(self) -> None:
        # Policy returns targets in policy order; LowCmd must be written in raw motor order.
        target_obs = self.policy.compute_target_q(self.observation())
        self.send_joint_cmd(self.write_obs_to_sdk(target_obs), self.kp_policy, self.kd_policy)

    # ----- State dispatch -----

    _STATE_HANDLERS: dict[str, Callable[[], None]] = {}

    def _init_handlers(self) -> None:
        if Controller._STATE_HANDLERS:
            return
        Controller._STATE_HANDLERS.update({
            DAMPING_STATE: self.step_damping,
            MOVE_TO_DEFAULT_STATE: self.step_move_to_default,
            RUN_POLICY_STATE: self.step_policy,
        })

    def step(self) -> None:
        if self.state != DAMPING_STATE and self.button_pressed("X"):
            self.transition(DAMPING_STATE)
            return

        handler = self._STATE_HANDLERS.get(self.state)
        if handler is not None:
            handler()

    # ----- Main loop and cleanup -----

    def spin(self) -> None:
        log(
            f"robot={self.robot} mode={self.config.mode} joints={self.num_joints} "
            f"topics: lowstate={self.lowstate_topic}, lowcmd={self.lowcmd_topic}"
        )
        if self.config.mode == "sim":
            log(
                f"sim keymap: {sim_key_for_button('A')} -> A, "
                f"{sim_key_for_button('Start')} -> Start, "
                f"{sim_key_for_button('X')} -> Damping, R -> reset sim"
            )
        log("A: zero torque -> default pose, Start: default pose -> run policy, X: back to zero torque")
        log("waiting for lowstate...")

        timer = LoopTimer(float(self.policy.policy_step_dt))
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
    parser.add_argument("--mode", choices=("real", "sim"), default=DEFAULT_MODE)
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface. Use lo for local sim.")
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


def main() -> None:
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


if __name__ == "__main__":
    main()
