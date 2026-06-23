import argparse
import signal
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from unitree_deploy.config.defaults import (
    DAMPING_STATE,
    DEFAULT_MODE,
    DEFAULT_NET,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
    RUN_POLICY_STATE,
    WIRELESS_REMOTE_BUTTON_BITS,
    sim_key_for_button,
)
from unitree_deploy.obs.observation import ObservationContext
from unitree_deploy.robot_model.robot_config import DEFAULT_ROBOT
from unitree_deploy.runtime.multi_ckpt import PolicyManager
from unitree_deploy.runtime.controller_state_machine import (
    ControllerStateMachine,
    load_state_machine_config,
    resolve_state_machine_path,
)
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_deploy.utils.terminal_status import ComponentConsole


console = ComponentConsole("controller", "bright_blue")


def log(message: str) -> None:
    console.log(message)


def status(fields) -> None:
    console.status(fields)


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
    multi_ckpt: Path | None = None
    state_machine: Path | None = None


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
        self.ckpt_dir = config.ckpt_dir.expanduser().resolve()
        self.policy_manager = PolicyManager.load(self.ckpt_dir, config.multi_ckpt)
        self.robot = config.robot or self.active_profile.policy.config.get("robot", DEFAULT_ROBOT)

        self.lowcmd_topic = LOWCMD_TOPIC
        self.lowstate_topic = LOWSTATE_TOPIC
        self.sdk_joint_order = list(self.active_profile.sdk_joint_order)
        self.obs_joint_order = list(self.active_profile.obs_joint_order)
        self.num_joints = len(self.sdk_joint_order)
        self.raw_command = np.zeros(3, dtype=np.float64)
        self.zero = np.zeros(self.num_joints, dtype=np.float64)
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
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.gyro = np.zeros(3, dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float64)
        self.last_policy_command = np.zeros(3, dtype=np.float64)
        self.remote = RemoteCommand()
        self.log = log

        if config.mode == "real":
            self.enter_debug_mode()

        state_machine_base = config.multi_ckpt.parent if config.multi_ckpt else self.ckpt_dir
        state_machine_path = resolve_state_machine_path(state_machine_base, config.state_machine)
        state_machine_config = load_state_machine_config(state_machine_path)
        self.state_machine_path = state_machine_path

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.lowstate_sub = ChannelSubscriber(self.lowstate_topic, LowState_)
        self.lowstate_sub.Init(self.on_lowstate, 1)
        self.lowcmd_pub = ChannelPublisher(self.lowcmd_topic, LowCmd_)
        self.lowcmd_pub.Init()

        signal.signal(signal.SIGINT, self.close)
        signal.signal(signal.SIGTERM, self.close)

        self.state_machine = ControllerStateMachine(self, state_machine_config)
        self.state = self.state_machine.current_name

    @property
    def active_profile(self):
        return self.policy_manager.active

    @property
    def active_profile_name(self) -> str:
        return self.policy_manager.active_name

    def reorder_policy_to_sdk(self, value: np.ndarray) -> np.ndarray:
        np.take(value, self.active_profile.obs_to_sdk, out=self.target_sdk)
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

            self.quat[:] = np.asarray(msg.imu_state.quaternion[:4], dtype=np.float64)
            self.gyro[:] = np.asarray(msg.imu_state.gyroscope[:3], dtype=np.float64)
            self.remote.set(msg.wireless_remote)
            self.raw_command[:] = [self.remote.ly, -self.remote.lx, -self.remote.rx]
            profile = self.active_profile
            np.clip(self.raw_command, profile.command_min, profile.command_max, out=self.command)
            self.has_low_state = True

    def observation(self) -> ObservationContext:
        with self.lock:
            profile = self.active_profile
            return ObservationContext(
                q=self.q[profile.sdk_to_obs].copy(),
                dq=self.dq[profile.sdk_to_obs].copy(),
                quat=self.quat.copy(),
                gyro=self.gyro.copy(),
                command=self.command.copy(),
            )

    # ----- State-machine controls -----

    def button_pressed(self, name: str) -> bool:
        with self.lock:
            return self.remote.button_pressed(name)

    def transition(self, state: str, *, force: bool = False) -> None:
        self.state_machine.transition(state, force=force)

    def switch_to_policy(self, name: str) -> bool:
        if name == self.active_profile_name:
            return False
        if self.state not in self.policy_manager.switch.only_when:
            allowed = ", ".join(sorted(self.policy_manager.switch.only_when))
            log(f"policy switch ignored in state={self.state}; allowed states: {allowed}")
            return False

        with self.lock:
            profile = self.policy_manager.switch_to(name)
            self.obs_joint_order = list(profile.obs_joint_order)
            np.clip(self.raw_command, profile.command_min, profile.command_max, out=self.command)
        log(f"policy -> {name} ({profile.policy_yaml_path})")
        if self.policy_manager.switch.on_switch:
            self.transition(self.policy_manager.switch.on_switch, force=True)
        return True

    def switch_to_next_policy(self) -> bool:
        if not self.policy_manager.switch_allowed(self.state):
            allowed = ", ".join(sorted(self.policy_manager.switch.only_when))
            log(f"policy switch ignored in state={self.state}; allowed states: {allowed}")
            return False

        with self.lock:
            profile = self.policy_manager.switch_next()
            self.obs_joint_order = list(profile.obs_joint_order)
            np.clip(self.raw_command, profile.command_min, profile.command_max, out=self.command)
        log(f"policy -> {profile.name} ({profile.policy_yaml_path})")
        if self.policy_manager.switch.on_switch:
            self.transition(self.policy_manager.switch.on_switch, force=True)
        return True

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

    # ----- State dispatch -----

    def step(self) -> None:
        if self.policy_manager.switch.enabled and self.button_pressed(self.policy_manager.switch.button):
            self.switch_to_next_policy()
            return

        self.state_machine.step()

    # ----- Main loop and cleanup -----

    def spin(self) -> None:
        log(
            f"robot={self.robot} mode={self.config.mode} joints={self.num_joints} "
            f"topics: lowstate={self.lowstate_topic}, lowcmd={self.lowcmd_topic}"
        )
        log(
            f"policy={self.active_profile_name} "
            f"available={','.join(self.policy_manager.profiles)}"
        )
        log(f"state_machine={self.state_machine_path or 'default'}")
        if self.config.mode == "sim":
            switch_hint = (
                f", {sim_key_for_button(self.policy_manager.switch.button)} -> switch policy"
                if self.policy_manager.switch.enabled
                else ""
            )
            log(
                f"sim keymap: {sim_key_for_button('A')} -> A, "
                f"{sim_key_for_button('Start')} -> Start, "
                f"{sim_key_for_button('X')} -> Damping{switch_hint}, R -> reset sim"
            )
        control_hint = (
            "A: damping/policy -> default pose, "
            "Start: default pose -> run policy, "
            "X: back to zero torque"
        )
        if self.policy_manager.switch.enabled:
            control_hint += f", {self.policy_manager.switch.button}: switch policy"
        log(control_hint)
        log("waiting for lowstate...")

        timer = LoopTimer(float(self.active_profile.policy.policy_step_dt))
        last_log = time.perf_counter()

        while self.alive:
            with self.lock:
                ready = self.has_low_state
            if ready:
                self.step()

            now = time.perf_counter()
            if now - last_log >= 1.0:
                command = self.last_policy_command
                state_style = "green" if self.state == RUN_POLICY_STATE else "yellow"
                status(
                    [
                        ("state", self.state, state_style),
                        ("policy", self.active_profile_name, "cyan"),
                        ("cmd", f"{command[0]:+.2f} {command[1]:+.2f} {command[2]:+.2f}", "white"),
                        ("lowstate", "yes" if ready else "no", "green" if ready else "red"),
                    ]
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
        console.stop()
        self.lowstate_sub.Close()
        self.lowcmd_pub.Close()


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Reusable Unitree controller for sim or real robot.")
    parser.add_argument("--mode", choices=("real", "sim"), default=DEFAULT_MODE)
    parser.add_argument("--net", default=DEFAULT_NET, help="DDS network interface. Use lo for local sim.")
    parser.add_argument("--robot", help="Robot name for logs. Defaults to controller.yaml robot.")
    parser.add_argument("--ckpt", type=Path, help="Checkpoint directory containing policy.yaml.")
    parser.add_argument(
        "--multi-ckpt",
        type=Path,
        help="YAML manifest containing multiple ckpt directories.",
    )
    parser.add_argument(
        "--state-machine",
        type=Path,
        help="Optional YAML state machine. Defaults to state_machine.yaml beside the ckpt or multi-ckpt YAML.",
    )
    args = parser.parse_args()
    if args.ckpt is None and args.multi_ckpt is None:
        parser.error("one of --ckpt or --multi-ckpt is required")

    ckpt_dir = args.ckpt or args.multi_ckpt.expanduser().resolve().parent
    return RuntimeConfig(
        mode=args.mode,
        net=args.net,
        ckpt_dir=ckpt_dir,
        robot=args.robot,
        multi_ckpt=args.multi_ckpt,
        state_machine=args.state_machine,
    )


def main() -> None:
    config = parse_args()

    if config.mode == "real":
        console.log(
            "WARNING: Please ensure there are no obstacles around the robot while running controller.py.",
            style="bold red",
        )
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
