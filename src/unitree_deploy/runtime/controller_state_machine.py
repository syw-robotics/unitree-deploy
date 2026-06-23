from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from unitree_deploy.config.defaults import (
    DAMPING_STATE,
    MOVE_TO_DEFAULT_STATE,
    MOVE_TO_DEFAULT_TIME,
    RUN_POLICY_STATE,
    WIRELESS_REMOTE_BUTTON_BITS,
)
from unitree_deploy.utils.yaml_utils import load_yaml


STATE_MACHINE_FILE = "state_machine.yaml"


def default_state_machine_config() -> dict[str, Any]:
    return {
        "initial": DAMPING_STATE,
        "states": {
            DAMPING_STATE: {
                "type": "damping",
                "transitions": {
                    "A": MOVE_TO_DEFAULT_STATE,
                },
            },
            MOVE_TO_DEFAULT_STATE: {
                "type": "move_to_target_pos",
                "target": "default_q",
                "duration": MOVE_TO_DEFAULT_TIME,
                "kp": "fixed_stand",
                "kd": "fixed_stand",
                "transitions": {
                    "Start": RUN_POLICY_STATE,
                    "X": DAMPING_STATE,
                },
            },
            RUN_POLICY_STATE: {
                "type": "policy",
                "transitions": {
                    "A": MOVE_TO_DEFAULT_STATE,
                    "X": DAMPING_STATE,
                },
            },
        },
    }


def resolve_state_machine_path(base_dir: Path, explicit_path: Path | None = None) -> Path | None:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()

    candidate = base_dir / STATE_MACHINE_FILE
    return candidate.resolve() if candidate.exists() else None


def load_state_machine_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_state_machine_config()
    config = load_yaml(path)
    if not isinstance(config, dict):
        raise TypeError(f"{path} must contain a mapping")
    return config


class ControllerState:
    state_type = "base"

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        self.name = name
        self.transitions = parse_transitions(config.get("transitions", {}), name)

    def on_enter(self, controller) -> None:
        pass

    def step(self, controller) -> str | None:
        return None

    def next_for_button(self, button: str) -> str | None:
        return self.transitions.get(button)

    def next_on_done(self) -> str | None:
        return self.transitions.get("done")


class DampingState(ControllerState):
    state_type = "damping"

    def step(self, controller) -> str | None:
        with controller.lock:
            q = controller.q.copy()
            kd_damping = controller.active_profile.kd_damping
        controller.send_joint_cmd(q, controller.zero, kd_damping)
        return None


class MoveToTargetPosState(ControllerState):
    state_type = "move_to_target_pos"

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        super().__init__(name, config)
        self.target_spec = config.get("target", "default_q")
        self.target_order = str(config.get("target_order", "sdk"))
        self.duration = float(config.get("duration", MOVE_TO_DEFAULT_TIME))
        self.kp_spec = config.get("kp", "fixed_stand")
        self.kd_spec = config.get("kd", "fixed_stand")
        self.allow_early_transitions = bool(config.get("allow_early_transitions", False))
        self.early_transition_events = parse_event_set(
            config.get("early_transition_events", ["X"]),
            self.name,
            "early_transition_events",
        )
        self.start_q: np.ndarray | None = None
        self.target_q_obs: np.ndarray | None = None
        self.command_q_obs: np.ndarray | None = None
        self.last_ratio = 0.0

    @property
    def is_done(self) -> bool:
        return self.start_q is not None and self.last_ratio >= 1.0

    def next_for_button(self, button: str) -> str | None:
        if self.allow_early_transitions or self.is_done or button in self.early_transition_events:
            return super().next_for_button(button)
        return None

    def on_enter(self, controller) -> None:
        profile = controller.active_profile
        with controller.lock:
            self.start_q = controller.q[profile.sdk_to_obs].copy()
        self.target_q_obs = resolve_target_q(controller, self.target_spec, self.target_order)
        self.command_q_obs = np.zeros_like(self.target_q_obs)
        self.last_ratio = 0.0

    def step(self, controller) -> str | None:
        if self.start_q is None or self.target_q_obs is None or self.command_q_obs is None:
            self.on_enter(controller)

        elapsed = time.perf_counter() - controller.state_enter_t
        ratio = 1.0 if self.duration <= 0.0 else float(np.clip(elapsed / self.duration, 0.0, 1.0))
        self.last_ratio = ratio
        self.command_q_obs[:] = self.target_q_obs
        self.command_q_obs -= self.start_q
        self.command_q_obs *= ratio
        self.command_q_obs += self.start_q

        controller.send_joint_cmd(
            controller.reorder_policy_to_sdk(self.command_q_obs),
            resolve_kp(controller, self.kp_spec),
            resolve_kd(controller, self.kd_spec),
        )

        if ratio >= 1.0:
            return self.next_on_done()
        return None


class PolicyState(ControllerState):
    state_type = "policy"

    def step(self, controller) -> str | None:
        profile = controller.active_profile
        observation = controller.observation()
        controller.last_policy_command[:] = observation.command
        policy_action = profile.policy.compute_target_q(observation)
        controller.send_joint_cmd(
            controller.reorder_policy_to_sdk(policy_action),
            profile.kp_policy,
            profile.kd_policy,
        )
        return None


STATE_TYPES = {
    DampingState.state_type: DampingState,
    MoveToTargetPosState.state_type: MoveToTargetPosState,
    PolicyState.state_type: PolicyState,
}


class ControllerStateMachine:
    def __init__(self, controller, config: Mapping[str, Any]) -> None:
        self.controller = controller
        self.initial = str(config.get("initial", DAMPING_STATE))
        self.states = build_states(config)
        if self.initial not in self.states:
            raise KeyError(f"initial state {self.initial!r} is not defined")
        validate_transitions(self.states)
        self.current_name = self.initial
        controller.state = self.initial
        controller.state_enter_t = time.perf_counter()
        self.states[self.current_name].on_enter(controller)

    @property
    def current(self) -> ControllerState:
        return self.states[self.current_name]

    def transition(self, state: str, *, force: bool = False) -> None:
        if state not in self.states:
            raise KeyError(f"unknown controller state {state!r}")
        if self.current_name == state and not force:
            return
        self.current_name = state
        self.controller.state = state
        self.controller.state_enter_t = time.perf_counter()
        self.controller.active_profile.policy.reset()
        self.current.on_enter(self.controller)
        self.controller.log(f"state -> {state}")

    def step(self) -> None:
        state = self.current
        for button in WIRELESS_REMOTE_BUTTON_BITS:
            next_state = state.next_for_button(button)
            if next_state and self.controller.button_pressed(button):
                self.transition(next_state)
                return

        next_state = state.step(self.controller)
        if next_state:
            self.transition(next_state)


def build_states(config: Mapping[str, Any]) -> dict[str, ControllerState]:
    specs = config.get("states")
    if not isinstance(specs, dict) or not specs:
        raise ValueError("state_machine.yaml must define a non-empty 'states' mapping")

    states: dict[str, ControllerState] = {}
    for name, spec in specs.items():
        if not isinstance(spec, dict):
            raise TypeError(f"state {name!r} must be a mapping")
        state_type = str(spec.get("type", ""))
        try:
            state_cls = STATE_TYPES[state_type]
        except KeyError as exc:
            known = ", ".join(sorted(STATE_TYPES))
            raise KeyError(f"unknown state type {state_type!r}; known types: {known}") from exc
        states[str(name)] = state_cls(str(name), spec)
    return states


def parse_transitions(value: Any, state_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"transitions for state {state_name!r} must be a mapping")
    transitions = {str(event): str(target) for event, target in value.items() if target is not None}
    known_events = set(WIRELESS_REMOTE_BUTTON_BITS) | {"done"}
    unknown_events = sorted(set(transitions) - known_events)
    if unknown_events:
        raise KeyError(f"state {state_name!r} has unknown transition events: {unknown_events}")
    return transitions


def parse_event_set(value: Any, state_name: str, field: str) -> set[str]:
    if isinstance(value, str):
        events = {value}
    else:
        try:
            events = {str(item) for item in value}
        except TypeError as exc:
            raise TypeError(f"{field} for state {state_name!r} must be a string or iterable") from exc

    known_events = set(WIRELESS_REMOTE_BUTTON_BITS) | {"done"}
    unknown_events = sorted(events - known_events)
    if unknown_events:
        raise KeyError(f"state {state_name!r} has unknown {field}: {unknown_events}")
    return events


def validate_transitions(states: Mapping[str, ControllerState]) -> None:
    known_states = set(states)
    for state in states.values():
        unknown_targets = sorted(set(state.transitions.values()) - known_states)
        if unknown_targets:
            raise KeyError(f"state {state.name!r} transitions to unknown states: {unknown_targets}")


def resolve_target_q(controller, spec: Any, target_order: str) -> np.ndarray:
    profile = controller.active_profile
    if isinstance(spec, str):
        if spec in ("default_q", "default_qpos"):
            return profile.default_q_obs.copy()
        if spec not in profile.policy.config:
            raise KeyError(f"unknown target {spec!r}; define it in policy.yaml or use default_q")
        values = profile.policy.config[spec]
    else:
        values = spec

    target = np.asarray(values, dtype=np.float32).reshape(-1)
    if target.size != controller.num_joints:
        raise ValueError(f"target {spec!r} has {target.size} joints, expected {controller.num_joints}")
    if target_order == "sdk":
        return target[profile.sdk_to_obs].copy()
    if target_order == "obs":
        return target.copy()
    raise ValueError(f"target_order must be 'sdk' or 'obs', got {target_order!r}")


def resolve_kp(controller, spec: Any) -> np.ndarray:
    profile = controller.active_profile
    if spec == "fixed_stand":
        return profile.kp_fixed_stand
    if spec == "policy":
        return profile.kp_policy
    return gain_array(controller, spec, "kp")


def resolve_kd(controller, spec: Any) -> np.ndarray:
    profile = controller.active_profile
    if spec == "fixed_stand":
        return profile.kd_fixed_stand
    if spec == "policy":
        return profile.kd_policy
    if spec == "damping":
        return profile.kd_damping
    return gain_array(controller, spec, "kd")


def gain_array(controller, spec: Any, label: str) -> np.ndarray:
    if isinstance(spec, str):
        config = controller.active_profile.policy.config
        if spec not in config:
            raise KeyError(f"unknown {label} gain {spec!r}; define it in policy.yaml or use a built-in gain")
        values = config[spec]
    else:
        values = spec

    gain = np.asarray(values, dtype=np.float64).reshape(-1)
    if gain.size != controller.num_joints:
        raise ValueError(f"{label} gain has {gain.size} joints, expected {controller.num_joints}")
    return gain
