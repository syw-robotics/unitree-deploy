from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import onnxruntime as ort
import yaml

from ..obs.registry import BUILTIN_OBSERVATION_TYPES
from ..obs.observation import (
    ObservationBase,
    ObservationContext,
    ObservationGroup,
    PreviousActionObservation,
)

ObservationRegistry: TypeAlias = Mapping[str, type[ObservationBase]]


class BasePolicy:
    """Base ONNX policy wrapper.

    Data flow:
      ObservationContext -> history-stacked observation vector -> ONNX action
      -> controlled joints -> full robot target_q in policy joint order

    Observation layout is configured in policy.yaml, not by subclassing the
    policy wrapper.
    """

    def __init__(
        self,
        policy_yaml_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        observation_types: ObservationRegistry | None = None,
    ) -> None:
        self.policy_yaml_path = Path(policy_yaml_path).expanduser().resolve()
        with self.policy_yaml_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.observation_types = dict(BUILTIN_OBSERVATION_TYPES)
        if observation_types:
            self.observation_types.update(observation_types)

        self._load_action_config()
        self._load_policy_config()
        self._load_observations()

        self.model_path = (self.policy_yaml_path.parent / self.config["policy_path"]).resolve()
        session_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=session_providers)

    # ----- Action, joint-order, and timing config -----

    def _load_action_config(self) -> None:
        self.obs_joint_order = list(self.config.get("obs_joint_order", []))
        self.action_joint_order = list(self.config.get("action_joint_order", []))
        self.sdk_joint_order = list(self.config.get("sdk_joint_order", self.obs_joint_order))
        if not self.obs_joint_order:
            raise KeyError("policy.yaml must define obs_joint_order")
        if not self.action_joint_order:
            raise KeyError("policy.yaml must define action_joint_order")

        self.obs_joint_index = {name: idx for idx, name in enumerate(self.obs_joint_order)}
        self.default_joint_pos = np.asarray(
            self.config["default_qpos"],
            dtype=np.float32,
        )
        if self.default_joint_pos.size != len(self.obs_joint_order):
            raise ValueError(
                f"default_qpos has {self.default_joint_pos.size} values, "
                f"expected {len(self.obs_joint_order)} from obs_joint_order"
            )
        self.physics_dt = float(self.config["physics_dt"])
        self.decimation = int(self.config["decimation"])

        self.action_dim = int(self.config["action_dim"])
        self.action_clip = float(self.config["action_clip"])
        if self.action_dim != len(self.action_joint_order):
            raise ValueError(
                f"action_dim={self.action_dim} does not match action_joint_order "
                f"length={len(self.action_joint_order)}"
            )
        if len(self.config["action_scale"]) != len(self.action_joint_order):
            raise ValueError(
                f"action_scale has {len(self.config['action_scale'])} values, "
                f"expected {len(self.action_joint_order)} from action_joint_order"
            )

        self.action_to_obs_indices = np.asarray(
            [self.obs_joint_index[name] for name in self.action_joint_order],
            dtype=np.int64,
        )

        self.default_joint_pos_action = self.default_joint_pos[self.action_to_obs_indices]

        action_scaling_cfg = {
            name: value for name, value in zip(self.action_joint_order, self.config["action_scale"])
        }
        self.action_scaling = np.asarray(
            [action_scaling_cfg[name] for name in self.action_joint_order],
            dtype=np.float32,
        )
        self.action = np.zeros(len(self.action_joint_order), dtype=np.float32)
        self.target_q = self.default_joint_pos

    # ----- ONNX input/output names -----

    def _load_policy_config(self) -> None:
        self.input_name = self.config["policy_input_name"]
        self.action_output_name = self.config["policy_output_name"]

    # ----- Observation layout -----

    def _load_observations(self) -> None:
        observations = []
        self.previous_action_observation = None

        for observation_spec in self.config["observations"]:
            observation = self._build_observation(observation_spec)
            observations.append(observation)
            if isinstance(observation, PreviousActionObservation):
                self.previous_action_observation = observation

        if self.previous_action_observation is None:
            raise ValueError("observations must include type: prev_action")
        self.observation = ObservationGroup(observations)

    def _build_observation(self, observation_spec: dict) -> ObservationBase:
        observation_type = observation_spec["type"]
        kwargs: dict[str, object] = {"history_len": int(observation_spec["history_len"])}

        if observation_type == "command":
            kwargs["command_range"] = observation_spec["command_range"]
        elif observation_type in ("joint_pos", "joint_vel"):
            kwargs["controlled_joint_indices"] = self.action_to_obs_indices
        elif observation_type == "prev_action":
            kwargs["action_dim"] = self.action_dim
        else:
            params = observation_spec.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise TypeError(f"observation params for {observation_type!r} must be a mapping")
            kwargs.update(params)

        try:
            observation_cls = self.observation_types[observation_type]
        except KeyError as exc:
            known_types = ", ".join(sorted(self.observation_types))
            raise KeyError(f"unknown observation type {observation_type!r}; known types: {known_types}") from exc

        return observation_cls(**kwargs)

    def reset(self) -> None:
        self.observation.reset()

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        self.observation.update(context)
        obs_vector = self.observation.compute().astype(np.float32, copy=False)
        outputs = self.session.run(
            [self.action_output_name],
            {self.input_name: obs_vector[None, :]},
        )
        policy_action = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        self.action[:] = policy_action
        self.previous_action_observation.record_action(self.action)
        np.clip(self.action, -self.action_clip, self.action_clip, out=self.action)

        # Return full robot targets in obs_joint_order; the controller converts to sdk_joint_order.
        self.target_q[:] = self.default_joint_pos
        self.target_q[self.action_to_obs_indices] = (
            self.default_joint_pos_action
            + self.action_scaling * self.action
        )
        return self.target_q
