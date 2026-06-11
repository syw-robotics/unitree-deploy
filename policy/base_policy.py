from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from .observation import (
    CommandObservation,
    JointPositionObservation,
    JointVelocityObservation,
    ObservationBase,
    ObservationContext,
    ObservationGroup,
    PreviousActionObservation,
    ProjectedGravityObservation,
    RootAngularVelocityObservation,
)


class BasePolicy:
    """Base ONNX policy wrapper.

    Data flow:
      ObservationContext -> history-stacked observation vector -> ONNX action
      -> controlled joints -> full robot target_q in policy joint order

    Subclasses can extend ``OBSERVATION_TYPES`` or override ``_build_observation``
    when a policy needs custom observation terms.
    """

    OBSERVATION_TYPES = {
        "command": CommandObservation,
        "base_angvel": RootAngularVelocityObservation,
        "projected_gravity": ProjectedGravityObservation,
        "joint_pos": JointPositionObservation,
        "joint_vel": JointVelocityObservation,
        "prev_action": PreviousActionObservation,
    }

    def __init__(
        self,
        policy_yaml_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.policy_yaml_path = Path(policy_yaml_path).expanduser().resolve()
        with self.policy_yaml_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self._load_action_config()
        self._load_policy_config()
        self._load_observation_config()
        if self.observation.size != self.policy_input_dim:
            raise ValueError(
                f"policy_input_dim={self.policy_input_dim}, but observations produce {self.observation.size}"
            )

        self.model_path = (self.policy_yaml_path.parent / self.config["policy_path"]).resolve()
        session_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=session_providers)

    # ----- Action, joint-order, and timing config -----

    def _load_action_config(self) -> None:
        self.robot_joint_names = list(self.config["robot_joint_names"])
        self.robot_joint_index = {name: idx for idx, name in enumerate(self.robot_joint_names)}
        self.default_joint_pos = np.asarray(
            self.config["default_qpos"],
            dtype=np.float32,
        )
        self.physics_dt = float(self.config["physics_dt"])
        self.decimation = int(self.config["decimation"])

        self.action_dim = int(self.config["action_dim"])
        self.action_clip = float(self.config["action_clip"])
        self.policy_action_joint_names = list(self.config["action_joint_names"])
        self.controlled_joint_names = list(self.config["controlled_joint_names"])
        # ONNX output order can differ from the motor subset controlled by this deployment.
        self.policy_to_controlled_reorder = np.asarray(
            self.config["policy_to_controlled_reorder"],
            dtype=np.int64,
        )

        self.controlled_joint_indices = np.asarray(
            [self.robot_joint_index[name] for name in self.controlled_joint_names],
            dtype=np.int64,
        )

        self.default_joint_pos_controlled = self.default_joint_pos[self.controlled_joint_indices]

        action_scaling_cfg = {
            name: value for name, value in zip(self.policy_action_joint_names, self.config["action_scale"])
        }
        self.action_scaling_controlled = np.asarray(
            [action_scaling_cfg[name] for name in self.controlled_joint_names],
            dtype=np.float32,
        )
        self.controlled_action = np.zeros(len(self.controlled_joint_names), dtype=np.float32)
        self.target_q = self.default_joint_pos

    # ----- ONNX input/output names -----

    def _load_policy_config(self) -> None:
        self.input_name = self.config["policy_input_name"]
        self.policy_input_dim = int(self.config["policy_input_dim"])
        self.action_output_name = self.config["policy_output_name"]

    # ----- Observation layout -----

    def _load_observation_config(self) -> None:
        observations = []
        self.previous_action_observation = None

        for observation_config in self.config["observations"]:
            observation = self._build_observation(observation_config)
            observations.append(observation)
            if isinstance(observation, PreviousActionObservation):
                self.previous_action_observation = observation

        if self.previous_action_observation is None:
            raise ValueError("observations must include type: prev_action")
        self.observation = ObservationGroup(observations)

    def _build_observation(self, observation_config: dict) -> ObservationBase:
        observation_type = observation_config["type"]
        kwargs: dict[str, object] = {"history_len": int(observation_config["history_len"])}

        if observation_type == "command":
            kwargs["command_range"] = observation_config["command_range"]
        elif observation_type in ("joint_pos", "joint_vel"):
            kwargs["controlled_joint_indices"] = self.controlled_joint_indices
        elif observation_type == "prev_action":
            kwargs["action_dim"] = self.action_dim

        return self.OBSERVATION_TYPES[observation_type](**kwargs)

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
        np.take(policy_action, self.policy_to_controlled_reorder, out=self.controlled_action)
        self.previous_action_observation.record_action(self.controlled_action)
        np.clip(self.controlled_action, -self.action_clip, self.action_clip, out=self.controlled_action)

        # Return full robot targets so the controller can handle raw-order conversion in one place.
        self.target_q[:] = self.default_joint_pos
        self.target_q[self.controlled_joint_indices] = (
            self.default_joint_pos_controlled
            + self.action_scaling_controlled * self.controlled_action
        )
        return self.target_q
