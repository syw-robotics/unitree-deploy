from __future__ import annotations

from pathlib import Path

import numpy as np

from unitree_deploy.obs.observation import ObservationContext
from unitree_deploy.policy.base_policy import BasePolicy, ObservationRegistry


class MotionTrackingPolicy(BasePolicy):
    """Policy wrapper for mjlab motion-tracking ONNX exports.

    The mjlab tracking exporter bundles the reference motion inside the ONNX
    model. The model accepts ``obs`` and ``time_step`` and returns both actions
    and the reference frame for that time step.
    """

    def __init__(
        self,
        policy_yaml_path: str | Path,
        *,
        providers=None,
        observation_types: ObservationRegistry | None = None,
    ) -> None:
        self.time_step = 0
        self.motion_reference: dict[str, np.ndarray] = {}
        super().__init__(
            policy_yaml_path,
            providers=providers,
            observation_types=observation_types,
        )
        self.time_input_name = str(self.config.get("time_input_name", "time_step"))
        self.reference_output_names = list(
            self.config.get(
                "reference_output_names",
                [
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
            )
        )
        self._validate_onnx_contract()
        for observation in self.observation.observations:
            setattr(observation, "policy", self)
        self._zero_obs = np.zeros((1, self.observation.size), dtype=np.float32)
        self._time_input = np.zeros((1, 1), dtype=np.float32)
        self._refresh_motion_reference()

    def reset(self) -> None:
        self.time_step = 0
        super().reset()
        if hasattr(self, "session"):
            self._refresh_motion_reference()

    def _run_model(self, obs_vector: np.ndarray, output_names: list[str]):
        self._time_input[0, 0] = float(self.time_step)
        return self.session.run(
            output_names,
            {
                self.input_name: obs_vector.reshape(1, -1).astype(np.float32, copy=False),
                self.time_input_name: self._time_input,
            },
        )

    def _refresh_motion_reference(self) -> None:
        outputs = self._run_model(self._zero_obs, self.reference_output_names)
        self.motion_reference = {
            name: np.asarray(value, dtype=np.float32).squeeze(axis=0)
            for name, value in zip(self.reference_output_names, outputs)
        }

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        self._refresh_motion_reference()
        if self._observation_needs_prime:
            self.observation.prime(context)
            self._observation_needs_prime = False
        else:
            self.observation.update(context)

        obs_vector = self.observation.compute().astype(np.float32, copy=False)
        policy_action = np.asarray(
            self._run_model(obs_vector, [self.action_output_name])[0],
            dtype=np.float32,
        ).reshape(-1)
        self.action[:] = policy_action
        prev_action = self.action * self.action_scaling if self.obs_use_scaled_prev_action else self.action
        self.previous_action_observation.record_action(prev_action)
        if self.action_clip is not None:
            np.clip(self.action, -self.action_clip, self.action_clip, out=self.action)

        self.target_q[:] = self.default_joint_pos
        self.target_q[self.action_to_obs_indices] = (
            self.default_joint_pos_action
            + self.action_scaling * self.action
        )
        self.time_step += 1
        return self.target_q

    def _validate_onnx_contract(self) -> None:
        input_names = {input_info.name: input_info for input_info in self.session.get_inputs()}
        if self.input_name not in input_names:
            raise ValueError(f"ONNX model has no policy input {self.input_name!r}")
        if self.time_input_name not in input_names:
            raise ValueError(f"ONNX model has no time input {self.time_input_name!r}")

        obs_shape = input_names[self.input_name].shape
        if len(obs_shape) >= 2 and isinstance(obs_shape[1], int) and obs_shape[1] != self.observation.size:
            raise ValueError(
                f"ONNX obs input expects {obs_shape[1]} values, "
                f"policy.yaml builds {self.observation.size}"
            )

        output_names = {output_info.name: output_info for output_info in self.session.get_outputs()}
        required_outputs = [self.action_output_name, *self.reference_output_names]
        missing_outputs = [name for name in required_outputs if name not in output_names]
        if missing_outputs:
            raise ValueError(f"ONNX model missing outputs: {missing_outputs}")

        action_shape = output_names[self.action_output_name].shape
        if len(action_shape) >= 2 and isinstance(action_shape[1], int) and action_shape[1] != self.action_dim:
            raise ValueError(
                f"ONNX action output has dim {action_shape[1]}, "
                f"policy.yaml action_dim is {self.action_dim}"
            )

        metadata = self.session.get_modelmeta().custom_metadata_map or {}
        joint_names = _metadata_csv(metadata, "joint_names")
        if joint_names:
            if joint_names != self.obs_joint_order:
                raise ValueError("obs_joint_order must match ONNX metadata joint_names")
            if joint_names != self.action_joint_order:
                raise ValueError("action_joint_order must match ONNX metadata joint_names")

        observation_names = _metadata_csv(metadata, "observation_names")
        if observation_names:
            configured_names = [
                _ONNX_OBSERVATION_NAME.get(spec["type"], spec["type"])
                for spec in self.config["observations"]
            ]
            if observation_names != configured_names:
                raise ValueError(
                    "policy.yaml observations do not match ONNX metadata "
                    f"observation_names: {configured_names} != {observation_names}"
                )


_ONNX_OBSERVATION_NAME = {
    "motion_command": "command",
    "base_lin_vel_zero": "base_lin_vel",
    "joint_pos_rel": "joint_pos",
    "prev_action": "actions",
}


def _metadata_csv(metadata: dict[str, str], key: str) -> list[str]:
    value = metadata.get(key, "")
    return [item.strip() for item in value.split(",") if item.strip()]
