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
        for observation in self.observation.observations:
            setattr(observation, "policy", self)
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
