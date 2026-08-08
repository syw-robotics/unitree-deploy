from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from unitree_deploy.obs.observation import ObservationContext
from unitree_deploy.policy.base_policy import BasePolicy, ObservationRegistry


class HeightmapTeacherPolicy(BasePolicy):
    """Adapter for the separately exported prop/heightmap/mask teacher ONNX."""

    def __init__(
        self,
        policy_yaml_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        observation_types: ObservationRegistry | None = None,
    ) -> None:
        super().__init__(
            policy_yaml_path,
            providers=providers,
            observation_types=observation_types,
        )

        self.prop_input_name = str(self.config.get("prop_input_name", "prop"))
        self.heightmap_input_name = str(self.config.get("heightmap_input_name", "heightmap"))
        self.exterio_mask_input_name = str(self.config.get("exterio_mask_input_name", "exterio_mask"))

        shape = np.asarray(self.config["heightmap_sensor_shape"], dtype=np.int64).reshape(-1)
        if shape.size != 2 or np.any(shape <= 0):
            raise ValueError("heightmap_sensor_shape must contain two positive dimensions")
        self.heightmap_sensor_shape = (int(shape[0]), int(shape[1]))
        self.heightmap_size = int(np.prod(shape))
        self.heightmap_transpose = bool(self.config.get("heightmap_transpose", False))
        self.heightmap_scale = float(self.config.get("heightmap_scale", 1.0))
        self.heightmap_offset = float(self.config.get("heightmap_offset", 0.0))
        heightmap_clip = np.asarray(
            self.config.get("heightmap_clip", [-10.0, 10.0]),
            dtype=np.float32,
        ).reshape(-1)
        if heightmap_clip.size != 2 or heightmap_clip[0] > heightmap_clip[1]:
            raise ValueError("heightmap_clip must contain [min, max]")
        self.heightmap_clip = (float(heightmap_clip[0]), float(heightmap_clip[1]))
        self.exterio_mask = np.asarray(
            [[float(self.config.get("exterio_mask_value", 1.0))]],
            dtype=np.float32,
        )

        if self.observation.size <= self.heightmap_size:
            raise ValueError(
                f"observation vector has {self.observation.size} values; "
                f"expected more than heightmap size {self.heightmap_size}"
            )
        self.prop_size = self.observation.size - self.heightmap_size
        self._validate_onnx_signature()

    def _validate_onnx_signature(self) -> None:
        inputs = {item.name: item for item in self.session.get_inputs()}
        expected = {
            self.prop_input_name: self.prop_size,
            self.heightmap_input_name: self.heightmap_size,
            self.exterio_mask_input_name: 1,
        }
        for name, width in expected.items():
            if name not in inputs:
                raise ValueError(f"policy ONNX is missing required input {name!r}")
            shape = inputs[name].shape
            if len(shape) != 2 or (isinstance(shape[1], int) and shape[1] != width):
                raise ValueError(
                    f"policy ONNX input {name!r} has shape {shape}, expected [batch, {width}]"
                )

    def _prepare_heightmap(self, sensor_values: np.ndarray) -> np.ndarray:
        heightmap = np.asarray(sensor_values, dtype=np.float32).reshape(self.heightmap_sensor_shape)
        if self.heightmap_transpose:
            heightmap = heightmap.T
        heightmap = heightmap.reshape(1, -1)
        if self.heightmap_scale != 1.0:
            heightmap = heightmap * self.heightmap_scale
        if self.heightmap_offset != 0.0:
            heightmap = heightmap + self.heightmap_offset
        np.clip(heightmap, self.heightmap_clip[0], self.heightmap_clip[1], out=heightmap)
        return heightmap.astype(np.float32, copy=False)

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        if self._observation_needs_prime:
            self.observation.prime(context)
            self._observation_needs_prime = False
        else:
            self.observation.update(context)

        obs_vector = self.observation.compute().astype(np.float32, copy=False)
        prop = obs_vector[: self.prop_size][None, :]
        heightmap = self._prepare_heightmap(obs_vector[self.prop_size :])
        outputs = self.session.run(
            [self.action_output_name],
            {
                self.prop_input_name: prop,
                self.heightmap_input_name: heightmap,
                self.exterio_mask_input_name: self.exterio_mask,
            },
        )
        return self._target_q_from_policy_action(outputs[0], context)
