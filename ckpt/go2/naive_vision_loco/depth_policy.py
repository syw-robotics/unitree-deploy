from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort

from unitree_deploy.obs.observation import ObservationContext
from unitree_deploy.policy.base_policy import BasePolicy, ObservationRegistry


class DepthRecurrentPolicy(BasePolicy):
    """Split depth-encoder + recurrent actor policy."""

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
        session_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        self.actor_session = self.session
        self.encoder_path = (self.policy_yaml_path.parent / self.config["depth_encoder_path"]).resolve()
        self.encoder_session = ort.InferenceSession(
            str(self.encoder_path),
            sess_options=options,
            providers=session_providers,
        )

        self.depth_input_name = self.config.get("depth_encoder_input_name", "depth_image")
        self.depth_output_name = self.config.get("depth_encoder_output_name", "depth_image_latent")
        self.actor_input_name = self.config.get("policy_input_name", "obs")
        self.action_output_name = self.config.get("policy_output_name", "actions")

        self.depth_height = None
        self.depth_width = None
        self.depth_history_len = None
        self.depth_vector_size = None
        for spec in self.config["observations"]:
            if spec["type"] != "depth":
                continue
            params = spec.get("params") or {}
            self.depth_height = int(params["height"])
            self.depth_width = int(params["width"])
            self.depth_history_len = int(spec["history_len"])
            self.depth_vector_size = self.depth_height * self.depth_width * self.depth_history_len
            break
        if self.depth_vector_size is None:
            raise ValueError("DepthRecurrentPolicy requires a depth observation")

        actor_inputs = {item.name: item for item in self.actor_session.get_inputs()}
        self.h_state = self._zeros_for_input(actor_inputs, "h_in")
        self.c_state = self._zeros_for_input(actor_inputs, "c_in") if "c_in" in actor_inputs else None

    @staticmethod
    def _zeros_for_input(actor_inputs: dict[str, ort.NodeArg], name: str) -> np.ndarray:
        if name not in actor_inputs:
            raise ValueError(f"actor ONNX is missing required input {name!r}")
        shape = [1 if not isinstance(dim, int) else dim for dim in actor_inputs[name].shape]
        return np.zeros(shape, dtype=np.float32)

    def reset(self) -> None:
        super().reset()
        self.h_state.fill(0.0)
        if self.c_state is not None:
            self.c_state.fill(0.0)

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        if self._observation_needs_prime:
            self.observation.prime(context)
            self._observation_needs_prime = False
        else:
            self.observation.update(context)

        obs_vector = self.observation.compute().astype(np.float32, copy=False)
        prop_obs = obs_vector[:-self.depth_vector_size]
        depth_obs = obs_vector[-self.depth_vector_size:].reshape(
            1,
            self.depth_history_len,
            self.depth_height,
            self.depth_width,
        )
        depth_latent = self.encoder_session.run(
            [self.depth_output_name],
            {self.depth_input_name: depth_obs},
        )[0].astype(np.float32, copy=False)

        actor_obs = np.concatenate((prop_obs[None, :], depth_latent), axis=1).astype(np.float32, copy=False)
        actor_inputs = {
            self.actor_input_name: actor_obs,
            "h_in": self.h_state,
        }
        output_names = [self.action_output_name, "h_out"]
        if self.c_state is not None:
            actor_inputs["c_in"] = self.c_state
            output_names.append("c_out")

        outputs = self.actor_session.run(output_names, actor_inputs)
        output_index = 0
        policy_action = np.asarray(outputs[output_index], dtype=np.float32).reshape(-1)
        output_index += 1
        self.h_state[:] = outputs[output_index]
        output_index += 1
        if self.c_state is not None:
            self.c_state[:] = outputs[output_index]

        return self._target_q_from_policy_action(policy_action, context)
