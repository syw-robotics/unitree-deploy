from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import onnxruntime as ort

from ..utils.import_utils import import_module, import_symbol
from ..utils.yaml_utils import load_yaml
from ..obs.observation import (
    ObservationBase,
    ObservationContext,
    ObservationGroup,
    PreviousActionObservation,
)

ObservationRegistry: TypeAlias = Mapping[str, type[ObservationBase]]

DEFAULT_POLICY_CLASS = "unitree_deploy.policy.base_policy:BasePolicy"


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
        self.config = load_yaml(self.policy_yaml_path)

        self.observation_types = dict(observation_types) if observation_types else {}

        self._load_action_config()
        self._load_policy_config()
        self._load_observations()
        self._observation_needs_prime = self.obs_prime_on_reset

        self.model_path = (self.policy_yaml_path.parent / self.config["policy_path"]).resolve()
        session_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=session_providers,
        )

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
        self.sdk_joint_index = {name: idx for idx, name in enumerate(self.sdk_joint_order)}
        self.sdk_to_obs_indices = np.asarray(
            [self.sdk_joint_index[name] for name in self.obs_joint_order],
            dtype=np.int64,
        )
        default_joint_pos_sdk = np.asarray(
            self.config["default_qpos"],
            dtype=np.float32,
        )
        if default_joint_pos_sdk.size != len(self.sdk_joint_order):
            raise ValueError(
                f"default_qpos has {default_joint_pos_sdk.size} values, "
                f"expected {len(self.sdk_joint_order)} from sdk_joint_order"
            )
        self.default_joint_pos = default_joint_pos_sdk[self.sdk_to_obs_indices]
        self.physics_dt = float(self.config["physics_dt"])
        self.policy_step_dt = float(self.config["policy_step_dt"])
        self.decimation = int(self.config["decimation"])

        self.action_dim = int(self.config["action_dim"])
        self.action_clip = self._parse_action_clip(self.config.get("action_clip"))
        if self.action_dim != len(self.action_joint_order):
            raise ValueError(
                f"action_dim={self.action_dim} does not match action_joint_order "
                f"length={len(self.action_joint_order)}"
            )
        action_scale_values = self.config["action_scale"]
        if len(action_scale_values) not in (len(self.sdk_joint_order), len(self.action_joint_order)):
            raise ValueError(
                f"action_scale has {len(action_scale_values)} values, "
                f"expected {len(self.sdk_joint_order)} from sdk_joint_order "
                f"or {len(self.action_joint_order)} from action_joint_order"
            )

        self.action_to_obs_indices = np.asarray(
            [self.obs_joint_index[name] for name in self.action_joint_order],
            dtype=np.int64,
        )

        self.default_joint_pos_action = self.default_joint_pos[self.action_to_obs_indices]

        action_scale_order = (
            self.sdk_joint_order if len(action_scale_values) == len(self.sdk_joint_order) else self.action_joint_order
        )
        action_scaling_cfg = {name: value for name, value in zip(action_scale_order, action_scale_values)}
        self.action_scaling = np.asarray(
            [action_scaling_cfg[name] for name in self.action_joint_order],
            dtype=np.float32,
        )
        self.obs_use_scaled_prev_action = bool(self.config.get("obs_use_scaled_prev_action", True))
        self.action = np.zeros(len(self.action_joint_order), dtype=np.float32)
        self.target_q = self.default_joint_pos.copy()

    @staticmethod
    def _parse_action_clip(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("none", "null"):
            return None
        return float(value)

    # ----- ONNX input/output names -----

    def _load_policy_config(self) -> None:
        self.input_name = self.config["policy_input_name"]
        self.action_output_name = self.config["policy_output_name"]
        self.obs_prime_on_reset = bool(
            self.config.get("obs_prime_on_reset", False)
        )

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
        self.observation = ObservationGroup(
            observations,
            obs_group_concat_mode=self.config.get("obs_group_concat_mode", "term_major"),
        )

    def _build_observation(self, observation_spec: dict) -> ObservationBase:
        observation_type = observation_spec["type"]
        kwargs: dict[str, object] = {"history_len": int(observation_spec["history_len"])}
        params = observation_spec.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise TypeError(f"observation params for {observation_type!r} must be a mapping")
        params = dict(params)
        clip = observation_spec.get("clip")
        scale = observation_spec.get("scale")

        if observation_type == "command":
            kwargs["command_range"] = observation_spec["command_range"]
        elif observation_type == "joint_pos_rel":
            kwargs["controlled_joint_indices"] = self.action_to_obs_indices
            kwargs["default_q"] = self.default_joint_pos_action
            kwargs.update(params)
        elif observation_type in ("joint_pos", "joint_vel"):
            kwargs["controlled_joint_indices"] = self.action_to_obs_indices
            kwargs.update(params)
        elif observation_type == "prev_action":
            kwargs["action_dim"] = self.action_dim
            kwargs.update(params)
        else:
            kwargs.update(params)

        try:
            observation_cls = self.observation_types[observation_type]
        except KeyError as exc:
            known_types = ", ".join(sorted(self.observation_types))
            raise KeyError(f"unknown observation type {observation_type!r}; known types: {known_types}") from exc

        observation = observation_cls(**kwargs)
        if scale is not None:
            observation.set_scale(scale)
        if clip is not None:
            observation.set_clip(clip)
        return observation

    def reset(self) -> None:
        self.observation.reset()
        self._observation_needs_prime = self.obs_prime_on_reset

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        if self._observation_needs_prime:
            self.observation.prime(context)
            self._observation_needs_prime = False
        else:
            self.observation.update(context)
        obs_vector = self.observation.compute().astype(np.float32, copy=False)
        outputs = self.session.run(
            [self.action_output_name],
            {self.input_name: obs_vector[None, :]},
        )
        policy_action = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        self.action[:] = policy_action
        prev_action = self.action * self.action_scaling if self.obs_use_scaled_prev_action else self.action
        self.previous_action_observation.record_action(prev_action)
        if self.action_clip is not None:
            np.clip(self.action, -self.action_clip, self.action_clip, out=self.action)

        # Return full robot targets in obs_joint_order; the controller converts to sdk_joint_order.
        self.target_q[:] = self.default_joint_pos
        self.target_q[self.action_to_obs_indices] = (
            self.default_joint_pos_action
            + self.action_scaling * self.action
        )
        return self.target_q



def _load_observation_types(
    module_names: Sequence[str],
    type_specs: dict[str, str],
    search_dir: Path,
) -> dict[str, type[ObservationBase]]:
    observation_types = {}
    for type_name, import_spec in type_specs.items():
        observation_cls = import_symbol(import_spec, search_dir)
        if not issubclass(observation_cls, ObservationBase):
            raise TypeError(f"observation type {type_name!r} must resolve to an ObservationBase subclass")
        observation_types[type_name] = observation_cls

    for module_name in module_names:
        module = import_module(module_name, search_dir)
        module_types = getattr(module, "OBSERVATION_TYPES", None)
        if module_types is None:
            continue
        if not isinstance(module_types, dict):
            raise TypeError(f"{module_name}.OBSERVATION_TYPES must be a dict")
        observation_types.update(module_types)
    return observation_types


def load_policy(
    policy_yaml_path: str | Path,
    *,
    providers: Sequence[str] | None = None,
) -> BasePolicy:
    """Load a policy from YAML, including optional user plugin modules.

    Optional policy.yaml fields:
      policy_class: "custom_policy:CustomPolicy"
      observation_types:
        custom_obs: "custom_observations:CustomObservation"
      observations:
        - type: custom_obs
          history_len: 1

    Module names are imported with the policy YAML directory temporarily added
    to sys.path, so deployment folders can carry their own plugin files.
    """

    policy_yaml_path = Path(policy_yaml_path).expanduser().resolve()
    config = load_yaml(policy_yaml_path)
    if "observations" not in config:
        raise KeyError(f"{policy_yaml_path} must define an 'observations' list")

    search_dir = policy_yaml_path.parent
    policy_class_spec = config.get("policy_class", DEFAULT_POLICY_CLASS)
    policy_class = import_symbol(policy_class_spec, search_dir)
    if not issubclass(policy_class, BasePolicy):
        raise TypeError(f"{policy_class_spec!r} must resolve to a BasePolicy subclass")

    observation_modules = config.get("observation_modules", [])
    if observation_modules is None:
        observation_modules = []
    if not isinstance(observation_modules, list):
        raise TypeError("'observation_modules' must be a list")

    observation_type_specs = config.get("observation_types", {})
    if observation_type_specs is None:
        observation_type_specs = {}
    if not isinstance(observation_type_specs, dict):
        raise TypeError("'observation_types' must be a mapping")

    observation_types: ObservationRegistry = _load_observation_types(
        observation_modules,
        observation_type_specs,
        search_dir,
    )

    return policy_class(
        policy_yaml_path,
        providers=providers,
        observation_types=observation_types,
    )
