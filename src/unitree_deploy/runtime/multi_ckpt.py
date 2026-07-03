from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unitree_deploy.config.defaults import (
    RUN_POLICY_STATE,
    WIRELESS_REMOTE_BUTTON_BITS,
)
from unitree_deploy.policy.base_policy import BasePolicy, load_policy
from unitree_deploy.utils.yaml_utils import load_yaml


@dataclass(frozen=True)
class CkptProfile:
    name: str
    policy_yaml_path: Path
    policy: BasePolicy
    sdk_joint_order: list[str]
    obs_joint_order: list[str]
    sdk_to_obs: np.ndarray
    obs_to_sdk: np.ndarray
    kp_policy: np.ndarray
    kd_policy: np.ndarray
    kp_fixed_stand: np.ndarray
    kd_fixed_stand: np.ndarray
    kd_damping: np.ndarray
    command_min: np.ndarray
    command_max: np.ndarray
    default_q_obs: np.ndarray
    default_q_sdk: np.ndarray


@dataclass(frozen=True)
class SwitchConfig:
    enabled: bool
    button: str
    order: list[str]
    only_when: set[str]
    on_switch: str


class PolicyManager:
    def __init__(
        self,
        profiles: dict[str, CkptProfile],
        active_name: str,
        switch: SwitchConfig,
    ) -> None:
        if active_name not in profiles:
            raise KeyError(f"default ckpt {active_name!r} is not defined")
        self.profiles = profiles
        self.active_name = active_name
        self.switch = switch

    @property
    def active(self) -> CkptProfile:
        return self.profiles[self.active_name]

    @classmethod
    def load(cls, ckpt_dir: Path, multi_ckpt: Path | None = None) -> PolicyManager:
        ckpt_dir = ckpt_dir.expanduser().resolve()
        if multi_ckpt is None:
            profile = build_ckpt_profile("default", ckpt_dir / "policy.yaml")
            return cls(
                {"default": profile},
                "default",
                build_switch_config(["default"], {}),
            )

        multi_ckpt = multi_ckpt.expanduser().resolve()
        config = load_yaml(multi_ckpt)
        profiles, default_name = load_multi_ckpt_profiles(multi_ckpt.parent, config)
        check_profile_compatibility(profiles)
        return cls(
            profiles,
            default_name,
            build_switch_config(list(profiles), config.get("switch", {})),
        )

    def switch_allowed(self, state: str) -> bool:
        return self.switch.enabled and state in self.switch.only_when

    def switch_to(self, name: str) -> CkptProfile:
        if name not in self.profiles:
            raise KeyError(f"unknown ckpt profile {name!r}")
        self.active_name = name
        self.active.policy.reset()
        return self.active

    def switch_next(self) -> CkptProfile:
        order = self.switch.order
        current_i = order.index(self.active_name) if self.active_name in order else -1
        return self.switch_to(order[(current_i + 1) % len(order)])


def load_multi_ckpt_profiles(base_dir: Path, config: dict) -> tuple[dict[str, CkptProfile], str]:
    ckpt_specs = config.get("ckpts")
    if not isinstance(ckpt_specs, dict) or not ckpt_specs:
        raise ValueError("multi-ckpt YAML must define a non-empty 'ckpts' mapping")

    profile_paths = {
        str(name): resolve_policy_yaml(base_dir, spec)
        for name, spec in ckpt_specs.items()
    }
    default_name = str(config.get("default", next(iter(profile_paths))))
    profiles = {
        name: build_ckpt_profile(name, policy_yaml_path)
        for name, policy_yaml_path in profile_paths.items()
    }
    return profiles, default_name


def resolve_policy_yaml(base_dir: Path, spec) -> Path:
    if isinstance(spec, str):
        path = base_dir / spec
    elif isinstance(spec, dict):
        if "policy_yaml" in spec:
            path = base_dir / spec["policy_yaml"]
        elif "ckpt" in spec:
            path = base_dir / spec["ckpt"]
        elif "path" in spec:
            path = base_dir / spec["path"]
        else:
            raise KeyError("ckpt spec must define one of: policy_yaml, ckpt, path")
    else:
        raise TypeError("ckpt spec must be a string or mapping")

    path = path.expanduser()
    if path.is_dir():
        path = path / "policy.yaml"
    return path.resolve()


def build_ckpt_profile(name: str, policy_yaml_path: Path) -> CkptProfile:
    policy = load_policy(policy_yaml_path)
    sdk_joint_order = list(policy.sdk_joint_order)
    obs_joint_order = list(policy.obs_joint_order)
    num_joints = len(sdk_joint_order)
    check_joint_config(policy, sdk_joint_order, obs_joint_order, num_joints)

    sdk_to_obs = reorder_indices(sdk_joint_order, obs_joint_order)
    obs_to_sdk = reorder_indices(obs_joint_order, sdk_joint_order)
    kp_policy = gain_array(policy, num_joints, "kp_policy", legacy_keys=("kp", "kps_real"))
    kd_policy = gain_array(policy, num_joints, "kd_policy", legacy_keys=("kd", "kds_real"))
    kp_fixed_stand = gain_array(policy, num_joints, "kp_fixed_stand", fallback=kp_policy)
    kd_fixed_stand = gain_array(policy, num_joints, "kd_fixed_stand", fallback=kd_policy)
    kd_damping = gain_array(
        policy,
        num_joints,
        "kd_damping",
        fallback=np.ones(num_joints, dtype=np.float64),
    )
    command_min, command_max = command_range(policy)
    default_q_obs = policy.default_joint_pos
    default_q_sdk = default_q_obs[obs_to_sdk]

    return CkptProfile(
        name=name,
        policy_yaml_path=policy_yaml_path,
        policy=policy,
        sdk_joint_order=sdk_joint_order,
        obs_joint_order=obs_joint_order,
        sdk_to_obs=sdk_to_obs,
        obs_to_sdk=obs_to_sdk,
        kp_policy=kp_policy,
        kd_policy=kd_policy,
        kp_fixed_stand=kp_fixed_stand,
        kd_fixed_stand=kd_fixed_stand,
        kd_damping=kd_damping,
        command_min=command_min,
        command_max=command_max,
        default_q_obs=default_q_obs,
        default_q_sdk=default_q_sdk,
    )


def reorder_indices(source: list[str], target: list[str]) -> np.ndarray:
    source_index = {name: i for i, name in enumerate(source)}
    return np.asarray([source_index[name] for name in target], dtype=np.int64)


def check_joint_config(
    policy: BasePolicy,
    sdk_joint_order: list[str],
    obs_joint_order: list[str],
    num_joints: int,
) -> None:
    if len(obs_joint_order) != num_joints:
        raise ValueError(
            f"joint count mismatch: {len(obs_joint_order)} obs joints, "
            f"{num_joints} sdk joints"
        )
    if set(obs_joint_order) != set(sdk_joint_order):
        missing_in_sdk = sorted(set(obs_joint_order) - set(sdk_joint_order))
        missing_in_obs = sorted(set(sdk_joint_order) - set(obs_joint_order))
        raise ValueError(
            f"obs_joint_order and sdk_joint_order must contain the same joints; "
            f"missing_in_sdk={missing_in_sdk}, missing_in_obs={missing_in_obs}"
        )
    if len(policy.default_joint_pos) != num_joints:
        raise ValueError(
            f"policy default_qpos has {len(policy.default_joint_pos)} joints, "
            f"controller has {num_joints}"
        )


def gain_array(
    policy: BasePolicy,
    num_joints: int,
    key: str,
    *,
    legacy_keys: tuple[str, ...] = (),
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    config_key = next(
        (candidate for candidate in (key, *legacy_keys) if candidate in policy.config),
        None,
    )
    if config_key is None:
        if fallback is None:
            names = ", ".join(repr(name) for name in (key, *legacy_keys))
            raise KeyError(f"policy.yaml must define one of: {names}")
        values = np.asarray(fallback, dtype=np.float64).reshape(-1).copy()
        config_key = key
    else:
        values = np.asarray(policy.config[config_key], dtype=np.float64).reshape(-1)

    if values.size != num_joints:
        raise ValueError(f"{config_key} has {values.size} values, expected {num_joints}")
    return values


def command_range(policy: BasePolicy) -> tuple[np.ndarray, np.ndarray]:
    runtime_command_dim = policy.config.get("runtime_command_dim")
    if runtime_command_dim is not None:
        dim = int(runtime_command_dim)
        return (
            np.full(dim, -1.0, dtype=np.float64),
            np.full(dim, 1.0, dtype=np.float64),
        )

    for observation_spec in policy.config["observations"]:
        if observation_spec["type"] != "command":
            continue
        command_range_array = np.asarray(
            observation_spec["command_range"],
            dtype=np.float64,
        )
        if command_range_array.shape != (3, 2):
            raise ValueError(
                f"command_range must have shape (3, 2), got {command_range_array.shape}"
            )
        return command_range_array[:, 0], command_range_array[:, 1]
    raise KeyError("policy.yaml observations must include type: command")


def check_profile_compatibility(profiles: dict[str, CkptProfile]) -> None:
    first = next(iter(profiles.values()))
    for profile in profiles.values():
        if profile.sdk_joint_order != first.sdk_joint_order:
            raise ValueError(
                "all switchable ckpts must use the same sdk_joint_order; "
                f"{profile.name!r} differs from {first.name!r}"
            )
        if profile.policy.policy_step_dt != first.policy.policy_step_dt:
            raise ValueError(
                "all switchable ckpts must use the same policy_step_dt; "
                f"{profile.name!r}={profile.policy.policy_step_dt}, "
                f"{first.name!r}={first.policy.policy_step_dt}"
            )


def build_switch_config(
    profile_names: list[str],
    switch_config: dict | None,
) -> SwitchConfig:
    switch_config = switch_config or {}
    if not isinstance(switch_config, dict):
        raise TypeError("'switch' must be a mapping")

    enabled = bool(switch_config.get("enabled", len(profile_names) > 1))
    button = str(switch_config.get("button", "B"))
    if button not in WIRELESS_REMOTE_BUTTON_BITS:
        known = ", ".join(sorted(WIRELESS_REMOTE_BUTTON_BITS))
        raise KeyError(f"unknown switch button {button!r}; known buttons: {known}")

    order = list(switch_config.get("order", profile_names))
    unknown = sorted(set(order) - set(profile_names))
    if unknown:
        raise KeyError(f"switch.order contains unknown ckpts: {unknown}")
    if not order:
        raise ValueError("switch.order must not be empty")

    only_when = set(
        switch_config.get(
            "only_when",
            [RUN_POLICY_STATE],
        )
    )

    on_switch_value = switch_config.get("on_switch", None)
    on_switch = "" if on_switch_value is None else str(on_switch_value)

    return SwitchConfig(
        enabled=enabled and len(profile_names) > 1,
        button=button,
        order=order,
        only_when=only_when,
        on_switch=on_switch,
    )
