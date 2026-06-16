from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import yaml

from .base_policy import BasePolicy, ObservationRegistry
from ..obs.observation import ObservationBase


DEFAULT_POLICY_CLASS = "unitree_deploy.policy.base_policy:BasePolicy"


@contextmanager
def _import_search_path(path: Path):
    path_str = str(path)
    added = path_str not in sys.path
    if added:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def _import_module(module_name: str, search_dir: Path) -> ModuleType:
    with _import_search_path(search_dir):
        return importlib.import_module(module_name)


def _import_symbol(spec: str, search_dir: Path):
    module_name, separator, symbol_name = spec.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError(f"import spec must be '<module>:<symbol>', got {spec!r}")

    module = _import_module(module_name, search_dir)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AttributeError(f"module {module_name!r} has no symbol {symbol_name!r}") from exc


def _load_observation_types(
    module_names: Sequence[str],
    type_specs: dict[str, str],
    search_dir: Path,
) -> dict[str, type[ObservationBase]]:
    observation_types = {}
    for type_name, import_spec in type_specs.items():
        observation_cls = _import_symbol(import_spec, search_dir)
        if not issubclass(observation_cls, ObservationBase):
            raise TypeError(f"observation type {type_name!r} must resolve to an ObservationBase subclass")
        observation_types[type_name] = observation_cls

    for module_name in module_names:
        module = _import_module(module_name, search_dir)
        module_types = getattr(module, "OBSERVATION_TYPES", None)
        if module_types is None:
            continue
        if not isinstance(module_types, dict):
            raise TypeError(f"{module_name}.OBSERVATION_TYPES must be a dict")
        observation_types.update(module_types)
    return observation_types


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


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
    config = _load_yaml(policy_yaml_path)
    if "observations" not in config:
        raise KeyError(f"{policy_yaml_path} must define an 'observations' list")

    search_dir = policy_yaml_path.parent
    policy_class_spec = config.get("policy_class", DEFAULT_POLICY_CLASS)
    policy_class = _import_symbol(policy_class_spec, search_dir)
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
