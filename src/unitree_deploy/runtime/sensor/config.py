from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sensor_yaml_path(sensor: Path | None) -> Path | None:
    if sensor is None:
        return None
    sensor = sensor.expanduser().resolve()
    if sensor.is_dir():
        raise ValueError(f"--sensor must point to a sensor yaml file, got directory: {sensor}")
    return sensor


def load_sensor_config(sensor: Path | None) -> dict:
    sensor_yaml = sensor_yaml_path(sensor)
    if sensor_yaml is None or not sensor_yaml.exists():
        return {}

    from unitree_deploy.utils.yaml_utils import load_yaml

    config = load_yaml(sensor_yaml)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"{sensor_yaml} must contain a mapping")
    return config


def load_sensor_section(sensor: Path | None, section_name: str) -> dict | None:
    section = load_sensor_config(sensor).get(section_name)
    if not isinstance(section, dict) or not section:
        return None
    if section.get("enabled", True) is False:
        return None
    return dict(section)


def sensor_shared_memory_name(
    config_path: Path,
    sensor_config: dict,
    *,
    prefix: str,
    default_name: str,
) -> str:
    configured_name = sensor_config.get("shared_memory_name")
    if configured_name:
        return str(configured_name)

    sensor_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(sensor_config.get("name", default_name)))
    digest = hashlib.sha1(config_path.expanduser().resolve().as_posix().encode()).hexdigest()[:12]
    return f"{prefix}_{digest}_{sensor_name}"
