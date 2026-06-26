from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from scipy.spatial.transform import Rotation

from unitree_deploy.runtime.sensor.config import load_sensor_section, sensor_shared_memory_name


def load_sensor_camera_config(sensor: Path | None) -> dict | None:
    camera_config = load_sensor_section(sensor, "camera")
    if camera_config is None:
        return None

    camera_type = str(camera_config.get("type", "depth")).lower()
    camera_source = str(camera_config.get("source", "mujoco")).lower()
    if camera_type != "depth" or camera_source not in ("mujoco", "sim"):
        return None

    return dict(camera_config)


def camera_shared_memory_name(config_path: Path, camera_config: dict) -> str:
    return sensor_shared_memory_name(
        config_path,
        camera_config,
        prefix="unitree_depth",
        default_name="depth_camera",
    )


def parse_depth_crop(preprocessing: dict) -> tuple[int, int, int, int]:
    crop = preprocessing.get("crop", {})
    if crop is None:
        crop = {}
    if not isinstance(crop, dict):
        raise TypeError("camera.preprocessing.crop must be a mapping")
    parsed = (
        int(crop.get("top", 0)),
        int(crop.get("bottom", 0)),
        int(crop.get("left", 0)),
        int(crop.get("right", 0)),
    )
    if any(value < 0 for value in parsed):
        raise ValueError(f"camera.preprocessing.crop values must be non-negative: {parsed}")
    return parsed


def _format_float_list(values, *, length: int, field: str) -> str:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"camera.{field} must contain {length} values")
    return " ".join(f"{float(value):.10g}" for value in values)


def _quat_wxyz_from_rpy(transform: dict) -> str:
    rpy = transform.get("rpy", [0.0, 0.0, 0.0])
    if not isinstance(rpy, (list, tuple)) or len(rpy) != 3:
        raise ValueError("camera.transform.rpy must contain 3 values")
    degrees = bool(transform.get("degrees", False))
    quat_xyzw = Rotation.from_euler("xyz", [float(value) for value in rpy], degrees=degrees).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return " ".join(f"{float(value):.10g}" for value in quat_wxyz)


def _find_body(root: ET.Element, body_name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def _remove_camera_by_name(root: ET.Element, camera_name: str) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "camera" and child.get("name") == camera_name:
                parent.remove(child)


def write_model_xml_with_sensor_camera(
    model_xml_path: Path,
    sensor_yaml_path: Path,
    camera_config: dict,
) -> Path:
    camera_name = str(camera_config.get("name", "depth_camera"))
    attach_body = str(camera_config.get("attach_body", "base_link"))
    transform = camera_config.get("transform", {})
    intrinsics = camera_config.get("intrinsics", {})
    if not isinstance(transform, dict):
        raise TypeError("camera.transform must be a mapping")
    if not isinstance(intrinsics, dict):
        raise TypeError("camera.intrinsics must be a mapping")

    pos = _format_float_list(
        transform.get("position", [0.0, 0.0, 0.0]),
        length=3,
        field="transform.position",
    )
    quat = _quat_wxyz_from_rpy(transform)
    fovy = float(intrinsics["fovy"])

    model_xml_path = model_xml_path.resolve()
    sensor_yaml_path = sensor_yaml_path.resolve()
    model_bytes = model_xml_path.read_bytes()
    sensor_bytes = sensor_yaml_path.read_bytes()
    cache_key = b"\0".join([model_xml_path.as_posix().encode(), model_bytes, sensor_bytes])
    digest = hashlib.sha256(cache_key).hexdigest()[:12]
    output_dir = Path(tempfile.gettempdir()) / "unitree-deploy-mujoco"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_xml_path.stem}_sensor_camera_{digest}.xml"

    root = ET.parse(model_xml_path).getroot()
    _remove_camera_by_name(root, camera_name)

    if attach_body.lower() in ("world", "worldbody"):
        parent = root.find("worldbody")
        if parent is None:
            raise ValueError(f"{model_xml_path} is missing <worldbody>")
    else:
        parent = _find_body(root, attach_body)
        if parent is None:
            raise ValueError(f"camera.attach_body not found in MuJoCo XML: {attach_body}")

    camera = ET.Element(
        "camera",
        {
            "name": camera_name,
            "pos": pos,
            "quat": quat,
            "fovy": f"{fovy:.10g}",
            "mode": str(camera_config.get("mode", "fixed")),
        },
    )
    parent.insert(0, camera)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path
