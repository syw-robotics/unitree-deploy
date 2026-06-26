from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from unitree_deploy.runtime.sensor.config import load_sensor_section, sensor_shared_memory_name


def load_sensor_height_scan_config(sensor: Path | None) -> dict | None:
    config = load_sensor_section(sensor, "height_scan")
    if config is None:
        return None

    scan_type = str(config.get("type", "height_scan")).lower()
    if scan_type != "height_scan":
        return None
    return config


def is_mujoco_height_scan_source(scan_config: dict) -> bool:
    # Real height-map producers are intentionally left external for now; they can
    # write the same shared-memory buffer that HeightScanObservation consumes.
    source = str(scan_config.get("source", "mujoco_raycast")).lower()
    return source in ("mujoco", "mujoco_raycast", "sim")


def height_scan_shared_memory_name(config_path: Path, scan_config: dict) -> str:
    return sensor_shared_memory_name(
        config_path,
        scan_config,
        prefix="unitree_height_scan",
        default_name="height_scan",
    )


class MujocoHeightScanSensor:
    """Height grid producer backed by MuJoCo raycasts against static terrain geoms."""

    def __init__(self, mj_model, mj_data, config: dict) -> None:
        self.model = mj_model
        self.data = mj_data
        self.config = config
        self.name = str(config.get("name", "height_scan"))
        self.attach_body = str(config.get("attach_body", "base_link"))
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.attach_body)
        if self.body_id < 0:
            raise ValueError(f"height_scan.attach_body not found in MuJoCo model: {self.attach_body}")

        grid = config.get("grid", {})
        if not isinstance(grid, dict):
            raise TypeError("height_scan.grid must be a mapping")
        shape = grid.get("shape", [15, 11])
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("height_scan.grid.shape must contain [height, width]")
        self.height = int(shape[0])
        self.width = int(shape[1])
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"height_scan.grid.shape must be positive, got {shape}")

        x_range = _range_pair(grid.get("x_range", [-0.5, 1.0]), "height_scan.grid.x_range")
        y_range = _range_pair(grid.get("y_range", [-0.4, 0.4]), "height_scan.grid.y_range")
        xs = np.linspace(x_range[0], x_range[1], self.height, dtype=np.float64)
        ys = np.linspace(y_range[0], y_range[1], self.width, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        self._local_xy = np.stack((xx, yy), axis=-1).reshape(-1, 2)

        ray = config.get("ray", {})
        if ray is None:
            ray = {}
        if not isinstance(ray, dict):
            raise TypeError("height_scan.ray must be a mapping")
        self.start_z = float(ray.get("start_z", 0.8))
        self.max_distance = float(ray.get("max_distance", 2.0))
        if self.max_distance <= 0.0:
            raise ValueError("height_scan.ray.max_distance must be positive")
        self.align_to_body_yaw = bool(ray.get("align_to_body_yaw", True))
        self.direction = _normalized_vec3(ray.get("direction", [0.0, 0.0, -1.0]), "height_scan.ray.direction")

        preprocessing = config.get("preprocessing", {})
        if preprocessing is None:
            preprocessing = {}
        if not isinstance(preprocessing, dict):
            raise TypeError("height_scan.preprocessing must be a mapping")
        self.mode = str(preprocessing.get("mode", "base_relative_height"))
        self.fill_miss = float(preprocessing.get("fill_miss", 1.0))
        self.clip_range = _range_pair(preprocessing.get("clip_range", [-1.0, 1.0]), "height_scan.preprocessing.clip_range")

        # MuJoCo ray group mask: terrain geoms are in group 0, robot visual/collision geoms are not.
        groups = ray.get("geom_groups", [0])
        self.geomgroup = np.zeros(6, dtype=np.uint8)
        for group in groups:
            group_id = int(group)
            if group_id < 0 or group_id >= self.geomgroup.size:
                raise ValueError(f"height_scan.ray.geom_groups entries must be in [0, 5], got {group_id}")
            self.geomgroup[group_id] = 1
        self.static_only = int(bool(ray.get("static_only", True)))
        self._geomid = np.array([-1], dtype=np.int32)
        self._origin = np.zeros(3, dtype=np.float64)
        self._output = np.empty((self.height, self.width), dtype=np.float32)
        self._hit_points = np.empty((self.height * self.width, 3), dtype=np.float64)
        self._hit_valid = np.zeros(self.height * self.width, dtype=bool)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def hit_points(self) -> np.ndarray:
        return self._hit_points

    @property
    def hit_valid(self) -> np.ndarray:
        return self._hit_valid

    def capture(self) -> np.ndarray:
        body_pos = self.data.xpos[self.body_id]
        if self.align_to_body_yaw:
            xmat = self.data.xmat[self.body_id].reshape(3, 3)
            yaw = math.atan2(float(xmat[1, 0]), float(xmat[0, 0]))
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            rot2 = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
        else:
            rot2 = self.data.xmat[self.body_id].reshape(3, 3)[:2, :2]

        for index, local_xy in enumerate(self._local_xy):
            world_xy = body_pos[:2] + rot2 @ local_xy
            self._origin[0] = world_xy[0]
            self._origin[1] = world_xy[1]
            self._origin[2] = body_pos[2] + self.start_z
            self._geomid[0] = -1
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                self._origin,
                self.direction,
                self.geomgroup,
                self.static_only,
                self.body_id,
                self._geomid,
            )
            self._update_hit_point(index, distance)
            self._output.flat[index] = self._ray_value(distance, body_pos[2])

        np.clip(self._output, self.clip_range[0], self.clip_range[1], out=self._output)
        return self._output

    def _ray_value(self, distance: float, base_z: float) -> float:
        if distance < 0.0 or distance > self.max_distance:
            return self.fill_miss

        hit_z = self._origin[2] + distance * self.direction[2]
        if self.mode == "base_relative_height":
            return float(hit_z - base_z)
        if self.mode == "clearance":
            return float(self._origin[2] - hit_z)
        if self.mode == "world_height":
            return float(hit_z)
        if self.mode == "distance":
            return float(distance)
        raise ValueError(f"unknown height_scan.preprocessing.mode: {self.mode!r}")

    def _update_hit_point(self, index: int, distance: float) -> None:
        if distance < 0.0 or distance > self.max_distance:
            self._hit_points[index] = self._origin + self.direction * self.max_distance
            self._hit_valid[index] = False
            return
        self._hit_points[index] = self._origin + self.direction * distance
        self._hit_valid[index] = True


def _range_pair(value, field: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain two values")
    return (float(value[0]), float(value[1]))


def _normalized_vec3(value, field: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain three values")
    vector = np.asarray([float(v) for v in value], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{field} must be non-zero")
    return vector / norm
