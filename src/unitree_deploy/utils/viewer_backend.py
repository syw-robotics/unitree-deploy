from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mujoco
import mujoco.viewer
import numpy as np

from unitree_deploy.robot_model.robot_config import RobotModel
from unitree_deploy.utils.yaml_utils import load_yaml


class ViewerBackend(Protocol):
    """Small lifecycle adapter so SimBridge does not branch on viewer type."""

    def run(self, simulate: Callable[[], None]) -> None:
        ...

    def sync(self) -> bool:
        ...

    def set_height_scan_points(
        self,
        points: np.ndarray,
        valid: np.ndarray,
        *,
        point_size: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        ...


@dataclass(frozen=True)
class ViewerCameraConfig:
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.85)
    distance: float = 2.0
    elevation: float = -15.0
    azimuth: float = 20.0
    track_body: str | None = None
    track_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def load_viewer_camera_config(robot: RobotModel) -> ViewerCameraConfig:
    path = robot.config_dir / "visualizer.yaml"
    if not path.exists():
        return ViewerCameraConfig()

    config = load_yaml(path)
    camera = config.get("viewer", {}).get("camera", {})
    if not isinstance(camera, dict):
        return ViewerCameraConfig()

    defaults = ViewerCameraConfig()
    lookat = camera.get("lookat", defaults.lookat)
    if len(lookat) != 3:
        raise ValueError(f"{path} viewer.camera.lookat must contain 3 values")

    track_offset = camera.get("track_offset", defaults.track_offset)
    if len(track_offset) != 3:
        raise ValueError(f"{path} viewer.camera.track_offset must contain 3 values")

    return ViewerCameraConfig(
        lookat=tuple(float(value) for value in lookat),
        distance=float(camera.get("distance", defaults.distance)),
        elevation=float(camera.get("elevation", defaults.elevation)),
        azimuth=float(camera.get("azimuth", defaults.azimuth)),
        track_body=camera.get("track_body"),
        track_offset=tuple(float(value) for value in track_offset),
    )


class MujocoViewerBackend:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera: ViewerCameraConfig,
        *,
        sim_hz: int,
        render_hz: int,
    ) -> None:
        self.model = model
        self.data = data
        self.camera = camera
        self.track_body_id = self.resolve_track_body_id()
        self.viewer = None
        self.viewer_tick = 0
        self.viewer_decim = max(1, sim_hz // render_hz)
        self._height_scan_points = None
        self._height_scan_valid = None
        self._height_scan_size = 0.025
        self._height_scan_rgba = (0.1, 0.75, 1.0, 0.9)
        self._marker_mat = np.eye(3, dtype=np.float64).reshape(-1)

    def resolve_track_body_id(self) -> int | None:
        if not self.camera.track_body:
            return None
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.camera.track_body,
        )
        if body_id < 0:
            raise ValueError(f"viewer.camera.track_body not found: {self.camera.track_body}")
        return int(body_id)

    def run(self, simulate: Callable[[], None]) -> None:
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            self.viewer = viewer
            viewer.cam.lookat[:] = self.camera.lookat
            viewer.cam.distance = self.camera.distance
            viewer.cam.elevation = self.camera.elevation
            viewer.cam.azimuth = self.camera.azimuth
            simulate()
            self.viewer = None

    def sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        self.viewer_tick += 1
        if self.viewer_tick % self.viewer_decim == 0:
            self.update_tracked_lookat()
            self.update_height_scan_markers()
            self.viewer.sync()
        return True

    def set_height_scan_points(
        self,
        points: np.ndarray,
        valid: np.ndarray,
        *,
        point_size: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        self._height_scan_points = np.asarray(points, dtype=np.float64).reshape(-1, 3).copy()
        self._height_scan_valid = np.asarray(valid, dtype=bool).reshape(-1).copy()
        self._height_scan_size = float(point_size)
        self._height_scan_rgba = tuple(float(value) for value in rgba)

    def update_tracked_lookat(self) -> None:
        if self.viewer is None or self.track_body_id is None:
            return
        self.viewer.cam.lookat[:] = self.data.xpos[self.track_body_id] + self.camera.track_offset

    def update_height_scan_markers(self) -> None:
        if self.viewer is None:
            return
        lock = getattr(self.viewer, "lock", None)
        if callable(lock):
            with lock():
                self._write_height_scan_markers()
        else:
            self._write_height_scan_markers()

    def _write_height_scan_markers(self) -> None:
        scene = self.viewer.user_scn
        if self._height_scan_points is None or self._height_scan_valid is None:
            scene.ngeom = 0
            return

        points = self._height_scan_points[self._height_scan_valid]
        count = min(points.shape[0], scene.maxgeom)
        scene.ngeom = count
        size = np.array([self._height_scan_size] * 3, dtype=np.float64)
        rgba = np.asarray(self._height_scan_rgba, dtype=np.float32)
        for index in range(count):
            mujoco.mjv_initGeom(
                scene.geoms[index],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size,
                points[index],
                self._marker_mat,
                rgba,
            )


def create_viewer_backend(
    viewer: str,
    robot: RobotModel,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    sim_hz: int,
    render_hz: int,
    log: Callable[[str], None],
) -> ViewerBackend:
    camera = load_viewer_camera_config(robot)
    if viewer == "mujoco":
        return MujocoViewerBackend(
            model,
            data,
            camera,
            sim_hz=sim_hz,
            render_hz=render_hz,
        )
    raise ValueError(f"unsupported viewer backend: {viewer}")
