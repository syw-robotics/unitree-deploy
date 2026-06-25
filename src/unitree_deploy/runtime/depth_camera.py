from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class DepthCameraBase(ABC):
    """Abstract base for depth camera sources."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        fov: float,
        near: float,
        far: float,
        clip_range: tuple[float, float],
        normalize_mode: str,
        fill_invalid: float,
    ) -> None:
        self.height = height
        self.width = width
        self.fov = fov
        self.near = near
        self.far = far
        self.clip_range = clip_range
        self.normalize_mode = normalize_mode
        self.fill_invalid = fill_invalid

    @abstractmethod
    def read_depth(self) -> np.ndarray:
        """Read raw depth image from camera source."""
        pass

    def preprocess_depth(self, depth_raw: np.ndarray) -> np.ndarray:
        """Apply preprocessing to raw depth image."""
        depth = depth_raw.copy()

        # Handle invalid values
        invalid_mask = np.isnan(depth) | np.isinf(depth)
        depth[invalid_mask] = self.fill_invalid

        # Clip to range
        depth = np.clip(depth, self.clip_range[0], self.clip_range[1])

        # Normalize
        if self.normalize_mode == "clip_scale":
            depth = (depth - self.clip_range[0]) / (self.clip_range[1] - self.clip_range[0])
        elif self.normalize_mode == "standard":
            depth = (depth - depth.mean()) / (depth.std() + 1e-8)
        # else: no normalization

        return depth.astype(np.float32)

    def capture(self) -> np.ndarray:
        """Capture and preprocess depth image."""
        depth_raw = self.read_depth()
        return self.preprocess_depth(depth_raw)


class MujocoDepthCamera(DepthCameraBase):
    """Mujoco depth camera source."""

    def __init__(self, mj_model, mj_data, camera_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.camera_name = camera_name
        self._renderer = None

    def read_depth(self) -> np.ndarray:
        """Render depth from Mujoco."""
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, self.height, self.width)

        self._renderer.update_scene(self.mj_data, camera=self.camera_name)
        depth_buffer = self._renderer.render()

        # Mujoco depth is in range [-1, 1], convert to meters
        extent = self.mj_model.stat.extent
        near = self.mj_model.vis.map.znear * extent
        far = self.mj_model.vis.map.zfar * extent
        depth_meters = near / (1 - depth_buffer * (1 - near / far))

        return depth_meters


class RealSenseDepthCamera(DepthCameraBase):
    """RealSense depth camera source."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pipeline = None
        self._align = None
        self._setup_camera()

    def _setup_camera(self) -> None:
        """Initialize RealSense pipeline."""
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)

        self._pipeline.start(config)
        self._align = rs.align(rs.stream.depth)

    def read_depth(self) -> np.ndarray:
        """Capture depth frame from RealSense."""
        import pyrealsense2 as rs

        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        if not depth_frame:
            return np.full((self.height, self.width), self.fill_invalid, dtype=np.float32)

        # Convert to numpy array (depth in mm)
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert mm to meters
        depth_meters = depth_image.astype(np.float32) / 1000.0

        return depth_meters

    def close(self) -> None:
        """Stop pipeline."""
        if self._pipeline:
            self._pipeline.stop()
