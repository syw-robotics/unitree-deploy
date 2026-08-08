from __future__ import annotations

from abc import ABC, abstractmethod
import math
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


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
        crop: tuple[int, int, int, int] = (0, 0, 0, 0),
        gaussian_blur: dict | None = None,
    ) -> None:
        self.height = height
        self.width = width
        self.fov = fov
        self.near = near
        self.far = far
        self.clip_range = clip_range
        self.normalize_mode = normalize_mode
        self.fill_invalid = fill_invalid
        self.crop_top, self.crop_bottom, self.crop_left, self.crop_right = crop
        self.output_height = self.height - self.crop_top - self.crop_bottom
        self.output_width = self.width - self.crop_left - self.crop_right
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError(
                "depth crop removes the full image: "
                f"height={self.height}, width={self.width}, crop={crop}"
            )
        gaussian_blur = gaussian_blur or {}
        self.blur_kernel_size = int(gaussian_blur.get("kernel_size", 0))
        self.blur_sigma = float(gaussian_blur.get("sigma", 0.0))
        if self.blur_kernel_size < 0 or (
            self.blur_kernel_size > 0 and self.blur_kernel_size % 2 == 0
        ):
            raise ValueError(
                "camera.preprocessing.gaussian_blur.kernel_size must be zero or a positive odd number"
            )
        if self.blur_sigma < 0.0:
            raise ValueError("camera.preprocessing.gaussian_blur.sigma must be non-negative")

    @abstractmethod
    def read_depth(self) -> np.ndarray:
        """Read raw depth image from camera source."""
        pass

    def preprocess_depth(self, depth_raw: np.ndarray) -> np.ndarray:
        """Apply preprocessing to raw depth image."""
        depth = depth_raw.copy()

        if any((self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)):
            bottom = self.height - self.crop_bottom if self.crop_bottom else self.height
            right = self.width - self.crop_right if self.crop_right else self.width
            depth = depth[self.crop_top:bottom, self.crop_left:right]

        # Handle invalid values
        invalid_mask = np.isnan(depth) | np.isinf(depth)
        depth[invalid_mask] = self.fill_invalid

        if self.blur_kernel_size > 1 and self.blur_sigma > 0.0:
            radius = (self.blur_kernel_size - 1) / 2.0
            truncate = radius / self.blur_sigma
            # scipy "mirror" matches torch.nn.functional.pad(..., mode="reflect"):
            # the edge sample itself is not duplicated outside the image.
            depth = gaussian_filter(depth, sigma=self.blur_sigma, truncate=truncate, mode="mirror")

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

    def __init__(
        self,
        mj_model,
        mj_data,
        camera_name: str,
        *,
        fovx: float | None = None,
        render_scale: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.camera_name = camera_name
        self.fovx = float(fovx) if fovx is not None else None
        self.render_scale = int(render_scale)
        if self.fovx is not None and not 0.0 < self.fovx < 180.0:
            raise ValueError("camera.intrinsics.fovx must be in (0, 180)")
        if self.render_scale < 1:
            raise ValueError("camera.intrinsics.render_scale must be >= 1")
        self.render_height = self.height * self.render_scale
        self.render_width = self.width * self.render_scale
        self._renderer = None

    @staticmethod
    def intrinsic_remap_coordinates(
        *,
        output_height: int,
        output_width: int,
        render_height: int,
        render_width: int,
        fovy: float,
        fovx: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map Isaac-style independent fovx/fovy pixel centers into a MuJoCo square-pixel render."""
        fovy_rad = math.radians(float(fovy))
        fovx_rad = math.radians(float(fovx))
        source_focal = render_height / (2.0 * math.tan(fovy_rad / 2.0))
        target_fx = output_width / (2.0 * math.tan(fovx_rad / 2.0))
        target_fy = output_height / (2.0 * math.tan(fovy_rad / 2.0))

        target_u = np.arange(output_width, dtype=np.float64) + 0.5
        target_v = np.arange(output_height, dtype=np.float64) + 0.5
        target_u, target_v = np.meshgrid(target_u, target_v)
        normalized_x = (target_u - output_width / 2.0) / target_fx
        normalized_y = (target_v - output_height / 2.0) / target_fy

        # map_coordinates uses array indices, while camera intrinsics use pixel centers.
        source_u = render_width / 2.0 + normalized_x * source_focal - 0.5
        source_v = render_height / 2.0 + normalized_y * source_focal - 0.5
        return source_v, source_u

    def _match_training_intrinsics(self, depth: np.ndarray) -> np.ndarray:
        if self.fovx is None and self.render_scale == 1:
            return depth
        fovx = self.fovx
        if fovx is None:
            fovx = math.degrees(
                2.0 * math.atan(math.tan(math.radians(self.fov) / 2.0) * self.width / self.height)
            )
        source_v, source_u = self.intrinsic_remap_coordinates(
            output_height=self.height,
            output_width=self.width,
            render_height=self.render_height,
            render_width=self.render_width,
            fovy=self.fov,
            fovx=fovx,
        )
        return map_coordinates(depth, [source_v, source_u], order=1, mode="nearest").astype(np.float32)

    def read_depth(self) -> np.ndarray:
        """Render depth from Mujoco."""
        import mujoco

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, self.render_height, self.render_width)
            self._renderer.enable_depth_rendering()

        self._renderer.update_scene(self.mj_data, camera=self.camera_name)
        return self._match_training_intrinsics(self._renderer.render())

    def close(self) -> None:
        """Release MuJoCo renderer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


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
