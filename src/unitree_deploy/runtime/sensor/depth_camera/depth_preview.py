from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import numpy as np


@dataclass(frozen=True)
class DepthPreviewConfig:
    title: str = "unitree depth camera"
    scale: int = 4
    normalize_mode: str = "clip_scale"
    clip_range: tuple[float, float] = (0.0, 1.0)


class DepthPreviewWindow:
    """Small optional OpenCV preview window for simulated depth frames."""

    def __init__(self, config: DepthPreviewConfig, *, log=print) -> None:
        self.config = config
        self.log = log
        self._cv2 = None
        self._initialized = False
        self._disabled = False

    def _init_window(self, frame_shape: tuple[int, int]) -> bool:
        if self._disabled:
            return False
        if self._initialized:
            return True
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self._disabled = True
            self.log("depth preview disabled: no display server found")
            return False
        try:
            import cv2

            self._cv2 = cv2
            cv2.namedWindow(self.config.title, cv2.WINDOW_NORMAL)
            height, width = frame_shape
            scale = max(1, int(self.config.scale))
            cv2.resizeWindow(self.config.title, width * scale, height * scale)
            self._initialized = True
            return True
        except Exception as exc:
            self._disabled = True
            self.log(f"depth preview disabled: {exc}")
            return False

    def show(self, depth_image: np.ndarray) -> None:
        if depth_image.ndim != 2 or not self._init_window(depth_image.shape):
            return

        try:
            cv2 = self._cv2
            display = self._normalize(depth_image)
            image_u8 = (display * 255.0).astype(np.uint8)
            cv2.imshow(self.config.title, image_u8)
            cv2.waitKey(1)
        except Exception as exc:
            self._disabled = True
            self.log(f"depth preview disabled: {exc}")

    def _normalize(self, depth_image: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_image, dtype=np.float32)
        valid = np.isfinite(depth)
        if not np.any(valid):
            return np.zeros(depth.shape, dtype=np.float32)

        if self.config.normalize_mode == "clip_scale":
            return np.clip(depth, 0.0, 1.0)

        if self.config.normalize_mode == "none":
            min_value, max_value = self.config.clip_range
        else:
            valid_values = depth[valid]
            min_value = float(valid_values.min())
            max_value = float(valid_values.max())

        if max_value <= min_value:
            return np.zeros(depth.shape, dtype=np.float32)
        display = (depth - min_value) / (max_value - min_value)
        display[~valid] = 0.0
        return np.clip(display, 0.0, 1.0)

    def close(self) -> None:
        if self._cv2 is not None and self._initialized:
            try:
                self._cv2.destroyWindow(self.config.title)
            except Exception:
                pass
        self._initialized = False
