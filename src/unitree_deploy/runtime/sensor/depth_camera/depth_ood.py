from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

from unitree_deploy.utils.yaml_utils import load_yaml


def load_depth_ood_config(config_ref, *, sensor_yaml_path: Path) -> dict | None:
    """Resolve an inline OOD mapping or a YAML path relative to the sensor config."""
    if config_ref is None:
        return None
    if isinstance(config_ref, dict):
        return dict(config_ref)
    if not isinstance(config_ref, (str, Path)):
        raise TypeError("camera.ood_injection must be a mapping or YAML path")

    config_path = Path(config_ref).expanduser()
    if not config_path.is_absolute():
        config_path = sensor_yaml_path.parent / config_path
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    if "depth_corruption" in config:
        config = config["depth_corruption"]
    if not isinstance(config, dict):
        raise TypeError(f"{config_path}: depth_corruption must be a mapping")
    return dict(config)


class DepthOODInjector:
    """Inject selectable OOD disturbances into preprocessed depth frames."""

    VALID_PATTERNS = (
        "local_reflection",
        "patch_mask",
        "mask_zero",
        "gaussian_noise",
    )

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        if not isinstance(config, dict):
            raise TypeError("camera.ood_injection must be a mapping")

        self.enabled = bool(config.get("enabled", False))
        self.key = str(config.get("key", "p")).strip().lower()
        if len(self.key) != 1:
            raise ValueError("camera.ood_injection.key must be one character")

        patterns = config.get("patterns", self.VALID_PATTERNS)
        if not isinstance(patterns, (list, tuple)) or not patterns:
            raise ValueError("camera.ood_injection.patterns must be a non-empty sequence")
        self.patterns = tuple(str(pattern).strip().lower() for pattern in patterns)
        invalid_patterns = set(self.patterns) - set(self.VALID_PATTERNS)
        if invalid_patterns:
            raise ValueError(
                "unsupported camera.ood_injection pattern(s): "
                + ", ".join(sorted(invalid_patterns))
            )
        if len(set(self.patterns)) != len(self.patterns):
            raise ValueError("camera.ood_injection.patterns must not contain duplicates")

        self._mode = "off"
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(config.get("seed"))

        output_range = np.asarray(config.get("output_range", [0.0, 1.0]), dtype=np.float32)
        if output_range.shape != (2,) or not np.all(np.isfinite(output_range)):
            raise ValueError("camera.ood_injection.output_range must contain two finite values")
        if output_range[0] >= output_range[1]:
            raise ValueError("camera.ood_injection.output_range must be increasing")
        self.output_min = float(output_range[0])
        self.output_max = float(output_range[1])

        reflection = config.get("local_reflection", {}) or {}
        if not isinstance(reflection, dict):
            raise TypeError("camera.ood_injection.local_reflection must be a mapping")
        self.reflection_frame_probability = float(
            reflection.get("frame_probability", 1.0)
        )
        if (
            not np.isfinite(self.reflection_frame_probability)
            or not 0.0 <= self.reflection_frame_probability <= 1.0
        ):
            raise ValueError(
                "camera.ood_injection.local_reflection.frame_probability "
                "must be finite and within [0, 1]"
            )
        self.reflection_patch_count = self._integer_range(
            reflection.get("patch_count_range", [1, 3]),
            "camera.ood_injection.local_reflection.patch_count_range",
        )
        self.reflection_height = self._integer_range(
            reflection.get("height_range", [3, 8]),
            "camera.ood_injection.local_reflection.height_range",
        )
        self.reflection_width = self._integer_range(
            reflection.get("width_range", [4, 12]),
            "camera.ood_injection.local_reflection.width_range",
        )
        reflection_values = np.asarray(
            reflection.get("value_choices", [self.output_min, self.output_max]),
            dtype=np.float32,
        ).reshape(-1)
        if reflection_values.size == 0 or not np.all(np.isfinite(reflection_values)):
            raise ValueError(
                "camera.ood_injection.local_reflection.value_choices must contain finite values"
            )
        if np.any(reflection_values < self.output_min) or np.any(
            reflection_values > self.output_max
        ):
            raise ValueError(
                "camera.ood_injection.local_reflection.value_choices must be within output_range"
            )
        self.reflection_values = reflection_values

        patch_mask = config.get("patch_mask", {}) or {}
        if not isinstance(patch_mask, dict):
            raise TypeError("camera.ood_injection.patch_mask must be a mapping")
        self.patch_probability = float(patch_mask.get("artifacts_prob", 0.01))
        if not np.isfinite(self.patch_probability) or self.patch_probability < 0.0:
            raise ValueError(
                "camera.ood_injection.patch_mask.artifacts_prob must be finite and non-negative"
            )
        max_blocks = patch_mask.get("max_blocks", 3)
        self.patch_max_blocks = int(max_blocks)
        if self.patch_max_blocks < 0 or self.patch_max_blocks != max_blocks:
            raise ValueError("camera.ood_injection.patch_mask.max_blocks must be a non-negative integer")
        self.patch_height_mean_std = self._mean_std(
            patch_mask.get("height_mean_std", [8.0, 2.0]),
            "camera.ood_injection.patch_mask.height_mean_std",
        )
        self.patch_width_mean_std = self._mean_std(
            patch_mask.get("width_mean_std", [8.0, 2.0]),
            "camera.ood_injection.patch_mask.width_mean_std",
        )
        patch_values = np.asarray(
            patch_mask.get("value_choices", [self.output_min, self.output_max]),
            dtype=np.float32,
        ).reshape(-1)
        if patch_values.size == 0 or not np.all(np.isfinite(patch_values)):
            raise ValueError(
                "camera.ood_injection.patch_mask.value_choices must contain finite values"
            )
        if np.any(patch_values < self.output_min) or np.any(patch_values > self.output_max):
            raise ValueError(
                "camera.ood_injection.patch_mask.value_choices must be within output_range"
            )
        self.patch_values = patch_values

        gaussian = config.get("gaussian_noise", {}) or {}
        if not isinstance(gaussian, dict):
            raise TypeError("camera.ood_injection.gaussian_noise must be a mapping")
        self.gaussian_mean = float(gaussian.get("mean", 0.0))
        self.gaussian_std = float(gaussian.get("std", 0.25))
        if not np.isfinite(self.gaussian_mean):
            raise ValueError("camera.ood_injection.gaussian_noise.mean must be finite")
        if not np.isfinite(self.gaussian_std) or self.gaussian_std < 0.0:
            raise ValueError("camera.ood_injection.gaussian_noise.std must be finite and non-negative")

    @staticmethod
    def _integer_range(value, field: str) -> tuple[int, int]:
        values = np.asarray(value)
        if values.shape != (2,):
            raise ValueError(f"{field} must contain two integers")
        lower, upper = int(values[0]), int(values[1])
        if lower < 1 or lower > upper or np.any(values != [lower, upper]):
            raise ValueError(f"{field} must be an increasing positive integer range")
        return lower, upper

    @staticmethod
    def _mean_std(value, field: str) -> tuple[float, float]:
        values = np.asarray(value, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{field} must contain finite mean and std values")
        mean, std = float(values[0]), float(values[1])
        if mean <= 0.0 or std < 0.0:
            raise ValueError(f"{field} must have positive mean and non-negative std")
        return mean, std

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def toggle_random(self) -> str:
        with self._lock:
            if not self.enabled:
                return self._mode
            if self._mode == "off":
                self._mode = str(self._rng.choice(self.patterns))
            else:
                self._mode = "off"
            return self._mode

    def apply(self, depth_image: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth OOD injection expects a 2-D image, got shape={depth.shape}")

        with self._lock:
            mode = self._mode
            if not self.enabled or mode == "off":
                return depth
            if mode == "mask_zero":
                return np.full_like(depth, self.output_min)
            if mode == "gaussian_noise":
                noise = self._rng.normal(self.gaussian_mean, self.gaussian_std, depth.shape)
                disturbed = depth + noise.astype(np.float32)
                return np.clip(disturbed, self.output_min, self.output_max).astype(
                    np.float32,
                    copy=False,
                )
            if mode == "local_reflection":
                if self._rng.random() >= self.reflection_frame_probability:
                    return depth
                return self._apply_local_reflection(depth)
            if mode == "patch_mask":
                return self._apply_patch_mask(depth)
        raise RuntimeError(f"unhandled depth OOD mode: {mode}")

    def _apply_local_reflection(self, depth: np.ndarray) -> np.ndarray:
        disturbed = depth.copy()
        height, width = disturbed.shape
        patch_count = self._sample_integer(self.reflection_patch_count)
        rows, columns = np.ogrid[:height, :width]

        for _ in range(patch_count):
            patch_height = min(self._sample_integer(self.reflection_height), height)
            patch_width = min(self._sample_integer(self.reflection_width), width)
            center_row = int(self._rng.integers(0, height))
            center_column = int(self._rng.integers(0, width))
            radius_row = max(patch_height / 2.0, 0.5)
            radius_column = max(patch_width / 2.0, 0.5)
            mask = (
                np.square((rows - center_row) / radius_row)
                + np.square((columns - center_column) / radius_column)
                <= 1.0
            )
            value = float(self._rng.choice(self.reflection_values))
            disturbed[mask] = value

        return disturbed

    def _sample_integer(self, bounds: tuple[int, int]) -> int:
        return int(self._rng.integers(bounds[0], bounds[1] + 1))

    def _apply_patch_mask(self, depth: np.ndarray) -> np.ndarray:
        disturbed = depth.copy()
        height, width = disturbed.shape
        expected_blocks = self.patch_probability * height * width
        if expected_blocks <= 0.0 or self.patch_max_blocks == 0:
            return disturbed

        block_count = int(
            np.clip(self._rng.poisson(expected_blocks), 1, self.patch_max_blocks)
        )
        for _ in range(block_count):
            block_height = self._sample_patch_size(self.patch_height_mean_std, height)
            block_width = self._sample_patch_size(self.patch_width_mean_std, width)
            top = int(self._rng.integers(0, height - block_height + 1))
            left = int(self._rng.integers(0, width - block_width + 1))
            value = float(self._rng.choice(self.patch_values))
            disturbed[top : top + block_height, left : left + block_width] = value

        return np.clip(disturbed, self.output_min, self.output_max).astype(
            np.float32,
            copy=False,
        )

    def _sample_patch_size(self, mean_std: tuple[float, float], max_size: int) -> int:
        size = int(np.rint(self._rng.normal(mean_std[0], mean_std[1])))
        return int(np.clip(size, 1, max_size))
