#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from unitree_deploy.runtime.sensor.depth_camera.depth_camera import MujocoDepthCamera


MODEL_XML = """
<mujoco model="depth-semantics-check">
  <visual>
    <map znear="0.01" zfar="20"/>
  </visual>
  <worldbody>
    <camera name="depth_camera" pos="0 0 0" quat="1 0 0 0" fovy="58"/>
    <geom name="background" type="plane" pos="0 0 -2" quat="0 1 0 0" size="10 10 0.01"/>
    <geom name="left" type="box" pos="-0.45 0 -0.80" size="0.09 0.09 0.09"/>
    <geom name="right" type="box" pos="0.45 0 -1.00" size="0.09 0.09 0.09"/>
    <geom name="up" type="box" pos="0 0.28 -1.20" size="0.08 0.07 0.08"/>
    <geom name="down" type="box" pos="0 -0.32 -1.40" size="0.08 0.08 0.08"/>
  </worldbody>
</mujoco>
"""


def minimum_location(depth: np.ndarray, rows: slice, columns: slice) -> tuple[int, int, float]:
    region = depth[rows, columns]
    local_row, local_column = np.unravel_index(np.argmin(region), region.shape)
    row_start = rows.start or 0
    column_start = columns.start or 0
    return row_start + local_row, column_start + local_column, float(region[local_row, local_column])


def main() -> None:
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = MujocoDepthCamera(
        model,
        data,
        camera_name="depth_camera",
        height=27,
        width=48,
        fov=58.0,
        fovx=87.0,
        render_scale=2,
        near=0.1,
        far=2.0,
        clip_range=(0.1, 2.0),
        normalize_mode="none",
        fill_invalid=2.0,
    )
    try:
        depth = camera.read_depth()
    finally:
        camera.close()

    if depth.shape != (27, 48):
        raise AssertionError(f"unexpected remapped image shape: {depth.shape}")

    height, width = depth.shape
    middle_rows = slice(height // 3, 2 * height // 3)
    middle_columns = slice(width // 3, 2 * width // 3)
    markers = {
        "left": (minimum_location(depth, middle_rows, slice(0, width // 2)), 0.71),
        "right": (minimum_location(depth, middle_rows, slice(width // 2, width)), 0.91),
        "up": (minimum_location(depth, slice(0, height // 2), middle_columns), 1.12),
        "down": (minimum_location(depth, slice(height // 2, height), middle_columns), 1.32),
    }
    for name, (location, expected_depth) in markers.items():
        if abs(location[2] - expected_depth) > 2.0e-2:
            raise AssertionError(
                f"{name} marker not found in its expected image region: "
                f"minimum={location[2]:.6g} m, expected={expected_depth:.6g} m"
            )

    corner_values = np.concatenate(
        [
            depth[:3, :3].reshape(-1),
            depth[:3, -3:].reshape(-1),
            depth[-3:, :3].reshape(-1),
            depth[-3:, -3:].reshape(-1),
        ]
    )
    max_background_error = float(np.max(np.abs(corner_values - 2.0)))
    if max_background_error > 2.0e-3:
        raise AssertionError(
            "MuJoCo depth is not constant image-plane z on the perpendicular plane: "
            f"max corner error={max_background_error:.6g} m"
        )

    print("PASS: image orientation is upright and not horizontally mirrored")
    print("PASS: perpendicular-plane depth is image-plane distance")
    print(f"marker minima (row, col, depth): {dict((name, value[0]) for name, value in markers.items())}")
    print(f"background max error: {max_background_error:.6g} m")


if __name__ == "__main__":
    main()
