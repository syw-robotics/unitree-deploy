from __future__ import annotations

import unittest

import numpy as np

from unitree_deploy.obs.observation import ObservationContext, PreviousActionObservation
from unitree_deploy.policy.base_policy import BasePolicy
from unitree_deploy.runtime.lowlevel_pd import clip_target_q_by_pd_torque
from unitree_deploy.runtime.sensor.depth_camera.depth_camera import DepthCameraBase, MujocoDepthCamera


class _ArrayDepthCamera(DepthCameraBase):
    def __init__(self, image: np.ndarray, **kwargs) -> None:
        self.image = np.asarray(image, dtype=np.float32)
        super().__init__(**kwargs)

    def read_depth(self) -> np.ndarray:
        return self.image.copy()


def _manual_training_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    coordinates = np.arange(-1, 2, dtype=np.float64)
    weights = np.exp(-0.5 * np.square(coordinates / sigma))
    kernel = np.outer(weights, weights)
    kernel /= kernel.sum()
    padded = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
    output = np.empty_like(image, dtype=np.float64)
    for row in range(image.shape[0]):
        for column in range(image.shape[1]):
            output[row, column] = np.sum(padded[row : row + 3, column : column + 3] * kernel)
    return output.astype(np.float32)


class DepthPreprocessingTest(unittest.TestCase):
    def test_clip_scale_uses_training_depth_range(self) -> None:
        image = np.array([[0.0, 0.1, 0.5, 2.0, 3.0]], dtype=np.float32)
        camera = _ArrayDepthCamera(
            image,
            height=1,
            width=5,
            fov=58.0,
            near=0.1,
            far=2.0,
            clip_range=(0.1, 2.0),
            normalize_mode="clip_scale",
            fill_invalid=2.0,
        )
        expected = (np.clip(image, 0.1, 2.0) - 0.1) / 1.9
        np.testing.assert_allclose(camera.capture(), expected, atol=1.0e-7, rtol=0.0)

    def test_gaussian_blur_matches_torch_reflect_padding(self) -> None:
        image = np.arange(20, dtype=np.float32).reshape(4, 5)
        camera = _ArrayDepthCamera(
            image,
            height=4,
            width=5,
            fov=58.0,
            near=0.1,
            far=30.0,
            clip_range=(-100.0, 100.0),
            normalize_mode="none",
            fill_invalid=100.0,
            gaussian_blur={"kernel_size": 3, "sigma": 1.0},
        )
        np.testing.assert_allclose(
            camera.capture(),
            _manual_training_blur(image),
            atol=3.0e-6,
            rtol=0.0,
        )


class CameraIntrinsicsTest(unittest.TestCase):
    def test_remap_coordinates_reproduce_independent_target_fovs(self) -> None:
        height, width = 27, 48
        render_height, render_width = 54, 96
        fovy, fovx = 58.0, 87.0
        source_v, source_u = MujocoDepthCamera.intrinsic_remap_coordinates(
            output_height=height,
            output_width=width,
            render_height=render_height,
            render_width=render_width,
            fovy=fovy,
            fovx=fovx,
        )

        source_focal = render_height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0))
        remapped_x = (source_u + 0.5 - render_width / 2.0) / source_focal
        remapped_y = (source_v + 0.5 - render_height / 2.0) / source_focal
        target_u = np.arange(width, dtype=np.float64) + 0.5
        target_v = np.arange(height, dtype=np.float64) + 0.5
        expected_x = (target_u - width / 2.0) / (width / (2.0 * np.tan(np.deg2rad(fovx) / 2.0)))
        expected_y = (target_v - height / 2.0) / (height / (2.0 * np.tan(np.deg2rad(fovy) / 2.0)))
        np.testing.assert_allclose(
            remapped_x,
            np.broadcast_to(expected_x[None, :], remapped_x.shape),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            remapped_y,
            np.broadcast_to(expected_y[:, None], remapped_y.shape),
            atol=1.0e-12,
            rtol=0.0,
        )


class ActionProcessingTest(unittest.TestCase):
    def _policy(self) -> BasePolicy:
        policy = BasePolicy.__new__(BasePolicy)
        policy.action_dim = 12
        policy.action = np.zeros(12, dtype=np.float32)
        policy.action_scaling = np.full(12, 0.25, dtype=np.float32)
        policy.default_joint_pos_action = np.zeros(12, dtype=np.float32)
        policy.default_joint_pos = np.zeros(12, dtype=np.float32)
        policy.action_to_obs_indices = np.arange(12, dtype=np.int64)
        policy.target_q = np.zeros(12, dtype=np.float32)
        policy.obs_use_scaled_prev_action = False
        policy.raw_action_clip = None
        policy.target_q_clip = (-10.0, 10.0)
        policy.max_torque = np.full(12, 27.0, dtype=np.float32)
        policy.max_torque_target_clip_modes = {"real"}
        policy.previous_action_observation = PreviousActionObservation(action_dim=12, history_len=2)
        policy.previous_action_observation.set_clip((-10.0, 10.0))
        return policy

    def test_previous_action_is_raw_but_observation_clipped(self) -> None:
        policy = self._policy()
        context = ObservationContext(
            q=np.zeros(12, dtype=np.float32),
            dq=np.zeros(12, dtype=np.float32),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro=np.zeros(3, dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),
        )
        policy._target_q_from_policy_action(np.full(12, 100.0, dtype=np.float32), context)
        np.testing.assert_array_equal(policy.previous_action_observation.buffer[-1], 10.0)
        np.testing.assert_array_equal(policy.action, 100.0)

    def test_policy_processing_does_not_apply_low_level_torque_clip(self) -> None:
        policy = self._policy()
        q = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
        dq = np.linspace(-4.0, 4.0, 12, dtype=np.float32)
        context = ObservationContext(
            q=q,
            dq=dq,
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro=np.zeros(3, dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),
        )
        target = policy._target_q_from_policy_action(np.full(12, 100.0, dtype=np.float32), context)
        np.testing.assert_array_equal(target, 10.0)

    def test_torque_clip_mode_selection(self) -> None:
        policy = self._policy()
        self.assertFalse(policy.torque_clip_enabled_for_mode("sim"))
        self.assertTrue(policy.torque_clip_enabled_for_mode("real"))

    def test_low_level_clip_recomputes_for_latest_state(self) -> None:
        target = np.full(12, 10.0, dtype=np.float64)
        kp = np.full(12, 25.0, dtype=np.float64)
        kd = np.full(12, 0.5, dtype=np.float64)
        limit = np.full(12, 27.0, dtype=np.float64)
        target_dq = np.zeros(12, dtype=np.float64)
        tau_ff = np.zeros(12, dtype=np.float64)
        clipped = np.empty(12, dtype=np.float64)

        for q, dq in ((np.zeros(12), np.zeros(12)), (np.full(12, 0.3), np.full(12, 4.0))):
            clip_target_q_by_pd_torque(
                target,
                q,
                dq,
                kp,
                kd,
                limit,
                target_dq,
                tau_ff,
                out=clipped,
            )
            torque = kp * (clipped - q) + kd * (target_dq - dq) + tau_ff
            np.testing.assert_allclose(torque, 27.0, atol=3.0e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
