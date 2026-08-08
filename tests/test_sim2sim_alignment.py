from __future__ import annotations

from multiprocessing import shared_memory
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from unitree_deploy.obs.observation import ObservationContext, PreviousActionObservation
from unitree_deploy.obs.exteropception_observation import DepthObservation
from unitree_deploy.policy.base_policy import BasePolicy, load_policy
from unitree_deploy.runtime.controller import Controller
from unitree_deploy.runtime.lowlevel_pd import clip_target_q_by_pd_torque
from unitree_deploy.runtime.sensor.depth_camera.depth_buffer import (
    DepthObservationBuffer,
    SharedDepthObservationBuffer,
)
from unitree_deploy.runtime.sensor.depth_camera.depth_camera import DepthCameraBase, MujocoDepthCamera
from unitree_deploy.runtime.sensor.depth_camera.depth_ood import (
    DepthOODInjector,
    load_depth_ood_config,
)
from unitree_deploy.utils.yaml_utils import load_yaml


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


class DepthOODInjectionTest(unittest.TestCase):
    @staticmethod
    def _injector() -> DepthOODInjector:
        return DepthOODInjector(
            {
                "enabled": True,
                "seed": 7,
                "patterns": [
                    "local_reflection",
                    "patch_mask",
                    "mask_zero",
                    "gaussian_noise",
                ],
                "local_reflection": {
                    "frame_probability": 1.0,
                    "patch_count_range": [1, 1],
                    "height_range": [3, 3],
                    "width_range": [5, 5],
                    "value_choices": [0.0, 1.0],
                },
                "patch_mask": {
                    "artifacts_prob": 1.0,
                    "max_blocks": 2,
                    "height_mean_std": [3.0, 0.0],
                    "width_mean_std": [4.0, 0.0],
                    "value_choices": [0.0],
                },
                "gaussian_noise": {"mean": 0.0, "std": 0.25},
            }
        )

    def test_p_toggles_random_corruption_pattern(self) -> None:
        injector = self._injector()
        image = np.full((9, 13), 0.5, dtype=np.float32)
        seen_patterns = set()

        self.assertIs(injector.apply(image), image)
        for _ in range(40):
            pattern = injector.toggle_random()
            self.assertIn(pattern, injector.patterns)
            seen_patterns.add(pattern)
            disturbed = injector.apply(image)

            if pattern == "local_reflection":
                self.assertTrue(np.any(disturbed != image))
                self.assertTrue(np.any(disturbed == 0.0) or np.any(disturbed == 1.0))
                self.assertTrue(np.any(disturbed == 0.5))
            elif pattern == "mask_zero":
                np.testing.assert_array_equal(disturbed, 0.0)
            elif pattern == "patch_mask":
                self.assertTrue(np.any(disturbed == 0.0))
                self.assertTrue(np.any(disturbed == 0.5))
            elif pattern == "gaussian_noise":
                self.assertTrue(np.any(disturbed != image))
                self.assertGreaterEqual(float(disturbed.min()), 0.0)
                self.assertLessEqual(float(disturbed.max()), 1.0)

            self.assertEqual(injector.toggle_random(), "off")
            self.assertIs(injector.apply(image), image)

        self.assertEqual(seen_patterns, set(injector.patterns))

    def test_local_reflection_can_be_configured_as_intermittent(self) -> None:
        injector = DepthOODInjector(
            {
                "enabled": True,
                "patterns": ["local_reflection"],
                "local_reflection": {"frame_probability": 0.0},
            }
        )
        image = np.full((9, 13), 0.5, dtype=np.float32)

        self.assertEqual(injector.toggle_random(), "local_reflection")
        self.assertIs(injector.apply(image), image)

    def test_proof_loco_uses_external_corruption_config_without_full_one_mask(self) -> None:
        repository_path = Path(__file__).parents[1]
        sensor_path = repository_path / "ckpt/go2/proof_loco/sensor_depth_camera.yaml"
        sensor_config = load_yaml(sensor_path)["camera"]
        config = load_depth_ood_config(
            sensor_config["ood_injection"],
            sensor_yaml_path=sensor_path,
        )

        self.assertEqual(
            config["patterns"],
            ["local_reflection", "patch_mask", "mask_zero", "gaussian_noise"],
        )
        self.assertEqual(config["local_reflection"]["frame_probability"], 0.05)

    def test_patch_mask_matches_strong_artifact_rectangular_block_semantics(self) -> None:
        injector = DepthOODInjector(
            {
                "enabled": True,
                "seed": 3,
                "patterns": ["patch_mask"],
                "patch_mask": {
                    "artifacts_prob": 1.0,
                    "max_blocks": 1,
                    "height_mean_std": [3.0, 0.0],
                    "width_mean_std": [4.0, 0.0],
                    "value_choices": [0.0],
                },
            }
        )
        image = np.full((9, 13), 0.5, dtype=np.float32)

        self.assertEqual(injector.toggle_random(), "patch_mask")
        disturbed = injector.apply(image)

        self.assertEqual(np.count_nonzero(disturbed == 0.0), 12)
        self.assertEqual(np.count_nonzero(disturbed == 0.5), image.size - 12)


class DepthTimingTest(unittest.TestCase):
    @staticmethod
    def _context() -> ObservationContext:
        return ObservationContext(
            q=np.zeros(12, dtype=np.float32),
            dq=np.zeros(12, dtype=np.float32),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro=np.zeros(3, dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),
        )

    def test_history_advances_every_second_camera_frame(self) -> None:
        depth_buffer = DepthObservationBuffer(height=1, width=1)
        observation = DepthObservation(
            history_len=2,
            height=1,
            width=1,
            depth_buffer=depth_buffer,
            history_skip_frames=2,
        )
        context = self._context()

        depth_buffer.update(np.array([[1.0]], dtype=np.float32))
        observation.prime(context)
        np.testing.assert_array_equal(observation.buffer[:, 0], [1.0, 1.0])

        depth_buffer.update(np.array([[2.0]], dtype=np.float32))
        observation.update(context)
        np.testing.assert_array_equal(observation.buffer[:, 0], [1.0, 1.0])

        depth_buffer.update(np.array([[3.0]], dtype=np.float32))
        observation.update(context)
        np.testing.assert_array_equal(observation.buffer[:, 0], [1.0, 3.0])

    def test_reset_reanchors_history_to_new_depth_frame(self) -> None:
        depth_buffer = DepthObservationBuffer(height=1, width=1)
        observation = DepthObservation(
            history_len=2,
            height=1,
            width=1,
            depth_buffer=depth_buffer,
            history_skip_frames=2,
        )
        context = self._context()

        depth_buffer.update(np.array([[1.0]], dtype=np.float32))
        observation.prime(context)
        observation.reset()
        depth_buffer.update(np.array([[9.0]], dtype=np.float32))
        observation.prime(context)

        np.testing.assert_array_equal(observation.buffer[:, 0], [9.0, 9.0])

    def test_shared_buffer_sequences_identical_frames(self) -> None:
        name = f"unitree_depth_test_{uuid4().hex}"
        writer = SharedDepthObservationBuffer.create(name=name, height=1, width=2)
        try:
            frame = np.array([[0.25, 0.75]], dtype=np.float32)
            writer.update(frame)
            writer.update(frame)
            read_script = """
import sys
from unitree_deploy.runtime.sensor.depth_camera.depth_buffer import SharedDepthObservationBuffer
reader = SharedDepthObservationBuffer.open(name=sys.argv[1], height=1, width=2)
try:
    frame, sequence = reader.get_latest_with_sequence()
    print(sequence, *frame.reshape(-1))
finally:
    reader.close()
"""
            repository_path = Path(__file__).parents[1]
            for _ in range(2):
                result = subprocess.run(
                    [sys.executable, "-c", read_script, name],
                    cwd=repository_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.stdout.strip(), "2 0.25 0.75")
        finally:
            writer.close()

        with self.assertRaises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)


class SimulationResetTest(unittest.TestCase):
    def test_hard_gate_policy_reset_starts_with_gate_open(self) -> None:
        policy_path = Path(__file__).parents[1] / "ckpt/go2/proof_loco/policy_v2-hard-gate.yaml"
        config = load_yaml(policy_path)
        config["gate_control"] = dict(config["gate_control"], mode="policy", decimation=5)
        config["gate_plot"] = dict(config.get("gate_plot") or {}, enabled=False)
        with patch("unitree_deploy.policy.base_policy.load_yaml", return_value=config):
            policy = load_policy(policy_path)

        np.testing.assert_array_equal(policy.gate_state, 1.0)
        np.testing.assert_array_equal(policy.gate_valid, 1.0)
        np.testing.assert_array_equal(policy.gate_open_state, True)
        np.testing.assert_array_equal(policy.last_gate, 1.0)

        policy.h_state.fill(1.0)
        policy.gate_state.fill(1.0)
        policy.gate_valid.fill(1.0)
        policy.gate_open_state.fill(True)
        policy.last_gate.fill(1.0)
        policy._observation_needs_prime = False
        policy.reset()

        np.testing.assert_array_equal(policy.h_state, 0.0)
        np.testing.assert_array_equal(policy.gate_state, 1.0)
        np.testing.assert_array_equal(policy.gate_valid, 1.0)
        np.testing.assert_array_equal(policy.gate_open_state, True)
        np.testing.assert_array_equal(policy.last_gate, 1.0)
        self.assertTrue(policy._observation_needs_prime)

        class _CountingSession:
            def __init__(self, session) -> None:
                self.session = session
                self.calls = 0

            def run(self, *args, **kwargs):
                self.calls += 1
                return self.session.run(*args, **kwargs)

        policy.actor_session = _CountingSession(policy.actor_session)
        policy.gate_hold_session = _CountingSession(policy.gate_hold_session)
        context = ObservationContext(
            q=policy.default_joint_pos.copy(),
            dq=np.zeros(12, dtype=np.float32),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            gyro=np.zeros(3, dtype=np.float32),
            command=np.zeros(3, dtype=np.float32),
        )
        for _ in range(4):
            policy.compute_target_q(context)
        self.assertEqual(policy.actor_session.calls, 0)
        self.assertEqual(policy.gate_hold_session.calls, 4)
        np.testing.assert_array_equal(policy.last_gate, 1.0)

        for _ in range(6):
            policy.compute_target_q(context)
        self.assertEqual(policy.actor_session.calls, 2)
        self.assertEqual(policy.gate_hold_session.calls, 8)

        for observation in policy.observation.observations:
            if hasattr(observation, "sensor_buffer") and observation.sensor_buffer is not None:
                observation.sensor_buffer.close()

    def test_reset_generation_resets_policy_outside_controller_lock(self) -> None:
        controller = Controller.__new__(Controller)
        controller.lock = threading.Lock()
        controller.sim_reset_sequence = None
        controller.policy_reset_pending = False

        class _Policy:
            reset_count = 0

            def reset(self) -> None:
                self.reset_count += 1
                self.lock_was_free = controller.lock.acquire(blocking=False)
                if self.lock_was_free:
                    controller.lock.release()

        policy = _Policy()
        controller.policy_manager = SimpleNamespace(active=SimpleNamespace(policy=policy))

        controller.observe_sim_reset_sequence(4)
        self.assertFalse(controller.policy_reset_pending)
        controller.observe_sim_reset_sequence(5)
        self.assertTrue(controller.policy_reset_pending)

        self.assertTrue(controller.reset_policy_if_requested())
        self.assertEqual(policy.reset_count, 1)
        self.assertTrue(policy.lock_was_free)
        self.assertFalse(controller.reset_policy_if_requested())


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

    def test_depth_render_uses_sensor_clip_planes_without_changing_viewer_model(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            """
<mujoco>
  <visual><map znear="0.01" zfar="50"/></visual>
  <worldbody>
    <camera name="depth"/>
    <geom type="box" pos="0 0 -0.3" size="0.1 0.1 0.1"/>
    <geom type="sphere" pos="24 0 0" size="0.01" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""
        )
        data = mujoco.MjData(model)
        camera = MujocoDepthCamera(
            model,
            data,
            "depth",
            height=1,
            width=1,
            fov=58.0,
            near=0.1,
            far=2.0,
            clip_range=(0.1, 2.0),
            normalize_mode="none",
            fill_invalid=2.0,
        )
        original_clip = (float(model.vis.map.znear), float(model.vis.map.zfar))

        class _Renderer:
            def update_scene(self, mj_data, *, camera) -> None:
                del mj_data, camera
                self.update_clip = (
                    float(model.vis.map.znear * model.stat.extent),
                    float(model.vis.map.zfar * model.stat.extent),
                )

            def render(self) -> np.ndarray:
                self.render_clip = (
                    float(model.vis.map.znear * model.stat.extent),
                    float(model.vis.map.zfar * model.stat.extent),
                )
                return np.ones((1, 1), dtype=np.float32)

        renderer = _Renderer()
        camera._renderer = renderer
        camera.read_depth()

        np.testing.assert_allclose(renderer.update_clip, (0.1, 2.0), atol=1.0e-7, rtol=0.0)
        np.testing.assert_allclose(renderer.render_clip, (0.1, 2.0), atol=1.0e-7, rtol=0.0)
        self.assertEqual((float(model.vis.map.znear), float(model.vis.map.zfar)), original_clip)


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
