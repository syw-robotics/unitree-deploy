from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from unitree_deploy.runtime.sensor.depth_camera.config import (
    load_sensor_camera_config,
    write_model_xml_with_sensor_camera,
)


REPOSITORY_PATH = Path(__file__).parents[1]
MODEL_PATH = (
    REPOSITORY_PATH
    / "src"
    / "unitree_deploy"
    / "robot_model"
    / "go2"
    / "go2.xml"
)


def _vector(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(value) for value in element.attrib[attribute].split())


class Go2ModelGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = ET.parse(MODEL_PATH).getroot()

    def _named(self, tag: str, name: str) -> ET.Element:
        element = self.root.find(f".//{tag}[@name='{name}']")
        self.assertIsNotNone(element, f"missing {tag} named {name}")
        return element

    def assertVectorEqual(
        self,
        element: ET.Element,
        attribute: str,
        expected: tuple[float, ...],
    ) -> None:
        self.assertEqual(_vector(element, attribute), expected)

    def test_realsense_mount_matches_training_urdf(self) -> None:
        rack = self._named("body", "realsense_rack_link")
        self.assertVectorEqual(rack, "pos", (0.286, 0.0, 0.081))
        self.assertVectorEqual(rack, "quat", (0.0, 0.0, 0.0, 1.0))

        mesh = self._named("mesh", "realsense_rack")
        self.assertEqual(mesh.attrib["file"], "cam_rack_2_30.stl")
        self.assertTrue((MODEL_PATH.parent / "assets" / mesh.attrib["file"]).is_file())

        expected_collisions = {
            "realsense_rack_collision": (
                (0.025, 0.05, 0.09),
                (-0.03, 0.0, -0.05),
            ),
            "realsense_rack_bottom_front_collision": (
                (0.02, 0.065, 0.015),
                (0.18, 0.0, -0.16),
            ),
            "realsense_rack_bottom_rear_collision": (
                (0.015, 0.065, 0.01),
                (0.395, 0.0, -0.15),
            ),
        }
        for name, (size, position) in expected_collisions.items():
            geom = self._named("geom", name)
            self.assertEqual(geom.attrib["type"], "box")
            self.assertVectorEqual(geom, "size", size)
            self.assertVectorEqual(geom, "pos", position)

    def test_depth_camera_matches_training_pose(self) -> None:
        depth_origin = self._named("body", "realsense_depth_origin_link")
        self.assertVectorEqual(depth_origin, "pos", (0.335, 0.015, 0.088))
        camera = self._named("camera", "depth_camera")
        self.assertIs(depth_origin.find("camera"), camera)
        self.assertVectorEqual(
            camera,
            "quat",
            (0.612372436, 0.353553391, -0.353553391, -0.612372436),
        )
        self.assertEqual(float(camera.attrib["fovy"]), 58.0)

    def test_robot_collisions_match_training_urdf(self) -> None:
        base_collision = self._named("geom", "base_collision")
        self.assertVectorEqual(base_collision, "size", (0.2, 0.1, 0.057))
        self.assertVectorEqual(base_collision, "pos", (0.05, 0.0, 0.0))

        for leg in ("FL", "FR", "RL", "RR"):
            thigh = self._named("geom", f"{leg}_thigh_collision")
            self.assertVectorEqual(thigh, "size", (0.1, 0.01225, 0.017))
            self.assertVectorEqual(thigh, "pos", (-0.015, 0.0, -0.06))

            calf = self._named("geom", f"{leg}_calf_collision")
            self.assertVectorEqual(calf, "size", (0.0125, 0.0125, 0.105))
            self.assertVectorEqual(calf, "pos", (0.005, 0.0, -0.085))

    def test_model_loads_in_mujoco(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera"),
            0,
        )

    def test_runtime_camera_model_resolves_original_assets(self) -> None:
        sensor_path = (
            REPOSITORY_PATH
            / "ckpt"
            / "go2"
            / "naive_vision_loco"
            / "sensor_depth_camera.yaml"
        )
        camera_config = load_sensor_camera_config(sensor_path)
        self.assertIsNotNone(camera_config)
        generated_path = write_model_xml_with_sensor_camera(
            MODEL_PATH,
            sensor_path,
            camera_config,
        )

        generated_root = ET.parse(generated_path).getroot()
        compiler = generated_root.find("compiler")
        self.assertIsNotNone(compiler)
        self.assertEqual(
            Path(compiler.attrib["meshdir"]),
            (MODEL_PATH.parent / "assets").resolve(),
        )

        model = mujoco.MjModel.from_xml_path(str(generated_path))
        self.assertEqual(model.ncam, 1)
        camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "depth_camera",
        )
        self.assertGreaterEqual(camera_id, 0)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.assertEqual(int(model.cam_bodyid[camera_id]), base_id)

        static_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        static_data = mujoco.MjData(static_model)
        generated_data = mujoco.MjData(model)
        mujoco.mj_forward(static_model, static_data)
        mujoco.mj_forward(model, generated_data)
        static_camera_id = mujoco.mj_name2id(
            static_model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "depth_camera",
        )
        np.testing.assert_allclose(
            static_data.cam_xpos[static_camera_id],
            generated_data.cam_xpos[camera_id],
            atol=1.0e-9,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            static_data.cam_xmat[static_camera_id],
            generated_data.cam_xmat[camera_id],
            atol=1.0e-9,
            rtol=0.0,
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "realsense_rack_collision"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
