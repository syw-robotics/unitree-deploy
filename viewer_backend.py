from __future__ import annotations

import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import mujoco
import mujoco.viewer

from robot_config import RobotModel


MJSWAN_PORT = 1234


class ViewerBackend(Protocol):
    """Small lifecycle adapter so SimBridge does not branch on viewer type."""

    def run(self, simulate: Callable[[], None]) -> None:
        ...

    def sync(self) -> bool:
        ...


class MujocoViewerBackend:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        sim_hz: int,
        render_hz: int,
    ) -> None:
        self.model = model
        self.data = data
        self.viewer = None
        self.viewer_tick = 0
        self.viewer_decim = max(1, sim_hz // render_hz)

    def run(self, simulate: Callable[[], None]) -> None:
        with mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            self.viewer = viewer
            simulate()
            self.viewer = None

    def sync(self) -> bool:
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        self.viewer_tick += 1
        if self.viewer_tick % self.viewer_decim == 0:
            self.viewer.sync()
        return True


class MjswanViewerBackend:
    def __init__(
        self,
        robot: RobotModel,
        *,
        log: Callable[[str], None],
        port: int = MJSWAN_PORT,
    ) -> None:
        self.robot = robot
        self.log = log
        self.port = port

    def run(self, simulate: Callable[[], None]) -> None:
        # mjswan serves an independent browser-side scene, so SimBridge can keep stepping.
        threading.Thread(
            target=self.launch,
            daemon=True,
            name="mjswan-viewer",
        ).start()
        simulate()

    def sync(self) -> bool:
        return True

    def launch(self) -> None:
        import mjswan

        output_dir = (
            Path(tempfile.gettempdir())
            / "unitree-deploy-mjswan"
            / f"{self.robot.name}_{self.robot.terrain}"
        )
        self.log(
            "starting mjswan viewer server; this is a browser-side MuJoCo scene "
            "and does not mirror SimBridge state yet"
        )
        builder = mjswan.Builder()
        project = builder.add_project(name=f"{self.robot.name} deploy")
        scene = project.add_scene(
            spec=mujoco.MjSpec.from_file(str(self.robot.xml_path)),
            name=f"{self.robot.name} {self.robot.terrain}",
        )
        scene.set_viewer_config(
            mjswan.ViewerConfig(
                lookat=(0.0, 0.0, 0.8),
                distance=4.0,
                elevation=-25.0,
                azimuth=45.0,
                origin_type=mjswan.ViewerConfig.OriginType.WORLD,
            )
        )
        app = builder.build(output_dir=output_dir)
        self.log(f"mjswan viewer: http://localhost:{self.port}")
        app.launch(port=self.port, open_browser=False)


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
    if viewer == "mujoco":
        return MujocoViewerBackend(model, data, sim_hz=sim_hz, render_hz=render_hz)
    if viewer == "mjswan":
        return MjswanViewerBackend(robot, log=log)
    raise ValueError(f"unsupported viewer backend: {viewer}")
