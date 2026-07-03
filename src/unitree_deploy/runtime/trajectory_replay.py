from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import viser
from unitree_deploy.robot_model.robot_config import (
    DEFAULT_ROBOT,
    DEFAULT_TERRAIN,
    RobotModel,
    load_robot_model,
)
from unitree_deploy.visualization.scene_config import StandaloneMujocoScene
from unitree_deploy.utils.terminal_status import ComponentConsole


console = ComponentConsole("replay", "bright_blue")


def log(message: str) -> None:
    console.log(message)


@dataclass(frozen=True)
class RuntimeConfig:
    trajectory: Path
    robot: RobotModel | None
    speed: float
    loop: bool


def _load_metadata(trajectory_path: Path) -> dict:
    metadata_path = trajectory_path.with_name("metadata.json")
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def _load_arrays(trajectory_path: Path) -> dict[str, np.ndarray]:
    with np.load(trajectory_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


class TrajectoryReplayApp:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.metadata = _load_metadata(config.trajectory)
        self.arrays = _load_arrays(config.trajectory)
        self.time = np.asarray(self.arrays.get("time", np.zeros(0)), dtype=np.float64)
        if self.time.ndim != 1:
            raise ValueError("trajectory time must be 1D")
        if len(self.time) == 0:
            raise ValueError("trajectory is empty")

        self.robot = config.robot or self._load_robot_from_metadata()
        self.model = mujoco.MjModel.from_xml_path(str(self.robot.xml_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        self.lock = threading.Lock()
        self.alive = True
        self.paused = True
        self.speed = max(float(config.speed), 0.0)
        self.loop = bool(config.loop)
        self.frame = 0
        self.play_start_wall_t = time.perf_counter()
        self.play_start_traj_t = float(self.time[0])
        self.last_render_frame = -1
        self.syncing_gui = False
        self.follow_robot = True  # Keep the camera centered unless the user opts out.
        self.follow_offset = np.array([2.8, -2.8, 1.8], dtype=np.float64)
        self.follow_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        self.server = viser.ViserServer()
        self.scene = StandaloneMujocoScene.create(
            self.server,
            self.model,
            show_camera_frustums=False,
            real_sense_configs=None,
        )
        self.controls = self.build_controls()
        self.server.on_client_connect(self.on_client_connect)
        self.apply_frame(0)
        self._apply_camera_follow()

    def _load_robot_from_metadata(self) -> RobotModel:
        """Prefer the original source XML so replay does not re-compose terrain twice."""
        robot_name = self.metadata.get("robot", DEFAULT_ROBOT)
        terrain = self.metadata.get("terrain", DEFAULT_TERRAIN)
        source_xml = self.metadata.get("source_xml_path") or self.metadata.get("source_xml")
        if source_xml:
            return load_robot_model(robot_name, Path(source_xml), terrain)

        model_xml = self.metadata.get("model_xml")
        model_xml_path = Path(model_xml) if model_xml else None
        return load_robot_model(robot_name, model_xml_path, terrain)

    def build_controls(self) -> dict[str, object]:
        controls: dict[str, object] = {}
        panel = self.server.gui.add_folder("Playback", expand_by_default=True)
        with panel:
            controls["paused"] = self.server.gui.add_checkbox(
                "Paused",
                True,
                hint="Pause or resume playback",
            )
            controls["speed"] = self.server.gui.add_slider(
                "Speed",
                0.0,
                3.0,
                0.1,
                self.speed,
                hint="Playback speed multiplier",
            )
            controls["follow"] = self.server.gui.add_checkbox(
                "Follow",
                True,
                hint="Keep the camera centered on the robot",
            )
            controls["frame"] = self.server.gui.add_slider(
                "Frame",
                0,
                int(len(self.time) - 1),
                1,
                0,
                hint="Seek to an exact frame",
            )
            controls["step_back"] = self.server.gui.add_button("Prev", hint="Step backward one frame")
            controls["step_next"] = self.server.gui.add_button("Next", hint="Step forward one frame")
            controls["restart"] = self.server.gui.add_button("Restart", hint="Jump to the first frame")
            controls["status_text"] = self.server.gui.add_markdown(self.status_markdown())

        controls["paused"].on_update(self.on_paused_changed)
        controls["speed"].on_update(self.on_speed_changed)
        controls["follow"].on_update(self.on_follow_changed)
        controls["frame"].on_update(self.on_frame_changed)
        controls["step_back"].on_click(self.on_step_back)
        controls["step_next"].on_click(self.on_step_next)
        controls["restart"].on_click(self.on_restart)
        return controls

    def status_markdown(self) -> str:
        t = float(self.time[self.frame])
        return (
            f"**Frame**: {self.frame + 1}/{len(self.time)}  \n"
            f"**t**: {t:.3f}s  \n"
            f"**Speed**: {self.speed:.2f}x"
        )

    def on_paused_changed(self, event) -> None:
        if self.syncing_gui:
            return
        with self.lock:
            self.paused = bool(event.target.value)

    def on_speed_changed(self, event) -> None:
        if self.syncing_gui:
            return
        with self.lock:
            self.speed = float(event.target.value)

    def on_follow_changed(self, event) -> None:
        if self.syncing_gui:
            return
        with self.lock:
            self.follow_robot = bool(event.target.value)
        self._apply_camera_follow()

    def on_frame_changed(self, event) -> None:
        if self.syncing_gui:
            return
        self.set_frame(int(event.target.value))

    def on_step_back(self, _event) -> None:
        self.set_frame(max(self.frame - 1, 0))

    def on_step_next(self, _event) -> None:
        self.set_frame(min(self.frame + 1, len(self.time) - 1))

    def on_restart(self, _event) -> None:
        self.set_frame(0)

    def set_frame(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, len(self.time) - 1))
        with self.lock:
            self.frame = frame
            self.play_start_wall_t = time.perf_counter()
            self.play_start_traj_t = float(self.time[frame])
        self._set_gui_value(self.controls["frame"], frame)
        self.apply_frame(frame)

    def _set_gui_value(self, handle, value) -> None:
        # GUI writes re-trigger update callbacks in viser, so guard programmatic syncs.
        self.syncing_gui = True
        try:
            handle.value = value
        finally:
            self.syncing_gui = False

    def apply_frame(self, frame: int) -> None:
        qpos = np.asarray(self.arrays["qpos"][frame], dtype=np.float64)
        qvel = np.asarray(self.arrays["qvel"][frame], dtype=np.float64)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        if "ctrl" in self.arrays and self.arrays["ctrl"].ndim == 2:
            ctrl = np.asarray(self.arrays["ctrl"][frame], dtype=np.float64)
            self.data.ctrl[:] = 0.0
            width = min(self.data.ctrl.shape[0], ctrl.shape[0])
            self.data.ctrl[:width] = ctrl[:width]
        mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_mjdata(self.data)
        self._apply_camera_follow()
        self.controls["status_text"].content = self.status_markdown()

    def on_client_connect(self, client) -> None:
        # New clients should start on the robot instead of an arbitrary default camera.
        self._set_client_camera(client)

    def _base_position(self) -> np.ndarray:
        if self.data.qpos.shape[0] < 3:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.data.qpos[:3], dtype=np.float64)

    def _apply_camera_follow(self) -> None:
        if not self.follow_robot:
            return
        for client in self.server.get_clients().values():
            self._set_client_camera(client)

    def _set_client_camera(self, client) -> None:
        if not self.follow_robot:
            return
        target = self._base_position()
        # Fixed offset keeps the robot centered without requiring per-user camera logic.
        position = target + self.follow_offset
        client.camera.position = position
        client.camera.look_at = target
        client.camera.up_direction = self.follow_up

    def tick(self) -> None:
        with self.lock:
            paused = self.paused
            speed = self.speed
            frame = self.frame
            start_wall = self.play_start_wall_t
            start_traj_t = self.play_start_traj_t

        if paused or speed <= 0.0:
            if self.last_render_frame != frame:
                self.apply_frame(frame)
                self.last_render_frame = frame
            return

        elapsed = (time.perf_counter() - start_wall) * speed
        target_t = start_traj_t + elapsed
        idx = int(np.searchsorted(self.time, target_t, side="right") - 1)
        if idx < 0:
            idx = 0
        if idx >= len(self.time):
            if self.loop:
                self.set_frame(0)
                return
            idx = len(self.time) - 1
            with self.lock:
                self.paused = True
            self._set_gui_value(self.controls["paused"], True)

        if idx != frame:
            with self.lock:
                self.frame = idx
            self._set_gui_value(self.controls["frame"], idx)
            self.apply_frame(idx)
            self.last_render_frame = idx
        elif self.last_render_frame != idx:
            self.apply_frame(idx)
            self.last_render_frame = idx

    def run(self) -> None:
        # Keep the UI responsive by ticking the replay in a short loop.
        log(f"trajectory={self.config.trajectory}")
        log(f"samples={len(self.time)} robot={self.robot.name} terrain={self.robot.terrain}")
        while self.alive:
            self.tick()
            time.sleep(0.02)

    def close(self, *_args) -> None:
        self.alive = False
        console.stop()
        try:
            self.scene.close()
        except Exception:
            pass
        try:
            self.server.stop()
        except Exception:
            pass


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Replay a saved Unitree trajectory in viser.")
    parser.add_argument("trajectory", type=Path, help="Path to trajectory.npz or its directory.")
    parser.add_argument("--robot", default=None, help="Optional robot folder override.")
    parser.add_argument("--model-xml", type=Path, help="Optional MuJoCo XML override.")
    parser.add_argument("--terrain", default=None, help="Optional terrain override.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--loop", action="store_true", help="Loop playback at the end.")
    args = parser.parse_args()

    trajectory = args.trajectory.expanduser().resolve()
    if trajectory.is_dir():
        trajectory = trajectory / "trajectory.npz"
    if not trajectory.exists():
        parser.error(f"trajectory file not found: {trajectory}")

    robot = None
    if args.robot is not None or args.model_xml is not None or args.terrain is not None:
        robot = load_robot_model(
            args.robot or DEFAULT_ROBOT,
            args.model_xml,
            args.terrain or DEFAULT_TERRAIN,
        )

    return RuntimeConfig(
        trajectory=trajectory,
        robot=robot,
        speed=float(args.speed),
        loop=bool(args.loop),
    )


def main() -> None:
    config = parse_args()
    app = TrajectoryReplayApp(config)
    try:
        app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
