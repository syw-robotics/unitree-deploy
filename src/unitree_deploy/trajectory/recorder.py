from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from unitree_deploy.trajectory.scene import write_scene_json


TRAJECTORY_FORMAT = "unitree-deploy-trajectory-v1"


class TrajectoryRecorder:
    """Collect MuJoCo states and write a portable trajectory bundle.

    The recorder is intentionally independent of a specific control loop. It can
    be used by an offline policy rollout, sim2sim, or a future real-log replay
    path as long as the caller can provide an MjData snapshot.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        out_dir: Path,
        *,
        metadata: dict[str, Any] | None = None,
        include_body_poses: bool = True,
        write_scene: bool = True,
    ) -> None:
        self.model = model
        self.out_dir = out_dir.expanduser().resolve()
        self.include_body_poses = include_body_poses
        self.write_scene = write_scene
        self.metadata = dict(metadata or {})

        self.time: list[float] = []
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        self.ctrl: list[np.ndarray] = []
        self.command: list[np.ndarray] = []
        self.target_q: list[np.ndarray] = []
        self.body_pos: list[np.ndarray] = []
        self.body_quat: list[np.ndarray] = []
        self.policy_name: list[str] = []

    def sample(
        self,
        data: mujoco.MjData,
        *,
        command: np.ndarray | None = None,
        target_q: np.ndarray | None = None,
        policy_name: str | None = None,
    ) -> None:
        self.time.append(float(data.time))
        self.qpos.append(np.asarray(data.qpos, dtype=np.float64).copy())
        self.qvel.append(np.asarray(data.qvel, dtype=np.float64).copy())
        self.ctrl.append(np.asarray(data.ctrl, dtype=np.float64).copy())
        self.command.append(_vector_or_empty(command))
        self.target_q.append(_vector_or_empty(target_q))
        self.policy_name.append(policy_name or "")
        if self.include_body_poses:
            self.body_pos.append(np.asarray(data.xpos, dtype=np.float64).copy())
            self.body_quat.append(np.asarray(data.xquat, dtype=np.float64).copy())

    def save(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = self.out_dir / "trajectory.npz"
        metadata_path = self.out_dir / "metadata.json"

        np.savez_compressed(
            trajectory_path,
            time=np.asarray(self.time, dtype=np.float64),
            qpos=_stack(self.qpos, (0, int(self.model.nq))),
            qvel=_stack(self.qvel, (0, int(self.model.nv))),
            ctrl=_stack(self.ctrl, (0, int(self.model.nu))),
            command=_stack_ragged_vectors(self.command),
            target_q=_stack_ragged_vectors(self.target_q),
            body_pos=self._body_pos_array(),
            body_quat=self._body_quat_array(),
            policy_name=np.asarray(self.policy_name, dtype=str),
        )
        metadata_path.write_text(
            json.dumps(self._metadata_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.write_scene:
            write_scene_json(self.model, self.out_dir / "scene.json", metadata=self.metadata)
        return trajectory_path

    def _metadata_payload(self) -> dict[str, Any]:
        payload = {
            "format": TRAJECTORY_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "samples": len(self.time),
            "model": {
                "nq": int(self.model.nq),
                "nv": int(self.model.nv),
                "nu": int(self.model.nu),
                "nbody": int(self.model.nbody),
                "timestep": float(self.model.opt.timestep),
            },
            "names": {
                "bodies": _names(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.nbody), "body"),
                "joints": _names(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(self.model.njnt), "joint"),
                "actuators": _names(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(self.model.nu), "actuator"),
            },
            "arrays": {
                "time": "seconds, shape [T]",
                "qpos": "MuJoCo qpos, shape [T, nq]",
                "qvel": "MuJoCo qvel, shape [T, nv]",
                "ctrl": "MuJoCo actuator controls, shape [T, nu]",
                "command": "policy command vector, shape [T, command_dim]",
                "target_q": "policy target joint positions in actuator order when available, shape [T, nu]",
                "body_pos": "world body positions from mj_forward/mj_step, shape [T, nbody, 3]",
                "body_quat": "world body quaternions in wxyz order, shape [T, nbody, 4]",
                "policy_name": "active policy profile name, shape [T]",
            },
        }
        payload.update(self.metadata)
        return payload

    def _body_pos_array(self) -> np.ndarray:
        if self.include_body_poses:
            return _stack(self.body_pos, (0, int(self.model.nbody), 3))
        return np.zeros((len(self.time), 0, 3), dtype=np.float64)

    def _body_quat_array(self) -> np.ndarray:
        if self.include_body_poses:
            return _stack(self.body_quat, (0, int(self.model.nbody), 4))
        return np.zeros((len(self.time), 0, 4), dtype=np.float64)


def _vector_or_empty(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1).copy()


def _stack(values: list[np.ndarray], empty_shape: tuple[int, ...]) -> np.ndarray:
    if not values:
        return np.zeros(empty_shape, dtype=np.float64)
    return np.stack(values, axis=0)


def _stack_ragged_vectors(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros((0, 0), dtype=np.float64)
    width = max((int(value.size) for value in values), default=0)
    out = np.zeros((len(values), width), dtype=np.float64)
    for i, value in enumerate(values):
        out[i, : value.size] = value
    return out


def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int, fallback: str) -> list[str]:
    names = []
    for obj_id in range(count):
        name = mujoco.mj_id2name(model, obj_type, obj_id)
        names.append(name or f"{fallback}_{obj_id}")
    return names
