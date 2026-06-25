from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


SCENE_FORMAT = "unitree-deploy-mujoco-scene-v1"


def write_scene_json(
    model: mujoco.MjModel,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scene_payload(model, metadata=metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def scene_payload(model: mujoco.MjModel, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    mesh_ids = sorted(
        {
            int(model.geom_dataid[geom_id])
            for geom_id in range(int(model.ngeom))
            if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            and int(model.geom_dataid[geom_id]) >= 0
        }
    )
    return {
        "format": SCENE_FORMAT,
        "metadata": dict(metadata or {}),
        "bodies": [
            {
                "id": body_id,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}",
            }
            for body_id in range(int(model.nbody))
        ],
        "geoms": [geom_payload(model, geom_id) for geom_id in range(int(model.ngeom))],
        "meshes": {str(mesh_id): mesh_payload(model, mesh_id) for mesh_id in mesh_ids},
    }


def geom_payload(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    geom_type = int(model.geom_type[geom_id])
    return {
        "id": geom_id,
        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}",
        "body_id": int(model.geom_bodyid[geom_id]),
        "group": int(model.geom_group[geom_id]),
        "type": geom_type_name(geom_type),
        "mesh_id": int(model.geom_dataid[geom_id]),
        "size": np.asarray(model.geom_size[geom_id], dtype=np.float64).tolist(),
        "pos": np.asarray(model.geom_pos[geom_id], dtype=np.float64).tolist(),
        "quat": np.asarray(model.geom_quat[geom_id], dtype=np.float64).tolist(),
        "rgba": geom_rgba(model, geom_id).tolist(),
    }


def mesh_payload(model: mujoco.MjModel, mesh_id: int) -> dict[str, Any]:
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    return {
        "id": mesh_id,
        "vertices": np.asarray(model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float64).tolist(),
        "faces": np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.int64).tolist(),
    }


def geom_rgba(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    mat_id = int(model.geom_matid[geom_id])
    if 0 <= mat_id < int(model.nmat):
        rgba = np.asarray(model.mat_rgba[mat_id], dtype=np.float64)
    else:
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float64)
    if np.allclose(rgba, 0.0):
        rgba = np.array([0.66, 0.68, 0.68, 1.0], dtype=np.float64)
    return np.clip(rgba, 0.0, 1.0)


def geom_type_name(geom_type: int) -> str:
    return mujoco.mjtGeom(geom_type).name.removeprefix("mjGEOM_").lower()
