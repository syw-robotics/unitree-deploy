from __future__ import annotations

import numpy as np

from unitree_deploy.obs.observation import ObservationBase, ObservationContext


def _normalize_quat(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm


def _quat_conjugate(quat) -> np.ndarray:
    q = _normalize_quat(quat)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def _quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = _normalize_quat(a)
    bw, bx, by, bz = _normalize_quat(b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def _quat_to_matrix(quat) -> np.ndarray:
    w, x, y, z = _normalize_quat(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class MotionCommandObservation(ObservationBase):
    def __init__(self, *, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=58, history_len=history_len, dtype=dtype)

    def compute(self, context: ObservationContext) -> np.ndarray:
        del context
        policy = getattr(self, "policy")
        ref = policy.motion_reference
        return np.concatenate([ref["joint_pos"].reshape(-1), ref["joint_vel"].reshape(-1)])


class MotionAnchorPositionObservation(ObservationBase):
    def __init__(self, *, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)

    def compute(self, context: ObservationContext) -> np.ndarray:
        del context
        # The deployed ONNX has the full 160-dim actor layout; no-state runs
        # fill unavailable state-estimation terms with zeros.
        return np.zeros(3, dtype=self.dtype)


class MotionAnchorOrientationObservation(ObservationBase):
    def __init__(self, *, history_len: int, anchor_body_index: int, dtype=np.float32) -> None:
        super().__init__(base_dim=6, history_len=history_len, dtype=dtype)
        self.anchor_body_index = int(anchor_body_index)

    def compute(self, context: ObservationContext) -> np.ndarray:
        policy = getattr(self, "policy")
        ref_quat = policy.motion_reference["body_quat_w"][self.anchor_body_index]
        relative_quat = _quat_mul(_quat_conjugate(context.quat), ref_quat)
        return _quat_to_matrix(relative_quat)[:, :2].reshape(-1)


class BaseLinearVelocityZeroObservation(ObservationBase):
    def __init__(self, *, history_len: int, dtype=np.float32) -> None:
        super().__init__(base_dim=3, history_len=history_len, dtype=dtype)

    def compute(self, context: ObservationContext) -> np.ndarray:
        del context
        # See MotionAnchorPositionObservation: this preserves the exported
        # 160-dim ONNX contract when base linear velocity is unavailable.
        return np.zeros(3, dtype=self.dtype)


OBSERVATION_TYPES = {
    "motion_command": MotionCommandObservation,
    "motion_anchor_pos_b": MotionAnchorPositionObservation,
    "motion_anchor_ori_b": MotionAnchorOrientationObservation,
    "base_lin_vel_zero": BaseLinearVelocityZeroObservation,
}
