from __future__ import annotations

import numpy as np


def clip_target_q_by_pd_torque(
    target_q: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    max_torque: np.ndarray,
    target_dq: np.ndarray,
    tau_ff: np.ndarray,
    *,
    out: np.ndarray,
) -> np.ndarray:
    """Clip position targets so the current low-level PD command stays in bounds."""
    if np.any(kp <= 0.0):
        raise ValueError("kp must be positive when target-position torque clipping is enabled")
    pd_bias = kd * (target_dq - dq) + tau_ff
    lower = q + (-max_torque - pd_bias) / kp
    upper = q + (max_torque - pd_bias) / kp
    np.clip(target_q, lower, upper, out=out)
    return out
