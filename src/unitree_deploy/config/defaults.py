from __future__ import annotations


DEFAULT_NET = "lo"
DEFAULT_MODE = "sim"

LOWCMD_TOPIC = "rt/lowcmd"
LOWSTATE_TOPIC = "rt/lowstate"
ODOM_TOPIC = "rt/odommodestate"

SIM_HZ = 500
STATE_HZ = 200
RENDER_HZ = 30

BASE_HEIGHT = 1.0
BASE_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)

GYRO_SENSOR_NAMES = ("imu_ang_vel", "imu_gyro")
ACC_SENSOR_NAMES = ("imu_lin_acc", "imu_acc")

BAND_SITES = ("left_gantry_attach_point", "right_gantry_attach_point")
BAND_CLEARANCE = 0.10
BAND_STIFFNESS = 400.0
BAND_DAMPING = 40.0
BAND_STEP = 0.1
BAND_MIN_Z = 0.8
BAND_MAX_Z = 2.2
BAND_MAX_FORCE = 400.0

REMOTE_STICK_SCALE = 1.0
WIRELESS_REMOTE_BUTTON_BITS = {
    "Start": (2, 2),
    "A": (3, 0),
    "X": (3, 2),
}
SIM_REMOTE_BUTTON_KEYS = {
    "a": WIRELESS_REMOTE_BUTTON_BITS["A"],
    "x": WIRELESS_REMOTE_BUTTON_BITS["X"],
    "s": WIRELESS_REMOTE_BUTTON_BITS["Start"],
}


def sim_key_for_button(button: str) -> str:
    target_bit = WIRELESS_REMOTE_BUTTON_BITS[button]
    for key, bit in SIM_REMOTE_BUTTON_KEYS.items():
        if bit == target_bit:
            return key
    raise KeyError(f"no sim keyboard key configured for virtual button {button!r}")

MOVE_TO_DEFAULT_TIME = 2.0
ZERO_TORQUE_STATE = "zero_torque_state"
MOVE_TO_DEFAULT_STATE = "move_to_default_qpos"
DEFAULT_QPOS_STATE = "default_qpos_state"
RUN_POLICY_STATE = "run_policy"

MJSWAN_PORT = 1234
