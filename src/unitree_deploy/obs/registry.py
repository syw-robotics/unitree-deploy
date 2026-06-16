from __future__ import annotations

from .observation import (
    BaseAngularVelocityObservation,
    CommandObservation,
    JointPositionObservation,
    JointVelocityObservation,
    PreviousActionObservation,
    ProjectedGravityObservation,
)


BUILTIN_OBSERVATION_TYPES = {
    "command": CommandObservation,
    "base_ang_vel": BaseAngularVelocityObservation,
    "base_angvel": BaseAngularVelocityObservation,
    "projected_gravity": ProjectedGravityObservation,
    "joint_pos": JointPositionObservation,
    "joint_vel": JointVelocityObservation,
    "prev_action": PreviousActionObservation,
}
