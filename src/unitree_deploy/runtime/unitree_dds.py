from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from unitree_deploy.config.defaults import (
    GO_ROBOTS,
    HG_MODE_MACHINE,
    HG_MODE_PR,
    HG_ROBOTS,
    LOWCMD_TOPIC,
    LOWSTATE_TOPIC,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as GoLowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as GoLowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HGLowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HGLowState_


@dataclass(frozen=True)
class LowLevelDDS:
    """Low-level DDS topic and IDL type used by a Unitree robot."""

    type: str
    lowcmd_topic: str
    lowstate_topic: str
    lowcmd_type: type
    lowstate_type: type
    make_lowcmd: Callable[[], object]
    make_lowstate: Callable[[], object]
    has_mode_fields: bool


def make_go_lowcmd_msg() -> object:
    msg = unitree_go_msg_dds__LowCmd_()
    msg.head[0] = 0xFE
    msg.head[1] = 0xEF
    msg.level_flag = 0xFF
    return msg


def make_hg_lowcmd_msg() -> object:
    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr = HG_MODE_PR
    msg.mode_machine = HG_MODE_MACHINE
    return msg


def resolve_low_level_dds(robot: str | None) -> LowLevelDDS:
    robot_name = (robot or "").strip().lower()
    if robot_name in GO_ROBOTS:
        return LowLevelDDS(
            type="unitree_go",
            lowcmd_topic=LOWCMD_TOPIC,
            lowstate_topic=LOWSTATE_TOPIC,
            lowcmd_type=GoLowCmd_,
            lowstate_type=GoLowState_,
            make_lowcmd=make_go_lowcmd_msg,
            make_lowstate=unitree_go_msg_dds__LowState_,
            has_mode_fields=False,  # whether need to set mode_pr and mode_machine
        )

    if robot_name in HG_ROBOTS:
        return LowLevelDDS(
            type="unitree_hg",
            lowcmd_topic=LOWCMD_TOPIC,
            lowstate_topic=LOWSTATE_TOPIC,
            lowcmd_type=HGLowCmd_,
            lowstate_type=HGLowState_,
            make_lowcmd=make_hg_lowcmd_msg,
            make_lowstate=unitree_hg_msg_dds__LowState_,
            has_mode_fields=True,  # whether need to set mode_pr and mode_machine
        )
    else:
        raise ValueError(f"unknown robot name {robot_name!r}, cannot resolve low-level DDS")
