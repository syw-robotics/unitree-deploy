from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROBOT = "g1"
ROBOT_MODEL_ROOT = Path(__file__).parent / "robot_model"


@dataclass(frozen=True)
class RobotModel:
    name: str
    xml_path: Path


def available_robots() -> list[str]:
    if not ROBOT_MODEL_ROOT.exists():
        return []
    return sorted(path.name for path in ROBOT_MODEL_ROOT.iterdir() if path.is_dir())


def load_robot_model(robot: str = DEFAULT_ROBOT, model_xml: str | Path | None = None) -> RobotModel:
    if model_xml is not None:
        xml_path = Path(model_xml).expanduser().resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"robot model XML not found: {xml_path}")
        return RobotModel(name=robot, xml_path=xml_path)

    robot_dir = ROBOT_MODEL_ROOT / robot
    xml_path = robot_dir / f"{robot}.xml"
    if not xml_path.exists():
        xml_files = sorted(robot_dir.glob("*.xml")) if robot_dir.exists() else []
        if len(xml_files) == 1:
            xml_path = xml_files[0]
        else:
            choices = ", ".join(available_robots()) or "none"
            raise FileNotFoundError(
                f"robot '{robot}' model not found at {xml_path}. Available robots: {choices}"
            )

    return RobotModel(name=robot, xml_path=xml_path.resolve())
