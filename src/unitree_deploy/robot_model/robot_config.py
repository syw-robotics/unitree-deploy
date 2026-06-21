from __future__ import annotations

import copy
import hashlib
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROBOT = "g1"
DEFAULT_TERRAIN = "flat"
DEFAULT_VIEWER = "mujoco"
VIEWER_CHOICES = ("mujoco",)
ROBOT_MODEL_ROOT = Path(__file__).parent
SCENE_ROOT = ROBOT_MODEL_ROOT / "scene"


@dataclass(frozen=True)
class RobotModel:
    """Resolved robot identity and MuJoCo XML path."""

    name: str
    xml_path: Path
    source_xml_path: Path
    config_dir: Path
    terrain: str
    terrain_xml_path: Path


def available_robots() -> list[str]:
    if not ROBOT_MODEL_ROOT.exists():
        return []
    return sorted(
        path.name
        for path in ROBOT_MODEL_ROOT.iterdir()
        if path.is_dir() and path.name != "scene" and not path.name.startswith("__")
    )


def available_terrains() -> list[str]:
    if not SCENE_ROOT.exists():
        return []
    return sorted(path.stem.removesuffix("_terrain") for path in SCENE_ROOT.glob("*_terrain.xml"))


def load_robot_model(
    robot: str = DEFAULT_ROBOT,
    model_xml: str | Path | None = None,
    terrain: str | Path = DEFAULT_TERRAIN,
) -> RobotModel:
    # Most scripts accept --robot for the default layout and --model-xml for one-off XML overrides.
    if model_xml is not None:
        xml_path = Path(model_xml).expanduser().resolve()
        if not xml_path.exists():
            raise FileNotFoundError(f"robot model XML not found: {xml_path}")
        return _robot_model_with_terrain(robot, xml_path, terrain)

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

    return _robot_model_with_terrain(robot, xml_path.resolve(), terrain)


def _robot_model_with_terrain(
    robot: str,
    source_xml_path: Path,
    terrain: str | Path,
) -> RobotModel:
    config_dir = source_xml_path.parent
    terrain_xml_path = resolve_terrain_xml(terrain)
    combined_xml_path = compose_model_scene_xml(source_xml_path, terrain_xml_path, robot)
    return RobotModel(
        name=robot,
        xml_path=combined_xml_path,
        source_xml_path=source_xml_path,
        config_dir=config_dir,
        terrain=terrain_xml_path.stem.removesuffix("_terrain"),
        terrain_xml_path=terrain_xml_path,
    )


def resolve_terrain_xml(terrain: str | Path) -> Path:
    terrain_name = str(terrain).strip()

    terrain_path = Path(terrain_name).expanduser()
    candidates: list[Path]
    if terrain_path.suffix == ".xml" or terrain_path.parent != Path("."):
        candidates = [terrain_path]
    else:
        candidates = [
            SCENE_ROOT / f"{terrain_name}.xml",
            SCENE_ROOT / f"{terrain_name}_terrain.xml",
        ]

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    choices = ", ".join(available_terrains()) or "none"
    raise FileNotFoundError(f"terrain '{terrain_name}' not found. Available terrains: {choices}")


def compose_model_scene_xml(source_xml_path: Path, terrain_xml_path: Path, robot: str) -> Path:
    source_root = _load_mjcf_with_includes(source_xml_path)
    terrain_root = _load_mjcf_with_includes(terrain_xml_path)
    source_bytes = ET.tostring(source_root, encoding="utf-8")
    terrain_bytes = ET.tostring(terrain_root, encoding="utf-8")
    cache_key = b"\0".join(
        [
            source_xml_path.as_posix().encode(),
            terrain_xml_path.as_posix().encode(),
            source_bytes,
            terrain_bytes,
        ]
    )
    digest = hashlib.sha256(cache_key).hexdigest()[:12]

    output_dir = Path(tempfile.gettempdir()) / "unitree-deploy-mujoco"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{robot}_{source_xml_path.stem}_{terrain_xml_path.stem}_{digest}.xml"
    if output_path.exists():
        return output_path

    _absolutize_compiler_dirs(source_root, source_xml_path.parent)
    _absolutize_asset_files(terrain_root, terrain_xml_path.parent)
    _merge_mjcf_roots(source_root, terrain_root)

    tree = ET.ElementTree(source_root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    return output_path


def _load_mjcf_with_includes(xml_path: Path) -> ET.Element:
    root = ET.parse(xml_path).getroot()
    _expand_mjcf_includes(root, xml_path.parent)
    return root


def _expand_mjcf_includes(element: ET.Element, base_dir: Path) -> None:
    expanded_children: list[ET.Element] = []
    for child in list(element):
        if child.tag != "include":
            _expand_mjcf_includes(child, base_dir)
            expanded_children.append(child)
            continue

        include_file = child.get("file")
        if not include_file:
            raise ValueError("<include> element is missing required 'file' attribute")

        include_path = Path(include_file).expanduser()
        if not include_path.is_absolute():
            include_path = (base_dir / include_path).resolve()

        include_root = ET.parse(include_path).getroot()
        if include_root.tag != "mujoco":
            raise ValueError(f"included MJCF file must have a <mujoco> root: {include_path}")

        _expand_mjcf_includes(include_root, include_path.parent)
        expanded_children.extend(copy.deepcopy(include_child) for include_child in include_root)

    element[:] = expanded_children


def _merge_mjcf_roots(robot_root: ET.Element, terrain_root: ET.Element) -> None:
    if robot_root.tag != "mujoco" or terrain_root.tag != "mujoco":
        raise ValueError("robot and terrain XML files must both have a <mujoco> root")

    for terrain_section in terrain_root:
        robot_section = robot_root.find(terrain_section.tag)
        if robot_section is None:
            robot_root.append(copy.deepcopy(terrain_section))
            continue

        for child in terrain_section:
            robot_section.append(copy.deepcopy(child))


def _absolutize_compiler_dirs(root: ET.Element, base_dir: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        return

    for attr in ("meshdir", "texturedir", "assetdir"):
        value = compiler.get(attr)
        if not value:
            continue
        value_path = Path(value).expanduser()
        if not value_path.is_absolute():
            value_path = (base_dir / value_path).resolve()
        compiler.set(attr, value_path.as_posix())


def _absolutize_asset_files(root: ET.Element, base_dir: Path) -> None:
    asset = root.find("asset")
    if asset is None:
        return

    for element in asset.iter():
        value = element.get("file")
        if not value:
            continue
        value_path = Path(value).expanduser()
        if not value_path.is_absolute():
            value_path = (base_dir / value_path).resolve()
        element.set("file", value_path.as_posix())
