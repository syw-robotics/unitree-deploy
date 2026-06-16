from __future__ import annotations

import argparse
from pathlib import Path

from unitree_deploy.robot_model.robot_config import DEFAULT_ROBOT, available_robots


CUSTOM_POLICY = '''from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from unitree_deploy.obs.observation import ObservationBase, ObservationContext
from unitree_deploy.policy.base_policy import BasePolicy, ObservationRegistry


class CustomPolicy(BasePolicy):
    """Only needed when BasePolicy action logic is not enough."""

    def __init__(
        self,
        policy_yaml_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        observation_types: ObservationRegistry | None = None,
    ) -> None:
        super().__init__(
            policy_yaml_path,
            providers=providers,
            observation_types=observation_types,
        )

    def reset(self) -> None:
        """Override when the policy has extra recurrent/internal state."""

        super().reset()

    def _build_observation(self, observation_spec: dict) -> ObservationBase:
        """Override only for special constructor wiring not expressible in YAML."""

        return super()._build_observation(observation_spec)

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        """Override to customize inference or action post-processing."""

        return super().compute_target_q(context)
'''


CUSTOM_OBSERVATIONS = '''from __future__ import annotations

import numpy as np

from unitree_deploy.obs.observation import ObservationBase, ObservationContext


class CustomObservation(ObservationBase):
    """Example observation term.

    Replace this with the tensors expected by your exported policy.
    """

    def __init__(self, *, history_len: int, scale: float = 1.0) -> None:
        super().__init__(base_dim=3, history_len=history_len)
        self.scale = float(scale)

    def _compute_current(self, context: ObservationContext) -> np.ndarray:
        return np.asarray(context.command, dtype=self.dtype).reshape(3) * self.scale

'''


POLICY_YAML = '''# Fill in the deployment values exported with your policy.
# Put the ONNX file in this directory, or make policy_path point to its relative path.
policy_path: "policy.onnx"

# Only uncomment this if BasePolicy action post-processing is not enough.
# policy_class: "custom_policy:CustomPolicy"

policy_step_dt: 0.02
physics_dt: 0.002
decimation: 10

obs_joint_order: []
action_joint_order: []
sdk_joint_order: []
default_qpos: []

policy_input_name: "policy"
policy_output_name: "action"

observation_types:
  custom_obs: "custom_observations:CustomObservation"

observations:
  - type: command
    history_len: 1
    command_range:
      - [-1.0, 1.0]
      - [-1.0, 1.0]
      - [-2.0, 2.0]
  - type: base_ang_vel
    history_len: 1
  - type: projected_gravity
    history_len: 1
  - type: joint_pos
    history_len: 1
  - type: joint_vel
    history_len: 1
  - type: prev_action
    history_len: 1
  - type: custom_obs
    history_len: 1
    params:
      scale: 1.0

action_dim: 0
action_clip: 1.0
action_scale: []

kp_policy: []
kd_policy: []
kp_fixed_stand: []
kd_fixed_stand: []
kd_damping: []
'''


README = '''# Unitree Deploy Plugin Template

这是一个 deployment/plugin 目录模板。大多数情况下只需要准备 `policy.yaml` 和 ONNX 文件，不用继承 policy。

## 必需文件

- `policy.yaml`: policy/action/joint/order/ONNX/observation 配置。
- `policy.onnx`: 默认 ONNX 文件名，也可以在 `policy.yaml` 的 `policy_path` 中改成其他相对路径。

## 可选文件

- `custom_observations.py`: 自定义 observation。
- `custom_policy.py`: 只有需要改 action 后处理或推理逻辑时才需要。

如果确实需要改 policy action 后处理，再额外加入：

```yaml
policy_class: "custom_policy:CustomPolicy"
```

`policy.yaml` 里的 `observation_types` 会注册自定义 observation；`params` 会作为关键字参数传给 observation 构造函数。
'''


FILES = {
    "policy.yaml": POLICY_YAML,
    "custom_policy.py": CUSTOM_POLICY,
    "custom_observations.py": CUSTOM_OBSERVATIONS,
    "README.md": README,
}


def write_template(destination: Path, *, force: bool = False) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in FILES.items():
        path = destination / name
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass --force to overwrite")
        path.write_text(content, encoding="utf-8")


def prompt_text(label: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default


def resolve_destination(args: argparse.Namespace) -> Path:
    if args.interactive or args.destination is None:
        base_path = Path(
            prompt_text("Base path", default=str(args.destination or "ckpt"))
        ).expanduser()
        robot = prompt_text("Robot type", default=args.robot or DEFAULT_ROBOT)
        project_name = prompt_text("Project name", default=args.name)
        return base_path / robot / project_name

    destination = args.destination.expanduser()
    if args.name or args.robot:
        robot = args.robot or DEFAULT_ROBOT
        if not args.name:
            raise ValueError("--name is required when --robot is used with a base destination")
        return destination / robot / args.name

    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a unitree-deploy plugin template.")
    parser.add_argument(
        "destination",
        nargs="?",
        type=Path,
        help="Legacy final directory, or base directory when --name/--robot is provided.",
    )
    parser.add_argument("--name", help="Deployment project name.")
    parser.add_argument("--robot", choices=available_robots() or None, help="Robot type.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt for path, project name, and robot.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing template files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    destination = resolve_destination(args)
    write_template(destination, force=args.force)
    print(f"plugin template written to {destination}")


if __name__ == "__main__":
    main()
