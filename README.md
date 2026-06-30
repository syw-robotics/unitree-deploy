# Unitree Deploy

[简体中文](README.zh-CN.md)

Lightweight deployment codebase for Unitree robots.

This repository wraps ONNX policies, robot observations, DDS topics, MuJoCo
models, and simple controller state machines into a reusable deployment flow.
It is useful for validating policies in simulation first, then running the same
controller path on supported Unitree hardware.

## ✨ Highlights

- Shared controller runtime for `sim` and `real` modes.
- ONNX policy loading with configurable `yaml` file.
- Multi-policy switching support.
- Template generator for custom deployment demands (such as customized obs).

## 📦 Project Layout

```text
src/unitree_deploy/
├── cli/              # console entry points
├── runtime/          # controller and sim bridge loops
├── policy/           # ONNX policy wrapper
├── obs/              # observation terms
├── visualization/    # robot state visualizer
├── utils/            # shared helpers
└── robot_model/      # bundled MuJoCo robot and terrain assets

ckpt/
├── g1/               # example G1 policies and multi-ckpt config
└── go2/              # example Go2 policies
```

## 🚀 Setup

```bash
uv sync
source .venv/bin/activate
```

For the viser viewer extras (realtime robot state visualization):

```bash
uv sync --extra viewer
```

`unitree_sdk2_python` is expected to be installed separately in the same Python
environment.

## 🕹️ Quick Start

- Run the MuJoCo-to-DDS bridge:

    ```bash
    unitree-sim-bridge --robot g1
    ```

- Run a controller against the simulator:

    ```bash
    unitree-controller --mode sim --ckpt ckpt/g1/vanilla_ppo_flat
    ```

- Run with a multi-policy manifest:

    ```bash
    unitree-controller --mode sim --multi-ckpt ckpt/g1/multi_ckpt.yaml
    ```

- Start the viser visualizer:

    ```bash
    unitree-visualizer --mode sim --robot g1
    ```

- Record a MuJoCo/DDS trajectory for rendering or analysis:

    ```bash
    unitree-trajectory-recorder \
      --robot g1 \
      --out export/runs/g1_walk \
      --record-hz 30
    ```

    Run it in a separate terminal while `unitree-sim-bridge` and the controller are
    running:

    ```bash
    unitree-sim-bridge --robot g1
    unitree-controller --mode sim --ckpt ckpt/g1/vanilla_ppo_flat
    ```

    Press `o` in the recorder process to start recording, then press `o` again to
    stop and save. Each recording is written under a timestamped subdirectory such
    as `export/runs/g1_walk/20260629-153012/`, containing `trajectory.npz`,
    `metadata.json`, and `scene.json`.

- Replay a saved trajectory in Viser:

    ```bash
    unitree-trajectory-replay export/runs/g1_walk/20260629-153012/trajectory.npz
    ```

    The replay UI starts paused and follows the robot by default. It includes
    pause, follow, playback speed, frame scrubbing, and one-step forward/back
    controls.

- For browser-based policy presentation, use the separate [`policy-web-viewer`](https://github.com/syw-robotics/policy-web-viewer) project:

    Check it out at [here](https://syw-robotics.github.io/policy-web-viewer/)

- For real hardware, pass the DDS network interface explicitly:

    ```bash
    unitree-controller --mode real --net <interface> --ckpt ckpt/g1/vanilla_ppo_flat
    ```

## 🧩 Deployment Folders

A policy deployment folder usually looks like this:

```text
ckpt/<robot>/<policy>/
├── policy.yaml             # policy, observation, joint order, and gain config
├── policy.onnx             # exported ONNX policy, or another relative path
├── custom_observations.py  # [optional] custom observation terms
└── custom_policy.py        # [optional] custom inference/action logic
```

- Generate a starter folder interactively:

    ```bash
    unitree-plugin-template
    ```

    Or script it:

    ```bash
    unitree-plugin-template ckpt --robot g1 --name my_policy
    ```

- Key `policy.yaml` fields:

```yaml
policy_path: "policy.onnx"
obs_joint_order: [...]
action_joint_order: [...]
sdk_joint_order: [...]
```

`obs_joint_order`, `action_joint_order`, and `sdk_joint_order` are matched by
joint name. Reorder indices are derived automatically, so they do not need to be
written by hand.

## 🔁 Multi-Policy Switching

Use `--multi-ckpt` when several policies should be switchable at runtime:

```yaml
default: vanilla_ppo_flat

ckpts:
  vanilla_ppo_flat: "./vanilla_ppo_flat"
  unitree_rl_lab_flat: "./unitree_rl_lab_flat"

switch:
  enabled: true
  button: B
  order: [vanilla_ppo_flat, unitree_rl_lab_flat]
  only_when: [run_policy]
  on_switch: null
```

In simulation, press `b` to switch policies. On hardware, use the remote `B`
button. Policies in one manifest must share `sdk_joint_order` and
`policy_step_dt`; observations, actions, gains, and ONNX files may differ.

## ⌨️ Default Controls

The default state machine uses these remote buttons:

- `A`: move to default joint positions.
- `Start`: run the active policy.
- `B`: switch policy when multi-ckpt switching is enabled.
- `X`: return to damping.

In simulation, the mapped keys are `enter` for `A`, `\` for `Start`, `b` for
`B`, and `x` for `X`.

## ✅ TODO

- [ ] Max torque clipping.
- [ ] G1 motion tracking policy support.
- [ ] VR teleoperation device port.
- [ ] Check viser usability for odometry and RealSense hardware deployment.
