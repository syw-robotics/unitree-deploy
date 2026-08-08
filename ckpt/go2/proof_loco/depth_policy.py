from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from collections import deque

import numpy as np
import onnxruntime as ort

from unitree_deploy.obs.observation import ObservationContext
from unitree_deploy.policy.base_policy import BasePolicy, ObservationRegistry


class _GatePlotter:
    def __init__(self, *, history: int, update_interval: int, dt: float, backend: str | None = None) -> None:
        self.history = max(2, int(history))
        self.update_interval = max(1, int(update_interval))
        self.dt = float(dt)
        self.values: deque[float] = deque(maxlen=self.history)
        self.step = 0
        self.enabled = True
        try:
            import matplotlib

            if backend:
                matplotlib.use(str(backend), force=True)
            elif matplotlib.get_backend().lower() == "agg":
                matplotlib.use("TkAgg", force=True)
            import matplotlib.pyplot as plt

            self.plt = plt
            if plt.get_backend().lower() == "agg":
                raise RuntimeError(
                    f"matplotlib backend {plt.get_backend()!r} is non-interactive; "
                    "start controller with a GUI backend, for example MPLBACKEND=TkAgg"
                )
            plt.ion()
            self.fig, self.ax = plt.subplots(num="proof_loco gate")
            (self.line,) = self.ax.plot([], [], linewidth=1.6)
            self.ax.set_title("Final gate")
            self.ax.set_xlabel("time [s]")
            self.ax.set_ylabel("gate")
            self.ax.set_ylim(-0.05, 1.05)
            self.ax.grid(True, alpha=0.25)
            self.fig.tight_layout()
            self.fig.show()
        except Exception as exc:
            self.enabled = False
            print(f"[WARN] gate plot disabled: {exc}")

    def update(self, gate: np.ndarray) -> None:
        if not self.enabled:
            return
        self.values.append(float(np.asarray(gate).reshape(-1)[0]))
        self.step += 1
        if self.step % self.update_interval != 0:
            return

        count = len(self.values)
        x0 = max(0, self.step - count) * self.dt
        xs = x0 + np.arange(count, dtype=np.float32) * self.dt
        ys = np.asarray(self.values, dtype=np.float32)
        self.line.set_data(xs, ys)
        self.ax.set_xlim(float(xs[0]) if count > 1 else 0.0, float(xs[-1] + self.dt))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


class DepthRecurrentPolicy(BasePolicy):
    """Split depth-encoder + recurrent actor policy for AP mask-gate students."""

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
        session_providers = list(providers) if providers is not None else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        gate_control = self.config.get("gate_control", {}) or {}
        if not isinstance(gate_control, dict):
            raise TypeError("gate_control must be a mapping")
        self.gate_control_mode = str(gate_control.get("mode", "policy")).strip().lower()
        if self.gate_control_mode not in ("policy", "manual"):
            raise ValueError("gate_control.mode must be 'policy' or 'manual'")

        actor_path_key = "policy_path" if self.gate_control_mode == "policy" else "manual_path"
        actor_path_value = gate_control.get(actor_path_key)
        if actor_path_value is None:
            if self.gate_control_mode == "manual":
                raise KeyError("gate_control.manual_path is required in manual mode")
            actor_path_value = self.config["policy_path"]
        selected_actor_path = (self.policy_yaml_path.parent / actor_path_value).resolve()
        if selected_actor_path != self.model_path:
            self.session = ort.InferenceSession(
                str(selected_actor_path),
                sess_options=options,
                providers=session_providers,
            )
            self.model_path = selected_actor_path

        self.actor_session = self.session
        self.encoder_path = (self.policy_yaml_path.parent / self.config["depth_encoder_path"]).resolve()
        self.encoder_session = ort.InferenceSession(
            str(self.encoder_path),
            sess_options=options,
            providers=session_providers,
        )

        self.depth_input_name = self.config.get("depth_encoder_input_name", "depth_image")
        self.depth_output_name = self.config.get("depth_encoder_output_name", "depth_image_latent")
        self.actor_input_name = self.config.get("policy_input_name", "obs")
        self.action_output_name = self.config.get("policy_output_name", "actions")

        self.depth_height = None
        self.depth_width = None
        self.depth_history_len = None
        self.depth_vector_size = None
        for spec in self.config["observations"]:
            if spec["type"] != "depth":
                continue
            params = spec.get("params") or {}
            self.depth_height = int(params["height"])
            self.depth_width = int(params["width"])
            self.depth_history_len = int(spec["history_len"])
            self.depth_vector_size = self.depth_height * self.depth_width * self.depth_history_len
            break
        if self.depth_vector_size is None:
            raise ValueError("DepthRecurrentPolicy requires a depth observation")

        actor_inputs = {item.name: item for item in self.actor_session.get_inputs()}
        self.h_state = self._zeros_for_input(actor_inputs, "h_in")
        self.c_state = self._zeros_for_input(actor_inputs, "c_in") if "c_in" in actor_inputs else None
        self.gate_state = None
        self.gate_valid = None
        self.gate_open_state = None
        self.manual_gate = None
        if self.gate_control_mode == "policy":
            self.gate_state = self._zeros_for_input(actor_inputs, "gate_in")
            self.gate_valid = self._zeros_for_input(actor_inputs, "gate_valid_in")
            if "gate_open_in" in actor_inputs:
                self.gate_open_state = self._zeros_for_input(actor_inputs, "gate_open_in")
            self.last_gate = np.zeros_like(self.gate_state)
        else:
            self.manual_gate = self._constant_for_input(
                actor_inputs,
                "gate",
                gate_control.get("value", 1.0),
            )
            self.last_gate = self.manual_gate.copy()

        gate_plot_cfg = self.config.get("gate_plot", {}) or {}
        self.gate_plotter = None
        if bool(gate_plot_cfg.get("enabled", False)):
            self.gate_plotter = _GatePlotter(
                history=int(gate_plot_cfg.get("history", 500)),
                update_interval=int(gate_plot_cfg.get("update_interval", 5)),
                dt=self.policy_step_dt,
                backend=gate_plot_cfg.get("backend"),
            )

    @staticmethod
    def _zeros_for_input(actor_inputs: dict[str, ort.NodeArg], name: str) -> np.ndarray:
        if name not in actor_inputs:
            raise ValueError(f"actor ONNX is missing required input {name!r}")
        node_arg = actor_inputs[name]
        shape = [1 if not isinstance(dim, int) else dim for dim in node_arg.shape]
        dtype_by_onnx_type = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(float16)": np.float16,
            "tensor(bool)": np.bool_,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }
        try:
            dtype = dtype_by_onnx_type[node_arg.type]
        except KeyError as exc:
            raise TypeError(f"unsupported ONNX input type for {name!r}: {node_arg.type}") from exc
        return np.zeros(shape, dtype=dtype)

    @classmethod
    def _constant_for_input(cls, actor_inputs: dict[str, ort.NodeArg], name: str, value) -> np.ndarray:
        target = cls._zeros_for_input(actor_inputs, name)
        configured = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(configured)):
            raise ValueError(f"gate_control.value must be finite, got {value!r}")
        if np.any(configured < 0.0) or np.any(configured > 1.0):
            raise ValueError(f"gate_control.value must be within [0, 1], got {value!r}")
        try:
            target[:] = configured
        except ValueError as exc:
            raise ValueError(
                f"gate_control.value with shape {configured.shape} cannot fill ONNX input {name!r} "
                f"with shape {target.shape}"
            ) from exc
        return target

    def reset(self) -> None:
        super().reset()
        self.h_state.fill(0.0)
        if self.c_state is not None:
            self.c_state.fill(0.0)
        if self.gate_state is not None:
            self.gate_state.fill(0.0)
        if self.gate_valid is not None:
            self.gate_valid.fill(0.0)
        if self.gate_open_state is not None:
            self.gate_open_state.fill(False)
        if self.manual_gate is not None:
            self.last_gate[:] = self.manual_gate

    def compute_target_q(self, context: ObservationContext) -> np.ndarray:
        if self._observation_needs_prime:
            self.observation.prime(context)
            self._observation_needs_prime = False
        else:
            self.observation.update(context)

        obs_vector = self.observation.compute().astype(np.float32, copy=False)
        prop_obs = obs_vector[:-self.depth_vector_size]
        depth_obs = obs_vector[-self.depth_vector_size:].reshape(
            1,
            self.depth_history_len,
            self.depth_height,
            self.depth_width,
        )
        depth_latent = self.encoder_session.run(
            [self.depth_output_name],
            {self.depth_input_name: depth_obs},
        )[0].astype(np.float32, copy=False)

        actor_obs = np.concatenate((prop_obs[None, :], depth_latent), axis=1).astype(np.float32, copy=False)
        actor_inputs = {
            self.actor_input_name: actor_obs,
            "h_in": self.h_state,
        }
        output_names = [self.action_output_name, "h_out"]
        if self.c_state is not None:
            actor_inputs["c_in"] = self.c_state
            output_names.append("c_out")
        if self.gate_control_mode == "policy":
            actor_inputs["gate_in"] = self.gate_state
            actor_inputs["gate_valid_in"] = self.gate_valid
            output_names.extend(["gate_out", "gate_valid_out"])
            if self.gate_open_state is not None:
                actor_inputs["gate_open_in"] = self.gate_open_state
                output_names.append("gate_open_out")
        else:
            actor_inputs["gate"] = self.manual_gate
            output_names.append("gate_out")

        outputs = self.actor_session.run(output_names, actor_inputs)
        output_index = 0
        policy_action = np.asarray(outputs[output_index], dtype=np.float32).reshape(-1)
        output_index += 1
        self.h_state[:] = outputs[output_index]
        output_index += 1
        if self.c_state is not None:
            self.c_state[:] = outputs[output_index]
            output_index += 1
        if self.gate_control_mode == "policy":
            self.gate_state[:] = outputs[output_index]
            self.gate_valid[:] = outputs[output_index + 1]
            if self.gate_open_state is not None:
                self.gate_open_state[:] = outputs[output_index + 2]
                self.last_gate[:] = self.gate_open_state
            else:
                self.last_gate[:] = self.gate_state
        else:
            self.last_gate[:] = outputs[output_index]
        if self.gate_plotter is not None:
            self.gate_plotter.update(self.last_gate)

        return self._target_q_from_policy_action(policy_action, context)
