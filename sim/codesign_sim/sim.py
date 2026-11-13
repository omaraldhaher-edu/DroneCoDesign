"""Single-environment CPU simulation wrapper (used for rendering, debugging and ROS)."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .controller import QuadController
from .drone_model import DroneDesign, build_mjcf
from .track import Figure8, Reference


class QuadSim:
    def __init__(self, design: DroneDesign | None = None, lap_time: float = 9.0,
                 n_gates: int = 6, start=(0.0, -3.0, 0.1)):
        self.design = design or DroneDesign()
        self.path = Figure8()
        self.track = self.path.track(n_gates)
        self.model = mujoco.MjModel.from_xml_string(
            build_mjcf(self.design, self.track, start_pos=start))
        self.data = mujoco.MjData(self.model)
        self.ctrl = QuadController(self.model, self.design)
        self.ref = Reference(self.path, start, lap_time=lap_time)
        self._gate_side = np.zeros(len(self.track.gates))
        self.gates_passed: list[tuple[float, int]] = []
        self.external_action = None   # (thrust, body_rates) e.g. from ROS / a policy
        mujoco.mj_forward(self.model, self.data)

    @property
    def t(self) -> float:
        return self.data.time

    def step(self, n: int = 1):
        for _ in range(n):
            if self.external_action is not None:
                thrust, rates = self.external_action
                self.data.ctrl[:] = self.ctrl.rate_controller(thrust, rates, self.data.qvel[3:6])
            else:
                self.data.ctrl[:] = self.ctrl(self.data, self.ref(self.data.time))
            mujoco.mj_step(self.model, self.data)
            self._check_gates()

    def _check_gates(self):
        p = self.data.qpos[:3]
        for i, g in enumerate(self.track.gates):
            n = np.array([math.cos(g.yaw), math.sin(g.yaw), 0.0])
            rel = p - np.asarray(g.pos)
            side = float(np.sign(rel @ n))
            lateral = rel - (rel @ n) * n
            if self._gate_side[i] < 0 < side and np.max(np.abs(lateral)) < g.inner / 2:
                self.gates_passed.append((self.data.time, i))
            self._gate_side[i] = side

    def summary(self) -> str:
        d = self.design
        return (f"mass {d.total_mass*1000:.0f} g | T/W {d.thrust_to_weight:.1f} | "
                f"hover throttle {d.total_mass*9.81/(4*d.max_thrust)*100:.0f}% | "
                f"issues: {d.check() or 'none'}")
