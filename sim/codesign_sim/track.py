"""Demo track: a figure-eight with gates placed along a smooth reference path.

The reference path is a placeholder for the time-optimal reference of Phase 2; here
it is just an analytic lemniscate so the demo has something to follow.
"""
from __future__ import annotations

import math

import numpy as np

from .drone_model import Gate, Track


class Figure8:
    """Lemniscate of Gerono with altitude variation, parameterised by phase phi."""

    def __init__(self, ax=9.0, ay=4.5, h=2.5, dz=0.8):
        self.ax, self.ay, self.h, self.dz = ax, ay, h, dz

    def p(self, phi):
        s, c = math.sin(phi), math.cos(phi)
        return np.array([self.ax * s, self.ay * 2 * s * c, self.h + self.dz * c])

    def dp(self, phi):
        s, c = math.sin(phi), math.cos(phi)
        return np.array([self.ax * c, self.ay * 2 * math.cos(2 * phi), -self.dz * s])

    def ddp(self, phi):
        s = math.sin(phi)
        return np.array([-self.ax * s, -self.ay * 4 * math.sin(2 * phi), -self.dz * math.cos(phi)])

    def track(self, n_gates=6) -> Track:
        gates = []
        for k in range(n_gates):
            phi = (k + 0.5) * 2 * math.pi / n_gates   # avoids the crossing at phi=0, pi
            t = self.dp(phi)
            gates.append(Gate(pos=tuple(self.p(phi)), yaw=math.atan2(t[1], t[0])))
        return Track(gates=gates)


def _smoothstep(u):
    """Quintic smoothstep and its first two derivatives w.r.t. u."""
    u = min(max(u, 0.0), 1.0)
    return (10 * u**3 - 15 * u**4 + 6 * u**5,
            30 * u**2 - 60 * u**3 + 30 * u**4,
            60 * u - 180 * u**2 + 120 * u**3)


class Reference:
    """Takeoff to the path start, then fly the figure-eight with a ramped-up speed."""

    def __init__(self, path: Figure8, start, lap_time=9.0, takeoff_time=2.0, ramp_time=3.0):
        self.path = path
        self.start = np.asarray(start, float)
        self.w = 2 * math.pi / lap_time
        self.t0, self.tr = takeoff_time, ramp_time

    def phase(self, t):
        """phi, phi_dot, phi_ddot after takeoff (closed form for the quintic speed ramp)."""
        tau = t - self.t0
        if tau <= 0:
            return 0.0, 0.0, 0.0
        if tau < self.tr:
            u = tau / self.tr
            s, ds, _ = _smoothstep(u)
            # integral of smoothstep(u) du = 2.5u^4 - 3u^5 + u^6
            integ = 2.5 * u**4 - 3 * u**5 + u**6
            return self.w * self.tr * integ, self.w * s, self.w * ds / self.tr
        integ_full = 0.5  # integral of smoothstep over [0,1]
        return self.w * (self.tr * integ_full + (tau - self.tr)), self.w, 0.0

    def __call__(self, t):
        """Return (pos, vel, acc) of the reference at time t."""
        if t < self.t0:
            goal = self.path.p(0.0)
            s, ds, dds = _smoothstep(t / self.t0)
            delta = goal - self.start
            return (self.start + s * delta, ds / self.t0 * delta, dds / self.t0**2 * delta)
        phi, w, dw = self.phase(t)
        d1 = self.path.dp(phi)
        return self.path.p(phi), d1 * w, self.path.ddp(phi) * w * w + d1 * dw
