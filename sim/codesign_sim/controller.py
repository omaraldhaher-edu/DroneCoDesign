"""Cascaded geometric controller (baseline / stand-in for the RL policy).

Outer loop:  position/velocity error -> collective thrust + desired attitude.
Inner loop:  attitude error -> body-rate command -> body torque (rate controller).
Mixer:       [thrust, torque] -> 4 motor thrusts.

`rate_controller` takes (collective thrust, body rates), the same action interface the
RL policy will use in Phase 2, so the policy can later replace `position_controller`.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .drone_model import DroneDesign

G = 9.81


def _vee(m):
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


class QuadController:
    def __init__(self, model: mujoco.MjModel, design: DroneDesign,
                 kp=6.0, kd=4.0, k_att=9.0, k_rate=np.array([28.0, 28.0, 12.0])):
        bid = model.body("quad").id
        self.mass = float(model.body_subtreemass[bid])
        # Full inertia matrix in the body frame.
        q = model.body_iquat[bid]
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, q)
        r = r.reshape(3, 3)
        self.J = r @ np.diag(model.body_inertia[bid]) @ r.T
        self.kp, self.kd, self.k_att, self.k_rate = kp, kd, k_att, k_rate
        self.fmax = design.max_thrust

        xy = design.motor_positions()
        s = design.spin_dirs
        mix = np.vstack([np.ones(4), xy[:, 1], -xy[:, 0], -s * design.torque_coeff])
        self.unmix = np.linalg.inv(mix)

    # ---- outer loop -------------------------------------------------------
    def position_controller(self, pos, vel, R, ref):
        p_r, v_r, a_r = ref
        a_cmd = a_r + self.kp * (p_r - pos) + self.kd * (v_r - vel) + np.array([0, 0, G])
        # keep the command within a cone of 60 deg from vertical, and bounded
        a_cmd[2] = max(a_cmd[2], 0.3 * G)
        z_d = a_cmd / np.linalg.norm(a_cmd)

        # Racing-style heading: face along the reference velocity.
        yaw = math.atan2(v_r[1], v_r[0]) if np.linalg.norm(v_r[:2]) > 0.5 else self._last_yaw
        self._last_yaw = yaw
        x_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        y_d = np.cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        R_d = np.column_stack([np.cross(y_d, z_d), y_d, z_d])

        thrust = self.mass * float(a_cmd @ R[:, 2])
        e_R = 0.5 * _vee(R_d.T @ R - R.T @ R_d)
        rates = -self.k_att * e_R                     # body-rate command
        return thrust, rates

    _last_yaw = 0.0

    # ---- inner loop -------------------------------------------------------
    def rate_controller(self, thrust, rates_cmd, omega):
        """(collective thrust [N], body rates [rad/s]) -> 4 motor thrusts [N]."""
        torque = self.J @ (self.k_rate * (rates_cmd - omega)) + np.cross(omega, self.J @ omega)
        f = self.unmix @ np.concatenate([[thrust], torque])
        return np.clip(f, 0.0, self.fmax)

    def __call__(self, data: mujoco.MjData, ref):
        pos = data.qpos[0:3]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, data.qpos[3:7])
        R = R.reshape(3, 3)
        vel = data.qvel[0:3]
        omega = data.qvel[3:6]          # free-joint angular velocity is in the body frame
        thrust, rates = self.position_controller(pos, vel, R, ref)
        return self.rate_controller(thrust, rates, omega)
