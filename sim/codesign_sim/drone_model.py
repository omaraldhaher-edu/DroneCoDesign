"""Parametric quadrotor model: design vector -> physical parameters -> MuJoCo MJCF.

This is the seed of Phase 1 ("parametric drone model"). The design is a small
dataclass; `build_mjcf` turns it (plus a track) into a MuJoCo XML string. Mass and
inertia are computed by MuJoCo from the geoms, so changing arm length or component
masses changes the dynamics consistently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DroneDesign:
    """Design parameters of an X-configuration quad (defaults ~ a 5" racer)."""

    arm_length: float = 0.11          # m, centre to motor axis
    core_mass: float = 0.38           # kg, frame plate + battery + FC/ESC/camera
    motor_mass: float = 0.032         # kg, per motor
    arm_mass: float = 0.012           # kg, per arm
    prop_radius: float = 0.0635       # m (5 inch prop)
    max_thrust: float = 9.0           # N per motor
    torque_coeff: float = 0.016       # m, yaw torque / thrust ratio
    motor_time_const: float = 0.03    # s, first-order thrust lag
    core_size: tuple = (0.07, 0.035, 0.025)  # half extents of the core box, m

    @property
    def total_mass(self) -> float:
        return self.core_mass + 4 * (self.motor_mass + self.arm_mass)

    @property
    def thrust_to_weight(self) -> float:
        return 4 * self.max_thrust / (self.total_mass * 9.81)

    def motor_positions(self) -> np.ndarray:
        """Motor xy positions, ordered FR, BL, FL, BR (Betaflight-like X)."""
        a = self.arm_length / math.sqrt(2)
        return np.array([[a, -a], [-a, a], [a, a], [-a, -a]])

    @property
    def spin_dirs(self) -> np.ndarray:
        """+1 = CCW prop (reaction torque -z), -1 = CW. Diagonal pairs share direction."""
        return np.array([1.0, 1.0, -1.0, -1.0])

    def check(self) -> list[str]:
        """Minimal feasibility checks; grows into the constraint checker later."""
        issues = []
        gap = self.arm_length * math.sqrt(2) - 2 * self.prop_radius  # adjacent motors
        if gap < 0.005:
            issues.append(f"props overlap (adjacent clearance {gap*1000:.1f} mm)")
        if self.thrust_to_weight < 2.0:
            issues.append(f"thrust-to-weight {self.thrust_to_weight:.2f} < 2")
        return issues


@dataclass
class Gate:
    pos: tuple            # gate centre, m
    yaw: float            # rad, direction the drone should fly through
    inner: float = 1.5    # opening width/height, m
    bar: float = 0.08     # frame bar thickness, m


@dataclass
class Track:
    gates: list = field(default_factory=list)


def _f(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def _gate_xml(i: int, g: Gate) -> str:
    h = g.inner / 2 + g.bar / 2
    t = g.bar / 2
    # Gate body local frame: x = flight direction, opening spans y/z.
    bars = [
        (f"0 {h:.4f} 0", f"{t} {t} {h + t}"),
        (f"0 {-h:.4f} 0", f"{t} {t} {h + t}"),
        (f"0 0 {h:.4f}", f"{t} {h + t} {t}"),
        (f"0 0 {-h:.4f}", f"{t} {h + t} {t}"),
    ]
    geoms = "\n".join(
        f'      <geom class="gate" pos="{p}" size="{s}"/>' for p, s in bars
    )
    post_h = g.pos[2] - g.inner / 2 - g.bar
    post = (
        f'      <geom class="gate_post" pos="0 0 {-(g.inner / 2 + g.bar + post_h / 2):.4f}" '
        f'size="0.03 {post_h / 2:.4f}"/>'
        if post_h > 0.02 else ""
    )
    return (
        f'    <body name="gate{i}" pos="{_f(g.pos)}" euler="0 0 {g.yaw:.6g}">\n'
        f"{geoms}\n{post}\n    </body>"
    )


def build_mjcf(design: DroneDesign, track: Track | None = None,
               start_pos=(0.0, 0.0, 0.1), timestep: float = 0.002) -> str:
    """Return a MuJoCo XML string with the drone, an optional track and a scene."""
    d = design
    cx, cy, cz = d.core_size
    arms, motors, props, sites, acts = [], [], [], [], []
    for i, ((x, y), s) in enumerate(zip(d.motor_positions(), d.spin_dirs)):
        arms.append(
            f'      <geom class="arm" fromto="0 0 0 {x:.5f} {y:.5f} 0" mass="{d.arm_mass}"/>'
        )
        motors.append(
            f'      <geom class="motor" pos="{x:.5f} {y:.5f} 0.012" mass="{d.motor_mass}"/>'
        )
        color = "prop_ccw" if s > 0 else "prop_cw"
        props.append(
            f'      <geom class="{color}" pos="{x:.5f} {y:.5f} 0.03" size="{d.prop_radius:.4f} 0.002"/>'
        )
        sites.append(f'      <site name="m{i}" pos="{x:.5f} {y:.5f} 0.03"/>')
        # 6D gear on a site: force along site z, plus reaction yaw torque.
        acts.append(
            f'    <general name="motor{i}" site="m{i}" gear="0 0 1 0 0 {-s * d.torque_coeff:.5f}" '
            f'ctrlrange="0 {d.max_thrust}" dyntype="filter" dynprm="{d.motor_time_const}"/>'
        )

    gates = "\n".join(_gate_xml(i, g) for i, g in enumerate(track.gates)) if track else ""

    return f"""
<mujoco model="codesign_quad">
  <option timestep="{timestep}" integrator="implicitfast" density="1.225" viscosity="1.8e-5"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.62 0.85" rgb2="0.08 0.1 0.18" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.26 0.22" rgb2="0.17 0.2 0.17" width="512" height="512" mark="edge" markrgb="0.3 0.35 0.3"/>
    <material name="grid" texture="grid" texrepeat="40 40" reflectance="0.05"/>
  </asset>
  <default>
    <default class="arm"><geom type="capsule" size="0.008" rgba="0.12 0.12 0.12 1"/></default>
    <default class="motor"><geom type="cylinder" size="0.014 0.01" rgba="0.75 0.75 0.78 1"/></default>
    <default class="prop_ccw"><geom type="cylinder" contype="0" conaffinity="0" mass="0" rgba="0.1 0.6 1 0.35"/></default>
    <default class="prop_cw"><geom type="cylinder" contype="0" conaffinity="0" mass="0" rgba="1 0.35 0.1 0.35"/></default>
    <default class="gate"><geom type="box" contype="0" conaffinity="0" rgba="1 0.45 0.05 1"/></default>
    <default class="gate_post"><geom type="cylinder" contype="0" conaffinity="0" rgba="0.3 0.3 0.3 1"/></default>
  </default>
  <worldbody>
    <light pos="0 0 30" dir="0 0 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="60 60 0.1" material="grid"/>
    <camera name="overview" pos="0 -17 12" xyaxes="1 0 0 0 0.6 0.8"/>
    <body name="quad" pos="{_f(start_pos)}">
      <freejoint name="root"/>
      <camera name="chase" mode="trackcom" pos="-2.2 -2.2 1.2" xyaxes="0.707 -0.707 0 0.25 0.25 0.935"/>
      <camera name="fpv" pos="{cx + 0.01} 0 0.01" xyaxes="0 -1 0 -0.26 0 0.966" fovy="110"/>
      <geom name="core" type="box" size="{cx} {cy} {cz}" mass="{d.core_mass}" rgba="0.15 0.15 0.18 1"/>
      <geom name="canopy" type="box" pos="{cx * 0.4:.4f} 0 {cz + 0.01:.4f}" size="{cx * 0.4:.4f} {cy * 0.7:.4f} 0.01" mass="0" rgba="0.9 0.1 0.2 1"/>
{chr(10).join(arms)}
{chr(10).join(motors)}
{chr(10).join(props)}
{chr(10).join(sites)}
      <site name="imu"/>
    </body>
{gates}
  </worldbody>
  <actuator>
{chr(10).join(acts)}
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <framepos name="pos" objtype="site" objname="imu"/>
    <framequat name="quat" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
"""
