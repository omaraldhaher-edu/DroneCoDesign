"""GPU-parallel feasibility check: N quads, each with its OWN design, stepped with MJX.

Each environment gets a different mass scale and arm length (motor site positions), and
a simple JAX hover controller. Reports throughput and shows that per-env designs give
different dynamics (hover altitude error under a nominal-mass thrust command).

    python mjx_parallel_check.py --n 4096 --steps 500
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from DroneCoDesign.sim.codesign_sim.drone_model import DroneDesign, build_mjcf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()

    design = DroneDesign()
    m = mujoco.MjModel.from_xml_string(build_mjcf(design, None, start_pos=(0, 0, 1.0)))
    # MJX does not support MuJoCo's built-in fluid model with implicitfast; drag will be
    # an explicit term in our own dynamics (Phase 1), so disable it here.
    m.opt.density = 0.0
    m.opt.viscosity = 0.0
    mx = mjx.put_model(m)
    quad = m.body("quad").id
    nominal_mass = float(m.body_subtreemass[quad])

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    mass_scale = jax.random.uniform(k1, (a.n,), minval=0.8, maxval=1.25)
    arm_scale = jax.random.uniform(k2, (a.n,), minval=0.8, maxval=1.3)

    # Per-environment model: scale masses/inertias of the quad body, move motor sites.
    def make_model(ms, ar):
        body_mass = mx.body_mass.at[quad].multiply(ms)
        body_inertia = mx.body_inertia.at[quad].multiply(ms)
        site_pos = mx.site_pos.at[:4, :2].multiply(ar)
        gear = mx.actuator_gear  # yaw torque coefficient unchanged
        return mx.replace(body_mass=body_mass, body_inertia=body_inertia,
                          site_pos=site_pos, actuator_gear=gear)

    d0 = mjx.make_data(mx)
    d0 = d0.replace(qpos=d0.qpos.at[2].set(1.0))
    batched_data = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (a.n,) + x.shape) if hasattr(x, "shape") else x, d0)

    hover = nominal_mass * 9.81 / 4

    def ctrl_and_step(ms, ar, data):
        model = make_model(ms, ar)      # this env's design
        # Altitude-only PD around the NOMINAL hover thrust: heavier designs sag, lighter rise.
        z, vz = data.qpos[2], data.qvel[2]
        u = hover + 0.4 * (1.0 - z) - 0.3 * vz
        data = data.replace(ctrl=jnp.full(4, u))
        return mjx.step(model, data)

    step = jax.vmap(ctrl_and_step)

    def rollout(d):
        def body(d, _):
            return step(mass_scale, arm_scale, d), None
        d, _ = jax.lax.scan(body, d, None, length=a.steps)
        return d

    rollout = jax.jit(rollout)
    t = time.perf_counter()
    out = rollout(batched_data)
    out.qpos.block_until_ready()
    compile_s = time.perf_counter() - t

    t = time.perf_counter()
    out = rollout(batched_data)
    out.qpos.block_until_ready()
    run_s = time.perf_counter() - t

    sps = a.n * a.steps / run_s
    z = out.qpos[:, 2]
    print(f"device: {jax.devices()[0]}  envs: {a.n}  steps: {a.steps} (dt={m.opt.timestep}s)")
    print(f"compile+first run {compile_s:.1f}s | run {run_s:.2f}s | {sps/1e6:.2f} M env-steps/s "
          f"| {sps*m.opt.timestep:,.0f}x real time aggregate")
    corr = jnp.corrcoef(mass_scale, z)[0, 1]
    print(f"final z: min {z.min():.2f} max {z.max():.2f}  corr(mass_scale, z) = {corr:.2f} "
          f"(negative => per-env designs really differ)")


if __name__ == "__main__":
    main()
