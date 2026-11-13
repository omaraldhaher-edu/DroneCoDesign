"""Render the parametric racing quad flying a figure-eight gate track.

    python render_demo.py                 # headless: writes videos/demo.mp4 + a PNG
    python render_demo.py --viewer        # interactive MuJoCo viewer (needs a display / WSLg)
    python render_demo.py --arm 0.15 --max-thrust 12   # try another design
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

if "--viewer" not in sys.argv and sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")   # headless rendering on Linux / WSL
import mujoco
import numpy as np

from DroneCoDesign.sim.codesign_sim.drone_model import DroneDesign
from DroneCoDesign.sim.codesign_sim.sim import QuadSim

TRAIL_RGBA = np.array([0.1, 0.9, 1.0, 0.9], dtype=np.float32)
REF_RGBA = np.array([1.0, 1.0, 1.0, 0.25], dtype=np.float32)


def add_polyline(scn: mujoco.MjvScene, pts, rgba, width=0.03):
    """Append line segments to a scene as user geoms (works for Renderer and viewer)."""
    for a, b in zip(pts[:-1], pts[1:]):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scn.ngeom += 1


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--lap-time", type=float, default=9.0)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--out", default="videos/demo.mp4")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--arm", type=float, default=None, help="arm length [m]")
    ap.add_argument("--max-thrust", type=float, default=None, help="per-motor thrust [N]")
    ap.add_argument("--core-mass", type=float, default=None, help="core mass [kg]")
    return ap.parse_args()


def make_design(a) -> DroneDesign:
    d = DroneDesign()
    if a.arm is not None:
        d.arm_length = a.arm
    if a.max_thrust is not None:
        d.max_thrust = a.max_thrust
    if a.core_mass is not None:
        d.core_mass = a.core_mass
    return d


def run_viewer(sim: QuadSim, seconds: float):
    import mujoco.viewer
    trail = []
    with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        v.cam.fixedcamid = sim.model.camera("chase").id
        t0 = time.perf_counter()
        while v.is_running() and sim.t < seconds:
            sim.step(10)
            trail.append(sim.data.qpos[:3].copy())
            with v.lock():
                v.user_scn.ngeom = 0
                add_polyline(v.user_scn, trail[-300:], TRAIL_RGBA)
            v.sync()
            lag = sim.t - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)


def run_headless(sim: QuadSim, seconds: float, out: str, fps: int):
    import imageio.v2 as imageio

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = 1280, 720
    big = mujoco.Renderer(sim.model, H, W)
    small = mujoco.Renderer(sim.model, H // 2, W // 2)
    opt = mujoco.MjvOption()
    steps_per_frame = max(1, round(1.0 / (fps * sim.model.opt.timestep)))
    ref_pts = [sim.path.p(phi) for phi in np.linspace(0, 2 * np.pi, 200)]

    trail, frames = [], []
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8)
    snapshot = None
    try:
        while sim.t < seconds:
            sim.step(steps_per_frame)
            trail.append(sim.data.qpos[:3].copy())

            big.update_scene(sim.data, camera="overview", scene_option=opt)
            add_polyline(big.scene, ref_pts, REF_RGBA, 0.015)
            add_polyline(big.scene, trail[-250:], TRAIL_RGBA)
            top = big.render()

            small.update_scene(sim.data, camera="chase", scene_option=opt)
            add_polyline(small.scene, trail[-250:], TRAIL_RGBA, 0.02)
            chase = small.render()
            small.update_scene(sim.data, camera="fpv", scene_option=opt)
            fpv = small.render()

            frame = np.vstack([top, np.hstack([chase, fpv])])
            writer.append_data(frame)
            if snapshot is None and sim.t > 0.55 * seconds:
                snapshot = frame
    finally:
        writer.close()
        big.close()
        small.close()
    png = out.with_suffix(".png")
    imageio.imwrite(png, snapshot if snapshot is not None else frame)
    return out, png


def main():
    a = parse()
    sim = QuadSim(make_design(a), lap_time=a.lap_time)
    print("design:", sim.summary())
    t0 = time.perf_counter()
    if a.viewer:
        run_viewer(sim, a.seconds)
    else:
        vid, png = run_headless(sim, a.seconds, a.out, a.fps)
        print(f"wrote {vid} and {png} in {time.perf_counter() - t0:.1f}s")
    gates = [i for _, i in sim.gates_passed]
    print(f"gates passed ({len(gates)}): {gates}")
    print(f"final position {np.round(sim.data.qpos[:3], 2)}")


if __name__ == "__main__":
    main()
