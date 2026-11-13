
## What it can do right now

- **Build a drone from design parameters.** `DroneDesign` (arm length, component masses, prop size, max thrust per motor, yaw torque coefficient, motor time constant) becomes a MuJoCo model. Mass and inertia are computed from the geometry.
- **Fly a race track.** A six-gate figure-eight track (18 m × 9 m, 1.5 m gates). A cascaded controller follows a smooth reference through every gate, and gate passes are detected and logged. The default 5" design (556 g, thrust-to-weight 6.6) completes laps at a 9 s lap time.
- **Render it.** Headless MP4 with three cameras (overview, chase, onboard FPV) plus a PNG snapshot, or an interactive 3D viewer.
- **Run thousands of drones on the GPU, each with its own design.** MJX steps a batch of quads where every environment has a different mass and arm length. Measured: 4,096 environments at about 0.8 M physics steps/s on an RTX 3070 Ti Laptop GPU, about 1,600× real time.
- **Talk ROS 2.** A ROS 2 Jazzy node publishes odometry, IMU, TF, the flown path, and gate markers for RViz. It accepts `(collective thrust, body rates)` commands, the same action interface the RL policy will use. 

## Why MuJoCo + MJX

The plan listed Aerial Gym, Isaac Lab, Flightmare, and a custom JAX/MuJoCo-MJX sim. We chose **MuJoCo 4.0 + MJX** because:

- Each parallel environment can have its own design parameters, which the co-design loop needs.
- It runs on an 8 GB laptop GPU and installs with pip.
- JAX makes differentiable simulation possible later (plan, Phase 5).

The other options were set aside for these reasons:

| Option | Why not (for now) |
|---|---|
| Isaac Lab | Recommends 16 GB or more of VRAM; WSL2 is not officially supported |
| Aerial Gym | Built on Isaac Gym Preview, which is deprecated |
| Flightmare | ROS 1 + Unity, no longer maintained |
| Gazebo + PX4 | Not GPU-parallel, so too slow for RL, but a good candidate for sim-to-real software-in-the-loop later |

## Setup

Tested on **Ubuntu 24.04 (also under WSL2 on Windows 11) with Python 3.12**. Other Linux versions should work; macOS and native Windows should run the CPU parts but are untested.

```bash
git clone <REPO_URL>
cd <REPO_NAME>

python3 -m venv .venv          # Ubuntu: needs `sudo apt install python3-venv` first
source .venv/bin/activate
pip install -r sim/requirements.txt

# Only if you have an NVIDIA GPU (driver installed; under WSL2 the Windows driver is enough):
pip install "jax[cuda12]==0.12.0.1"
```

No sudo? Use [uv](https://docs.astral.sh/uv/) instead of `python3 -m venv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r sim/requirements.txt
```

Check it works:

```bash
python -c "import mujoco, jax; print(mujoco.__version__, jax.devices())"
```

## Running

All commands are run from `sim/` with the venv active.

```bash
cd sim

# Headless video: writes videos/demo.mp4 and videos/demo.png
python render_demo.py                 # 16 s of flight; about 4-5 min on CPU rendering
python render_demo.py --seconds 6     # quick check (takeoff + speed-up take 5 s; first gate at ~4.3 s)

# Interactive 3D viewer (needs a display; WSLg works on Windows 11)
python render_demo.py --viewer

# Try another design
python render_demo.py --arm 0.15 --core-mass 0.6 --max-thrust 12 --out videos/heavy.mp4

# GPU batch: N drones, each with a different mass and arm length
python mjx_parallel_check.py --n 4096 --steps 500
```

`render_demo.py` prints the design summary, the gates passed, and the final position. A healthy run of the default design prints `gates passed (8): [0, 1, 2, 3, 4, 5, 0, 1]`.

### ROS 2 (optional)

Install ROS 2 Jazzy once. It needs sudo and takes 10–20 min:

```bash
bash sim/ros2/install_ros2_jazzy.sh
```

Then use two terminals, both with `source /opt/ros/jazzy/setup.bash` and the venv active, from `sim/`:

```bash
python ros2/quad_bridge.py --ros-args -p fpv_hz:=10.0     # terminal 1
rviz2 -d ros2/quad.rviz                                   # terminal 2

# Take over control (thrust 6 N, yaw rate 1 rad/s). Stop publishing and the autopilot resumes.
ros2 topic pub -r 50 /quad/cmd geometry_msgs/msg/TwistStamped "{twist: {linear: {z: 6.0}, angular: {z: 1.0}}}"
```

| Topic | Dir | Content |
|---|---|---|
| `/quad/odom` | out | `nav_msgs/Odometry`: world-frame pose, body-frame twist |
| `/quad/imu` | out | `sensor_msgs/Imu`: gyro + accelerometer |
| `/tf` | out | `world → base_link` |
| `/quad/path` | out | `nav_msgs/Path`: flown trail |
| `/quad/model`, `/track/gates` | out | `MarkerArray` for RViz |
| `/track/gates_passed` | out | `std_msgs/Int32`: index of each gate passed |
| `/quad/fpv/image_raw` | out | `sensor_msgs/Image`: only when `fpv_hz > 0` |
| `/quad/cmd` | in | `geometry_msgs/TwistStamped`: `linear.z` = collective thrust [N], `angular` = body rates [rad/s] |

## How the code fits together

The pipeline is design parameters → MuJoCo XML → simulation loop:

```
DroneDesign ──build_mjcf()──▶ MuJoCo model ──▶ QuadSim.step():  controller → 4 motor thrusts → mj_step (2 ms)
```

| File | What it does |
|---|---|
| `sim/codesign_sim/drone_model.py` | `DroneDesign` dataclass and `build_mjcf()`, which generates the scene (quad, gates, cameras). Motors are actuators with thrust + yaw reaction torque and a first-order lag; `ctrl` is per-motor thrust in N. Motor order: FR, BL, FL, BR. |
| `sim/codesign_sim/track.py` | Figure-eight path, gate placement, and a smooth time reference (takeoff, then speed ramp). Placeholder for a time-optimal reference. |
| `sim/codesign_sim/controller.py` | Cascaded controller. `position_controller` produces (thrust, body rates); `rate_controller` turns those into motor thrusts via the mixer. **The RL policy is meant to replace `position_controller`.** |
| `sim/codesign_sim/sim.py` | `QuadSim`: CPU sim loop, gate-pass detection, and an `external_action` hook (used by ROS, and later by the policy). |
| `sim/render_demo.py` | Video, snapshot, and interactive viewer. |
| `sim/mjx_parallel_check.py` | The GPU pattern for RL: per-environment design parameters, `vmap` + `lax.scan`. |
| `sim/ros2/` | ROS 2 bridge node, RViz config, install script. |

## Status and known limits

- **ROS 2 bridge not yet run end to end.** It compiles, but ROS wasn't tested yet.
- **Physics is simplified.** There is no battery model, no thrust-vs-RPM/voltage curves, and only MuJoCo's basic body drag (disabled in MJX, which doesn't support it with the `implicitfast` integrator). 
- **The controller is a hand-written stand-in**, not a racing policy.

### Setup and rendering issues

- **Rendering in WSL2 falls back to CPU (llvmpipe)**, about 40 ms per 720p frame. Training doesn't render, so this only affects videos.
- **JAX can run out of GPU memory** when another program is using the GPU. Run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`; `mjx_parallel_check.py` sets this for you.
- **On Linux, headless rendering uses `MUJOCO_GL=egl`.** It must be set before `import mujoco`; `render_demo.py` handles this.

## Next steps

1. **Phase 1:** a realistic motor/prop/battery/drag model and a parts database, plus sanity checks (hover throttle, thrust-to-weight, top speed, flight time).
2. **Phase 2:** PPO in MJX on the fixed design and track, with the action space `(collective thrust, body rates)`, benchmarked against a time-optimal reference.
