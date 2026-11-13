"""ROS 2 bridge for the MuJoCo racing-quad sim.

Runs the sim in (approximately) real time and exposes it over ROS 2:

  publishes   /tf                    world -> base_link
              /quad/odom             nav_msgs/Odometry       (world frame pose, body twist)
              /quad/imu              sensor_msgs/Imu         (gyro + accelerometer)
              /quad/path             nav_msgs/Path           (flown trail)
              /quad/model            visualization_msgs/MarkerArray (drone body, frame-locked)
              /track/gates           visualization_msgs/MarkerArray (latched)
              /track/gates_passed    std_msgs/Int32          (index of each gate passed)
              /quad/fpv/image_raw    sensor_msgs/Image       (optional, param fpv_hz > 0)
  subscribes  /quad/cmd              geometry_msgs/TwistStamped
                                       linear.z = collective thrust [N]
                                       angular  = body-rate command [rad/s]
              Same action interface the RL policy will use. If no command arrives for
              0.25 s the built-in tracking controller ("autopilot") flies the figure-eight.

Run (WSL, with ROS 2 Jazzy sourced and the venv active):
    python ros2/quad_bridge.py
    rviz2 -d ros2/quad.rviz
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Int32
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from DroneCoDesign.sim.codesign_sim.sim import QuadSim


def _quat_from_yaw(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))   # w, x, y, z


class QuadBridge(Node):
    def __init__(self):
        super().__init__("quad_sim")
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("fpv_hz", 0.0)
        self.declare_parameter("lap_time", 9.0)
        self.sim = QuadSim(lap_time=self.get_parameter("lap_time").value)
        self.get_logger().info("design: " + self.sim.summary())

        self.rate = self.get_parameter("rate_hz").value
        self.substeps = max(1, round(1.0 / (self.rate * self.sim.model.opt.timestep)))
        self.tf = TransformBroadcaster(self)
        self.pub_odom = self.create_publisher(Odometry, "/quad/odom", 10)
        self.pub_imu = self.create_publisher(Imu, "/quad/imu", 10)
        self.pub_path = self.create_publisher(PathMsg, "/quad/path", 1)
        self.pub_model = self.create_publisher(MarkerArray, "/quad/model", 1)
        self.pub_pass = self.create_publisher(Int32, "/track/gates_passed", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_gates = self.create_publisher(MarkerArray, "/track/gates", latched)
        self.create_subscription(TwistStamped, "/quad/cmd", self.on_cmd, 10)

        self.path = PathMsg()
        self.path.header.frame_id = "world"
        self.last_cmd_wall = -1.0
        self.n_passed = 0
        self.publish_gates()
        self.publish_model()
        self.t_wall0 = time.perf_counter()
        self.create_timer(1.0 / self.rate, self.tick)
        self.create_timer(0.1, self.publish_path)
        self.create_timer(1.0, self.publish_model)

        fpv_hz = self.get_parameter("fpv_hz").value
        if fpv_hz > 0:
            self.renderer = mujoco.Renderer(self.sim.model, 240, 320)
            self.pub_img = self.create_publisher(Image, "/quad/fpv/image_raw", 1)
            self.create_timer(1.0 / fpv_hz, self.publish_fpv)

    # ---------------------------------------------------------------- inputs
    def on_cmd(self, msg: TwistStamped):
        a = msg.twist.angular
        self.sim.external_action = (msg.twist.linear.z, np.array([a.x, a.y, a.z]))
        self.last_cmd_wall = time.perf_counter()

    # ------------------------------------------------------------------ loop
    def tick(self):
        if self.sim.external_action is not None and time.perf_counter() - self.last_cmd_wall > 0.25:
            self.sim.external_action = None
            self.get_logger().info("no /quad/cmd for 0.25 s -> autopilot")
        # keep sim time locked to wall time (catch up if the timer is late)
        target = time.perf_counter() - self.t_wall0
        n = 0
        while self.sim.t < target and n < 5 * self.substeps:
            self.sim.step(1)
            n += 1
        while self.n_passed < len(self.sim.gates_passed):
            self.pub_pass.publish(Int32(data=self.sim.gates_passed[self.n_passed][1]))
            self.n_passed += 1
        self.publish_state()

    def stamp(self):
        return self.get_clock().now().to_msg()

    def publish_state(self):
        d = self.sim.data
        now = self.stamp()
        p, q = d.qpos[:3], d.qpos[3:7]   # MuJoCo quat = w, x, y, z

        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = now, "world", "base_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, p)
        r = t.transform.rotation
        r.w, r.x, r.y, r.z = map(float, q)
        self.tf.sendTransform(t)

        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = now, "world", "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, p)
        o.pose.pose.orientation = t.transform.rotation
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        v_body = R.reshape(3, 3).T @ d.qvel[:3]
        tw = o.twist.twist
        tw.linear.x, tw.linear.y, tw.linear.z = map(float, v_body)
        tw.angular.x, tw.angular.y, tw.angular.z = map(float, d.qvel[3:6])
        self.pub_odom.publish(o)

        imu = Imu()
        imu.header.stamp, imu.header.frame_id = now, "base_link"
        imu.orientation = t.transform.rotation
        g = d.sensor("gyro").data
        acc = d.sensor("accel").data
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, g)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = map(float, acc)
        self.pub_imu.publish(imu)

        ps = PoseStamped()
        ps.header = o.header
        ps.pose = o.pose.pose
        self.path.poses.append(ps)
        self.path.poses = self.path.poses[-1500:]

    def publish_path(self):
        self.path.header.stamp = self.stamp()
        self.pub_path.publish(self.path)

    def publish_fpv(self):
        self.renderer.update_scene(self.sim.data, camera="fpv")
        img = self.renderer.render()
        m = Image()
        m.header.stamp, m.header.frame_id = self.stamp(), "fpv"
        m.height, m.width = img.shape[:2]
        m.encoding, m.step = "rgb8", img.shape[1] * 3
        m.data = img.tobytes()
        self.pub_img.publish(m)

    # -------------------------------------------------------------- markers
    def publish_gates(self):
        arr = MarkerArray()
        for i, g in enumerate(self.sim.track.gates):
            h = g.inner / 2 + g.bar / 2
            for j, (dy, dz, sy, sz) in enumerate([(h, 0, g.bar, g.inner + 2 * g.bar),
                                                   (-h, 0, g.bar, g.inner + 2 * g.bar),
                                                   (0, h, g.inner + 2 * g.bar, g.bar),
                                                   (0, -h, g.inner + 2 * g.bar, g.bar)]):
                m = Marker()
                m.header.frame_id = "world"
                m.ns, m.id, m.type, m.action = "gates", i * 4 + j, Marker.CUBE, Marker.ADD
                c, s = math.cos(g.yaw), math.sin(g.yaw)
                m.pose.position.x = g.pos[0] - s * dy
                m.pose.position.y = g.pos[1] + c * dy
                m.pose.position.z = g.pos[2] + dz
                w, x, y, z = _quat_from_yaw(g.yaw)
                m.pose.orientation.w, m.pose.orientation.z = w, z
                m.scale.x, m.scale.y, m.scale.z = g.bar, sy, sz
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.45, 0.05, 1.0
                arr.markers.append(m)
        self.pub_gates.publish(arr)

    def publish_model(self):
        """Drone body as frame-locked markers on base_link (generated from the design)."""
        d = self.sim.design
        arr = MarkerArray()

        def mk(i, typ, pos, scale, rgba, pts=None):
            m = Marker()
            m.header.frame_id = "base_link"
            m.ns, m.id, m.type, m.action, m.frame_locked = "quad", i, typ, Marker.ADD, True
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pos)
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = map(float, scale)
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            return m

        cx, cy, cz = d.core_size
        arr.markers.append(mk(0, Marker.CUBE, (0, 0, 0), (2 * cx, 2 * cy, 2 * cz), (0.15, 0.15, 0.18, 1.0)))
        for k, ((x, y), s) in enumerate(zip(d.motor_positions(), d.spin_dirs)):
            L = math.hypot(x, y)
            arm = mk(10 + k, Marker.CUBE, (x / 2, y / 2, 0), (L, 0.016, 0.01), (0.1, 0.1, 0.1, 1.0))
            yaw = math.atan2(y, x)
            arm.pose.orientation.w, arm.pose.orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)
            arr.markers.append(arm)
            col = (0.1, 0.6, 1.0, 0.5) if s > 0 else (1.0, 0.35, 0.1, 0.5)
            r = 2 * d.prop_radius
            arr.markers.append(mk(20 + k, Marker.CYLINDER, (x, y, 0.03), (r, r, 0.004), col))
        self.pub_model.publish(arr)


def main():
    rclpy.init()
    node = QuadBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
