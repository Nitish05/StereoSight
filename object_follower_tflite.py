#!/usr/bin/env python3

"""
TFLiteFollower Node

Subscribes to:
  • /tflite_data                  (voxl_msgs/msg/Aidetection)
  • /fmu/out/vehicle_local_position (px4_msgs/msg/VehicleLocalPosition)
  • /fmu/out/vehicle_status        (px4_msgs/msg/VehicleStatus)
  • /fmu/out/vehicle_control_mode  (px4_msgs/msg/VehicleControlMode)

Publishes:
  • /fmu/in/trajectory_setpoint   (px4_msgs/msg/TrajectorySetpoint)
  • /fmu/in/offboard_control_mode (px4_msgs/msg/OffboardControlMode)
  • /fmu/in/vehicle_command       (px4_msgs/msg/VehicleCommand)
"""

import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from voxl_msgs.msg import Aidetection
from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleStatus,
    VehicleControlMode,
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand
)


class TfliteFollower(Node):
    def __init__(self, target_class: str):
        super().__init__('tflite_follower')
        self.target_class = target_class
        self.get_logger().info(f"Tracking target class: '{self.target_class}'")

        # --- parameters ---
        self.declare_parameter('follow_distance', 0.5)
        self.declare_parameter('hover_height', -1.0)
        self.declare_parameter('yaw_gain', 1.0)
        # image resolution for bounding-box normalization
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        self.follow_dist  = self.get_parameter('follow_distance').value
        self.hover_height = self.get_parameter('hover_height').value
        self.K_YAW        = self.get_parameter('yaw_gain').value
        self.img_w        = self.get_parameter('image_width').value
        self.img_h        = self.get_parameter('image_height').value

        # pixel→meter gains (tuning)
        self.K_LAT  = 0.1
        self.K_VERT = 0.001
        self.K_FWD  = 0.1

        # state
        self.current_local_pos = None
        self.offboard_enabled  = False
        self.desired_area      = None
        self.last_center       = None

        # PX4 QoS
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # --- Subscribers ---
        self.create_subscription(
            Aidetection,
            '/tflite_data',
            self.detection_callback,
            img_qos
        )
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.local_position_callback,
            px4_qos
        )
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            px4_qos
        )
        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.control_mode_callback,
            px4_qos
        )

        # --- Publishers ---
        self.pub_sp = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos
        )
        self.offb_ctrl_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            px4_qos
        )

        # heartbeat for offboard control
        self.create_timer(0.05, self.publish_offboard_control_heartbeat_signal)

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.offb_ctrl_pub.publish(msg)

    def status_callback(self, msg: VehicleStatus):
        armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.get_logger().info(f"Armed={armed}, nav_state={msg.nav_state}")

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.current_local_pos = msg

    def control_mode_callback(self, msg: VehicleControlMode):
        self.offboard_enabled = bool(msg.flag_control_offboard_enabled)

    def detection_callback(self, msg: Aidetection):
        cls = msg.class_name
        self.get_logger().info(f'detected class name - {cls}')
        if cls != self.target_class:
            # self.get_logger().info(f"Detected '{cls}', not tracking '{self.target_class}'.")
            return

        # target detected
        self.get_logger().info(f"Detected target class '{cls}' — processing tracking.")

        # bounding box center and area
        x_min, y_min, x_max, y_max = msg.x_min, msg.y_min, msg.x_max, msg.y_max
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        area = (x_max - x_min) * (y_max - y_min)

        # initialize desired area/center on first detection
        if self.desired_area is None or self.last_center is None:
            self.desired_area = area
            self.last_center  = (cx, cy)
            self.get_logger().info(f"Initialized desired area={self.desired_area:.0f}")
            return

        # compute pixel errors normalized
        ex = (cx - self.img_w / 2.0) / self.img_w
        ey = (cy - self.img_h / 2.0) / self.img_h

        # ensure we have current position
        pos = self.current_local_pos
        if pos is None:
            self.get_logger().warn("Local position unknown; cannot compute setpoint.")
            return

        # compute new setpoint
        north_sp = pos.x + self.K_FWD * ((self.desired_area - area) / self.desired_area)
        east_sp  = pos.y + self.K_LAT * ex
        down_sp  = pos.z + self.K_VERT * ey
        yaw_sp   = pos.heading + self.K_YAW * ex

        sp = TrajectorySetpoint()
        sp.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        sp.position     = [north_sp, east_sp, down_sp]
        sp.velocity     = [0.0, 0.0, 0.0]
        sp.acceleration = [0.0, 0.0, 0.0]
        sp.jerk         = [0.0, 0.0, 0.0]
        sp.yaw          = yaw_sp
        sp.yawspeed     = 0.0

        if self.offboard_enabled:
            self.pub_sp.publish(sp)
        else:
            self.get_logger().info("OFFBOARD not active; skipping setpoint")

    def destroy_node(self):
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(
        description="TFLite follower: tracks only a specified class from /tflite_data"
    )
    parser.add_argument(
        '--object', '-o',
        required=True,
        help="Name of the object class to track (e.g., 'chair')"
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = TfliteFollower(target_class=args.object)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
