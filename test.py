#!/usr/bin/env python3

"""
VOXL‑Aided Object Follower Node

Now subscribes to /tflite_data (voxl_msgs/msg/Aidetection) from the VOXL2,
and uses a terminal prompt to select which class_name to follow.
"""

import math
import rclpy
from rclpy.node import Node
from voxl_msgs.msg import Aidetection
from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleStatus,
    VehicleControlMode,
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand
)

class VOXLFollower(Node):
    def __init__(self):
        super().__init__('voxl_object_follower')

        # --- user inputs via terminal ---
        self.target_class = input("Enter class_name to follow (e.g. 'laptop'): ").strip()
        self.img_w = 1024
        self.img_h = 768

        # --- control gains & state ---
        self.K_LAT  = 0.5    # east gain
        self.K_FWD  = 0.5    # north gain (based on area)
        self.K_VERT = 0.0001 # down gain
        self.K_YAW  = 1.0    # yaw gain
        self.desired_area = None
        self.current_pos  = None
        self.offboard_enabled = False

        # --- QoS profiles ---
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        tf = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # --- subscriptions ---
        self.create_subscription(
            Aidetection,
            '/tflite_data',
            self.detection_callback,
            tf
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

        # --- publishers ---
        self.offb_ctrl_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos
        )
        self.sp_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            px4_qos
        )

        # heartbeat timer for offboard
        self.create_timer(0.05, self._publish_offboard_heartbeat)

        self.get_logger().info(f"Following class: '{self.target_class}'")

    def _publish_offboard_heartbeat(self):
        hb = OffboardControlMode()
        hb.position     = True
        hb.velocity     = False
        hb.acceleration = False
        hb.attitude     = False
        hb.body_rate    = False
        hb.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.offb_ctrl_pub.publish(hb)

    def status_callback(self, msg: VehicleStatus):
        armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.get_logger().info(f"Armed={armed}, nav_state={msg.nav_state}")

    def control_mode_callback(self, msg: VehicleControlMode):
        self.offboard_enabled = bool(msg.flag_control_offboard_enabled)

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.current_pos = msg

    def detection_callback(self, msg: Aidetection):
        # only process if it matches the user-selected class
        if msg.class_name != self.target_class:
            return

        # bounding‐box center & area
        cx = (msg.x_min + msg.x_max) / 2.0
        cy = (msg.y_min + msg.y_max) / 2.0
        area = (msg.x_max - msg.x_min) * (msg.y_max - msg.y_min)

        # initialize desired area on first detection
        if self.desired_area is None:
            self.desired_area = area
            self.get_logger().info(f"Set desired area = {self.desired_area:.1f}")

        # normalized pixel errors
        ex = (cx - self.img_w/2) / self.img_w
        ey = (cy - self.img_h/2) / self.img_h

        # compute new setpoint relative to current vehicle_local_position
        if self.current_pos is None:
            self.get_logger().warn("No vehicle position yet—skipping setpoint")
            return

        north_sp = self.current_pos.x + self.K_FWD * ((self.desired_area - area) / self.desired_area)
        east_sp  = self.current_pos.y + self.K_LAT * ex
        down_sp  = self.current_pos.z + self.K_VERT * ey
        desired_yaw = self.current_pos.heading + self.K_YAW * ex

        sp = TrajectorySetpoint()
        sp.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        sp.position     = [north_sp, east_sp, down_sp]
        sp.velocity     = [0.0, 0.0, 0.0]
        sp.acceleration = [0.0, 0.0, 0.0]
        sp.jerk         = [0.0, 0.0, 0.0]
        sp.yaw          = desired_yaw
        sp.yawspeed     = 0.0

        if self.offboard_enabled:
            self.sp_pub.publish(sp)
            self.get_logger().info(
                f"→ SP: N{north_sp:.2f}, E{east_sp:.2f}, D{down_sp:.2f}, Yaw{desired_yaw:.2f}"
            )
        else:
            self.get_logger().info("OFFBOARD not active, skipping SP")

def main(args=None):
    rclpy.init(args=args)
    node = VOXLFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
