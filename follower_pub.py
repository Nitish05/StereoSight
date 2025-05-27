#!/usr/bin/env python3

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

        # user inputs
        self.target_class = input("Enter class_name to follow (e.g. 'laptop'): ").strip()
        self.img_w = 1024
        self.img_h = 768

        # control gains & state
        self.K_LAT  = 0.3
        self.K_FWD  = 0.3
        self.K_VERT = 0.0001
        self.K_YAW  = 1.0
        self.desired_area   = None
        self.current_pos    = None
        self.offboard_enabled = False

        # QoS
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

        # subscriptions
        self.create_subscription(Aidetection, '/tflite_data', self.detection_callback, tf)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position',
                                 self.local_position_callback, px4_qos)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status',
                                 self.status_callback, px4_qos)
        self.create_subscription(VehicleControlMode, '/fmu/out/vehicle_control_mode',
                                 self.control_mode_callback, px4_qos)

        # publishers
        self.offb_ctrl_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos
        )
        # now publishing to planner/goal instead of directly to PX4
        self.goal_pub = self.create_publisher(
            TrajectorySetpoint,
            '/planner/goal',
            px4_qos
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            px4_qos
        )

        # heartbeat timer for offboard
        self.create_timer(0.05, self._publish_offboard_heartbeat)

        self.get_logger().info(f"VOXLFollower ready - following '{self.target_class}'")

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
        if msg.class_name != self.target_class:
            return

        cx = (msg.x_min + msg.x_max) / 2.0
        cy = (msg.y_min + msg.y_max) / 2.0
        area = (msg.x_max - msg.x_min) * (msg.y_max - msg.y_min)
        print('area =',area)

        if self.desired_area is None:
            self.desired_area = area
            self.get_logger().info(f"Set desired area = {self.desired_area:.1f}")

        ex = (cx - self.img_w/2) / self.img_w
        ey = (cy - self.img_h/2) / self.img_h

        if self.current_pos is None:
            self.get_logger().warn("No vehicle position yet—skipping setpoint")
            return

        # compute new setpoint
        north_sp = self.current_pos.x + self.K_FWD * ((self.desired_area - area) / self.desired_area)
        east_sp  = self.current_pos.y + self.K_LAT * ex
        down_sp  = self.current_pos.z + self.K_VERT * ey
        desired_yaw = self.current_pos.heading + self.K_YAW * ex

        sp = TrajectorySetpoint()
        sp.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        # round each coordinate to 1 decimal place
        sp.position     = [round(north_sp,1), round(east_sp,1), round(down_sp,1)]
        sp.velocity     = [0.0, 0.0, 0.0]
        sp.acceleration = [0.0, 0.0, 0.0]
        sp.jerk         = [0.0, 0.0, 0.0]
        sp.yaw          = desired_yaw
        sp.yawspeed     = 0.0

        if self.offboard_enabled:
            self.goal_pub.publish(sp)
            self.get_logger().info(
                f"→ goal [N{sp.position[0]:.1f}, E{sp.position[1]:.1f}, D{sp.position[2]:.1f}]"
            )
        else:
            self.get_logger().info("OFFBOARD not active, skipping goal publish")

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
