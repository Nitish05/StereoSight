#!/usr/bin/env python3

"""
YoloV8Follower Node

Usage:
 1. Source your ROS2 and workspace installs:
      source /opt/ros/<distro>/setup.bash
      source ~/your_ws/install/setup.bash

 2. Install Python dependencies:
      pip install ultralytics opencv-python

 3. Ensure ROS2 packages:
      sudo apt install ros-<distro>-cv-bridge ros-<distro>-px4-msgs

 4. Build & source your workspace:
      colcon build --packages-select <your_package>
      source install/setup.bash

 5. Launch PX4 (SITL or hardware) and wait for vehicle to connect.

 6. Run this node:
      ros2 run <your_package> object_follower.py

 7. Focus the `Detections` window, press a number key (0-9) to select a detected object.

 8. Arm & switch to OFFBOARD:
    • In QGroundControl: click ARM then set mode to OFFBOARD, or
    • Uncomment `node.arm_and_offboard()` in `main()` to have the node send VehicleCommand.

Once armed and in OFFBOARD, the drone will yaw to center the chosen detection and adjust position.

Topics:
 • Subscribes:
    - /image_rect/compressed       (sensor_msgs/msg/CompressedImage)
    - /fmu/out/vehicle_local_position (px4_msgs/msg/VehicleLocalPosition)
    - /fmu/out/vehicle_status        (px4_msgs/msg/VehicleStatus)
    - /fmu/out/vehicle_control_mode  (px4_msgs/msg/VehicleControlMode)

 • Publishes:
    - /fmu/in/trajectory_setpoint   (px4_msgs/msg/TrajectorySetpoint)
    - /fmu/in/offboard_control_mode (px4_msgs/msg/OffboardControlMode)
    - /fmu/in/vehicle_command       (px4_msgs/msg/VehicleCommand)
    - /image_rect/yolo_annotated     (sensor_msgs/msg/Image)

Parameters:
  - follow_distance (float, default 0.5)  m behind target
  - hover_height    (float, default -1.0) m offset down
  - yaw_gain        (float, default 1.0) rad per normalized pixel error
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleStatus,
    VehicleControlMode,
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand
)

class YoloV8Follower(Node):
    def __init__(self):
        super().__init__('yolo_v8_follower')

        # --- parameters (tweak as needed) ---
        self.declare_parameter('follow_distance', 0.5)   # m behind tag
        self.declare_parameter('hover_height', -1.0)     # m offset in z
        self.declare_parameter('yaw_gain', 1.0)          # rad per normalized pixel error

        self.follow_dist  = self.get_parameter('follow_distance').value
        self.hover_height = self.get_parameter('hover_height').value
        self.K_YAW        = self.get_parameter('yaw_gain').value

        # --- YOLO model + CV bridge ---
        self.model  = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # --- state for tracking & control ---
        self.target_index      = None
        self.desired_area      = None
        self.last_center       = None
        self.current_local_pos = None
        self.offboard_enabled  = False

        # --- pixel → meter gains (tune these!) ---
        self.K_LAT  = 0.1   # lateral (east) gain
        self.K_VERT = 0.001 # vertical (down) gain
        self.K_FWD  = 0.1   # forward (north) gain

        # --- QoS for PX4 topics ---
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # --- Subscriptions ---
        self.create_subscription(
            CompressedImage,
            '/image_rect/compressed',
            self.image_callback,
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
        self.pub_annotated = self.create_publisher(
            Image,
            '/image_rect/yolo_annotated',
            QoSProfile(depth=1)
        )
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

        # send offboard heartbeat at 20 Hz
        self.create_timer(0.05, self.publish_offboard_control_heartbeat_signal)

        cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
        self.get_logger().info('YoloV8 follower node ready.')

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position     = True   # enable position+yaw control
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
        self.get_logger().debug(f"Offboard enabled: {self.offboard_enabled}")

    def arm_and_offboard(self):
        # Arm
        arm_cmd = VehicleCommand()
        arm_cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        arm_cmd.param1  = 1.0
        self.cmd_pub.publish(arm_cmd)
        # Switch to OFFBOARD
        mode_cmd = VehicleCommand()
        mode_cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        mode_cmd.param1  = 1.0  # custom
        mode_cmd.param2  = 6.0  # OFFBOARD
        self.cmd_pub.publish(mode_cmd)
        self.get_logger().info("Sent ARM + OFFBOARD commands")

    def image_callback(self, msg: CompressedImage):
        # 1) Decode
        np_arr = np.frombuffer(msg.data, np.uint8)
        img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        h, w = img.shape[:2]
        # 2) YOLO
        results = self.model(img, verbose=False, show=False)[0]
        boxes   = results.boxes.xyxy.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()
        clss    = results.boxes.cls.cpu().numpy().astype(int)

        # 3) Draw
        for i, (box, conf, c) in enumerate(zip(boxes, confs, clss)):
            x1,y1,x2,y2 = box.astype(int)
            label = f'{i}: {self.model.names[c]} {conf:.2f}'
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, label, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        cv2.imshow('Detections', img)
        key = cv2.waitKey(1) & 0xFF

        # 4) Select target
        if self.target_index is None and len(boxes)>0:
            if 48<=key<=57:
                idx = key-48
                if idx < len(boxes):
                    self.target_index = idx
                    x1,y1,x2,y2 = boxes[idx].astype(int)
                    self.last_center  = ((x1+x2)/2, (y1+y2)/2)
                    self.desired_area = (x2-x1)*(y2-y1)
                    self.get_logger().info(
                        f'Selected object {idx} (area={self.desired_area})'
                    )

        # 5) Tracking & set‑point
        if self.target_index is not None and self.current_local_pos is not None:
            if len(boxes)==0:
                self.get_logger().warn('No detections: clearing target')
                self.target_index = None
            else:
                # match to last center
                centers = [((b[0]+b[2])/2,(b[1]+b[3])/2) for b in boxes]
                dists   = [math.hypot(cx-self.last_center[0],
                                      cy-self.last_center[1])
                           for cx,cy in centers]
                best = int(np.argmin(dists))
                box = boxes[best].astype(int)
                cx,cy = centers[best]
                area  = (box[2]-box[0])*(box[3]-box[1])
                self.last_center = (cx,cy)

                # normalized pixel errors
                ex_norm = (cx - w/2) / w   # + → object is right
                ey_norm = (cy - h/2) / h   # + → object is below

                pos = self.current_local_pos
                # compute XYZ set‑points
                north_sp = pos.x + self.K_FWD * ((self.desired_area-area)/self.desired_area)
                east_sp  = pos.y + self.K_LAT * ex_norm
                down_sp  = pos.z + self.K_VERT * ey_norm

                # compute yaw set‑point
                desired_heading = pos.heading + self.K_YAW * ex_norm

                sp = TrajectorySetpoint()
                sp.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
                sp.position     = [north_sp, east_sp, down_sp]
                sp.velocity     = [0.0, 0.0, 0.0]
                sp.acceleration = [0.0, 0.0, 0.0]
                sp.jerk         = [0.0, 0.0, 0.0]
                sp.yaw          = desired_heading
                sp.yawspeed     = 0.0

                # publish!
                if self.offboard_enabled:
                    self.pub_sp.publish(sp)
                    self.get_logger().info(
                        f"SP→ N:{north_sp:.2f},E:{east_sp:.2f},D:{down_sp:.2f},"
                        f" yaw:{desired_heading:.2f}"
                    )
                else:
                    self.get_logger().info('OFFBOARD not active; skipping setpoint')

        # 6) annotated image out
        annotated = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        annotated.header = msg.header
        self.pub_annotated.publish(annotated)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Follower()
    # you can call node.arm_and_offboard() here or trigger from QGC
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
