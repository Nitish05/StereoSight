#!/usr/bin/env python3

import threading
import time

import numpy as np
from scipy.ndimage import median_filter

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude

class OccupancyVisualizer(Node):
    def __init__(self):
        super().__init__('occupancy_visualizer')

        # ---- parameters ----
        self.ORIGIN_RADIUS = 0.5        # ignore points within this radius (m)
        self.GRID_BINS    = 40         # total cells per axis
        self.scale        = 10         # display scale (pixels per cell)

        # compute resolution (meters per cell) from desired total X span (4 m)
        resolution = 4.0 / self.GRID_BINS

        # X only positive: from 0 to 4 m (ahead of drone)
        self.X_RANGE = (0.0, self.GRID_BINS * resolution)
        # Y symmetric: −2 m to +2 m (left/right of drone)
        half_cells = self.GRID_BINS // 2
        self.Y_RANGE = (
            -half_cells * resolution,
             half_cells * resolution
        )

        # build the bin edges
        self.x_edges = np.linspace(
            self.X_RANGE[0], self.X_RANGE[1],
            self.GRID_BINS + 1)
        self.y_edges = np.linspace(
            self.Y_RANGE[0], self.Y_RANGE[1],
            self.GRID_BINS + 1)

        # ---- state ----
        self.latest_points = None
        self.points_lock   = threading.Lock()
        self.drone_pos     = np.zeros(3, dtype=np.float32)
        self.att_q         = (1.0, 0.0, 0.0, 0.0)  # quaternion (w,x,y,z)

        # ---- QoS profiles ----
        pc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ---- subscriptions ----
        self.create_subscription(
            PointCloud2,
            '/voa_pc_out',
            self.pc_callback,
            qos_profile=pc_qos
        )
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos_profile=px4_qos
        )
        self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile=px4_qos
        )

        # prepare the OpenCV window for resizing
        cv2.namedWindow('Occupancy Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            'Occupancy Grid',
            self.GRID_BINS * self.scale,
            self.GRID_BINS * self.scale)

        # spin ROS callbacks in background
        threading.Thread(
            target=rclpy.spin,
            args=(self,),
            daemon=True
        ).start()

    def pc_callback(self, msg: PointCloud2):
        pts = [(x, y, z) for x, y, z in pc2.read_points(
            msg, field_names=('x','y','z'), skip_nans=True)]
        if pts:
            with self.points_lock:
                self.latest_points = np.array(pts, dtype=np.float32)

    def position_callback(self, msg: VehicleLocalPosition):
        # NED position
        self.drone_pos = np.array([msg.x, msg.y, msg.z],
                                   dtype=np.float32)

    def attitude_callback(self, msg: VehicleAttitude):
        # store quaternion
        self.att_q = (msg.q[0], msg.q[1],
                      msg.q[2], msg.q[3])

    def run(self):
        rate_hz = 30.0
        dt = 1.0 / rate_hz

        while rclpy.ok():
            with self.points_lock:
                pts = (None if self.latest_points is None
                       else self.latest_points.copy())

            if pts is not None and pts.size:
                # down‑sample
                if pts.shape[0] > 50000:
                    idx = np.random.choice(
                        pts.shape[0], 50000, replace=False)
                    pts = pts[idx]

                xs, ys = pts[:,0], pts[:,1]
                mask = np.hypot(xs, ys) > self.ORIGIN_RADIUS
                fx, fy = xs[mask], ys[mask]

                # occupancy histogram (fy → rows, fx → cols)
                hist, _, _ = np.histogram2d(
                    fy, fx,
                    bins=[self.y_edges, self.x_edges]
                )
                hist_filt = median_filter(hist, size=7)
                thresh = hist_filt.mean()

                # build OpenCV image
                occ_img = (hist_filt > thresh).astype(np.uint8) * 255

                # scale up for display
                disp = cv2.resize(
                    occ_img,
                    (self.GRID_BINS * self.scale,
                     self.GRID_BINS * self.scale),
                    interpolation=cv2.INTER_NEAREST
                )

                # show occupancy
                cv2.imshow('Occupancy Grid', disp)
                cv2.waitKey(1)

            time.sleep(dt)

        # clean up on exit
        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = OccupancyVisualizer()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
