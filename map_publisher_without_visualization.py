#!/usr/bin/env python3

import threading
import time
import numpy as np
import math
from scipy.ndimage import median_filter, uniform_filter, binary_dilation
from sklearn.cluster import DBSCAN
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Quaternion

class OccupancyPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_publisher')

        # ---- parameters ----
        self.ORIGIN_RADIUS = 0.35       # ignore points within this radius (m)
        self.GRID_BINS    = 60         # cells per axis
        self.resolution   = 4.0 / self.GRID_BINS  # meters per cell
        self.X_RANGE      = (0.0, 4.0)             # forward 4 m
        half = self.GRID_BINS // 2
        self.Y_RANGE      = (-half * self.resolution,
                              half * self.resolution)

        # padding parameter: number of cells to pad around each obstacle
        self.declare_parameter('padding', 2)
        self.padding = self.get_parameter('padding').value

        # clustering parameters
        self.MIN_POINTS_PER_CLUSTER = 30  # minimum points to keep a cluster
        self.DBSCAN_EPS = 0.2             # max distance (m) between points in a cluster
        self.MIN_POINTS_PER_CELL = 5      # minimum points to consider a cell occupied

        # edges for histogram2d
        self.x_edges = np.linspace(self.X_RANGE[0], self.X_RANGE[1],
                                   self.GRID_BINS + 1)
        self.y_edges = np.linspace(self.Y_RANGE[0], self.Y_RANGE[1],
                                   self.GRID_BINS + 1)

        # ---- state ----
        self.latest_points = None
        self.points_lock   = threading.Lock()
        self.drone_pos     = np.zeros(3, dtype=np.float32)
        self.att_q         = (1.0, 0.0, 0.0, 0.0)  # (w,x,y,z)

        # ---- publisher ----
        self.map_pub = self.create_publisher(OccupancyGrid,
                                             'occupancy_grid', 10)
        self.frame_id = 'map'

        # ---- QoS ----
        pc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5
        )
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ---- subscriptions ----
        self.create_subscription(PointCloud2,
            '/stereo_front_pc', self.pc_callback, qos_profile=pc_qos)
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback, qos_profile=px4_qos)
        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback, qos_profile=px4_qos)

        # camera→NED fixed rotation
        β = math.radians(90)
        R_rot_y = np.array([
            [ math.cos(β), 0, math.sin(β)],
            [ 0,           1, 0        ],
            [-math.sin(β), 0, math.cos(β)]
        ])
        alpha = math.radians(90)
        R_rot_z = np.array([
            [ math.cos(alpha), -math.sin(alpha), 0],
            [ math.sin(alpha),  math.cos(alpha), 0],
            [ 0,            0,           1]
        ])
        self.R_cam_to_ned = R_rot_y @ R_rot_z

        threading.Thread(target=rclpy.spin,
                         args=(self,), daemon=True).start()

    def pc_callback(self, msg: PointCloud2):
        # rotate from camera into NED once
        pts = [
            self.R_cam_to_ned @ np.array([x, y, z])
            for x, y, z in pc2.read_points(
                msg, field_names=('x','y','z'), skip_nans=True
            )
        ]
        if pts:
            with self.points_lock:
                self.latest_points = np.array(pts, dtype=np.float32)

    def position_callback(self, msg: VehicleLocalPosition):
        # store current NED position
        self.drone_pos = np.array([msg.x, msg.y, msg.z],
                                  dtype=np.float32)

    def attitude_callback(self, msg: VehicleAttitude):
        # store quaternion (w,x,y,z)
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
                # down-sample if needed
                if pts.shape[0] > 50000:
                    idx = np.random.choice(pts.shape[0],
                                           50000, replace=False)
                    pts = pts[idx]

                # 1) compute relative vectors in NED body frame
                rel_xy = pts[:, :2] - self.drone_pos[:2]

                # 2) build yaw‐rotation matrix
                w, x, y, z = self.att_q
                yaw = math.atan2(2*(w*z + x*y),
                                 1 - 2*(y*y + z*z))
                c, s = math.cos(yaw), math.sin(yaw)
                R2 = np.array([[c, -s],
                               [s,  c]], dtype=np.float32)

                # 3) rotate about drone, then translate back into world
                rot_xy   = rel_xy  # (R2 @ rel_xy.T).T  # if you want world-aligned
                world_xy = rot_xy + self.drone_pos[:2]

                # 4) mask out anything too close
                mask = np.hypot(rot_xy[:,0], rot_xy[:,1]) > self.ORIGIN_RADIUS

                rot_xy = world_xy[mask]

                # 5) cluster points with DBSCAN to remove noise
                if rot_xy.shape[0] > 0:
                    db = DBSCAN(eps=self.DBSCAN_EPS, min_samples=5).fit(rot_xy)
                    labels = db.labels_
                    # keep points in valid clusters (not noise, and large enough)
                    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                    valid_labels = unique_labels[counts >= self.MIN_POINTS_PER_CLUSTER]
                    valid_mask = np.isin(labels, valid_labels)
                    rot_xy = rot_xy[valid_mask]
                else:
                    rot_xy = np.empty((0, 2), dtype=np.float32)

                # 6) histogram & filtering
                if rot_xy.shape[0] > 0:

                    fx, fy = rot_xy[:,0], rot_xy[:,1]

                    # 5) histogram & filtering
                    hist = np.histogram2d(
                        fy, fx,
                        bins=[self.y_edges, self.x_edges]
                    )[0]
                    filt = hist
                    # filt = uniform_filter(hist, size=9)
                    filt = median_filter(filt, size=3)
                    thresh = filt.mean()
                    occ_binary = filt > thresh
                
                else:
                    occ_binary = np.zeros((self.GRID_BINS, self.GRID_BINS), dtype=bool)

                # 6) pad obstacles with dilation
                struct = np.ones((2*self.padding + 1, 2*self.padding + 1), dtype=bool)
                occ_padded = binary_dilation(occ_binary, structure=struct)

                # convert to occupancy values [0,100]
                occ = occ_padded.astype(np.int8) * 100

                # ---- publish grid with moving origin & yaw ----
                grid = OccupancyGrid()
                grid.header.stamp = self.get_clock().now().to_msg()
                grid.header.frame_id = self.frame_id
                grid.info.resolution = self.resolution
                grid.info.width      = self.GRID_BINS
                grid.info.height     = self.GRID_BINS

                # grid origin = drone + rotated offset
                offset = np.array([self.X_RANGE[0], self.Y_RANGE[0]])
                origin_xy = self.drone_pos[:2] + (R2 @ offset)
                grid.info.origin.position = Point(
                    x=float(origin_xy[0]),
                    y=float(origin_xy[1]),
                    z=0.0
                )

                # rotate grid axes by current yaw
                qz = math.sin(yaw/2.0)
                qw = math.cos(yaw/2.0)
                grid.info.origin.orientation = Quaternion(
                    x=0.0, y=0.0, z=qz, w=qw
                )

                grid.data = occ.flatten(order='C').tolist()
                self.map_pub.publish(grid)

            time.sleep(dt)

def main():
    rclpy.init()
    node = OccupancyPublisher()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
