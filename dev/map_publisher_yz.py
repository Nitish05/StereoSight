#!/usr/bin/env python3

import threading
import time
import numpy as np
from scipy.ndimage import median_filter, uniform_filter
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Quaternion

class OccupancyPublisherXZ(Node):
    def __init__(self):
        super().__init__('occupancy_publisher_xz')

        # ---- parameters ----
        self.ORIGIN_RADIUS = 0.3        # ignore points within this radius (m)
        self.GRID_BINS    = 40          # cells per axis
        self.scale        = 10          # pixels per cell for display
        self.resolution   = 4.0 / self.GRID_BINS  # meters per cell

        # Y covers [-4 … +4] m, Z covers [-4 … +4] m
        self.Y_RANGE = (-2.0, 2.0)
        self.Z_RANGE = (-2.0, 2.0)

        # bin edges for histogram2d: first dim=Z, second dim=Y
        self.y_edges = np.linspace(self.Y_RANGE[0], self.Y_RANGE[1],
                                   self.GRID_BINS + 1)
        self.z_edges = np.linspace(self.Z_RANGE[0], self.Z_RANGE[1],
                                   self.GRID_BINS + 1)

        # ---- state ----
        self.latest_points = None
        self.points_lock   = threading.Lock()
        # drone_pos = (x, y, z)
        self.drone_pos     = np.zeros(3, dtype=np.float32)
        self.att_q         = (1.0, 0.0, 0.0, 0.0)  # (w,x,y,z)

        # ---- publisher ----
        self.map_pub = self.create_publisher(OccupancyGrid,
                                             'occupancy_grid_xz', 10)
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
            '/voa_pc_out', self.pc_callback, qos_profile=pc_qos)
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback, qos_profile=px4_qos)
        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback, qos_profile=px4_qos)

        # OpenCV window
        cv2.namedWindow('Occupancy Grid YZ', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Occupancy Grid YZ',
                        self.GRID_BINS * self.scale,
                        self.GRID_BINS * self.scale)

        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()

    def pc_callback(self, msg: PointCloud2):
        pts = [(x, y, z) for x, y, z in pc2.read_points(
            msg, field_names=('x','y','z'), skip_nans=True)]
        if pts:
            with self.points_lock:
                self.latest_points = np.array(pts, dtype=np.float32)

    def position_callback(self, msg: VehicleLocalPosition):
        # NED → just take x,y,z
        self.drone_pos = np.array([msg.x, msg.y, msg.z],
                                  dtype=np.float32)

    def attitude_callback(self, msg: VehicleAttitude):
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
                # down‑sample if too large
                if pts.shape[0] > 50000:
                    idx = np.random.choice(pts.shape[0],
                                           50000, replace=False)
                    pts = pts[idx]

                # --- transform points into world XY and Z ---
                # subtract drone horizontal position for rotation
                body_xy = pts[:, :2] - self.drone_pos[:2]

                # compute yaw from quaternion
                w, x, y, z = self.att_q
                yaw = np.arctan2(2*(w*z + x*y),
                                 1 - 2*(y*y + z*z))

                # rotate into world XY
                c, s = np.cos(yaw), np.sin(yaw)
                R2 = np.array([[ c, -s],
                               [ s,  c]], dtype=np.float32)
                world_xy = (R2 @ body_xy.T).T

                # world coordinates
                wx = world_xy[:, 0]                   # X axis
                wy = world_xy[:, 1]                   # Y axis
                wz = pts[:, 2] - self.drone_pos[2]    # Z axis

                # mask out points too close
                mask = np.hypot(wx, wy) > self.ORIGIN_RADIUS
                fx, fy, fz = wx[mask], wy[mask], wz[mask]

                # compute centroid in world XY plane
                cx_world = fx.mean()
                cy_world = fy.mean()
                depth = float(np.hypot(cx_world, cy_world))

                # histogram for occupancy in Z–Y plane
                hist = np.histogram2d(
                    fz, fy,
                    bins=[self.z_edges, self.y_edges]
                )[0]
                # filt   = median_filter(hist, size=7)
                # filt   = uniform_filter(hist, size=7)
                filt = uniform_filter(hist, size=9)
                filt = median_filter(filt, size=3)
                thresh = filt.mean()
                occ = (filt > thresh).astype(np.int8) * 100

                # publish OccupancyGrid
                grid = OccupancyGrid()
                grid.header.stamp = self.get_clock().now().to_msg()
                grid.header.frame_id = self.frame_id
                grid.info.resolution = self.resolution
                grid.info.width      = self.GRID_BINS
                grid.info.height     = self.GRID_BINS
                grid.info.origin.position = Point(
                    x=self.Y_RANGE[0],
                    y=self.Z_RANGE[0],  # Z→grid‑Y axis
                    z=0.0
                )
                grid.info.origin.orientation = Quaternion(
                    x=0.0, y=0.0, z=0.0, w=1.0
                )
                grid.data = occ.flatten(order='C').tolist()
                self.map_pub.publish(grid)

                # --- display with centroid overlay ---
                binary = (occ > 50).astype(np.uint8)
                img_gray = binary * 255
                disp_gray = cv2.resize(
                    img_gray,
                    (self.GRID_BINS * self.scale,
                     self.GRID_BINS * self.scale),
                    interpolation=cv2.INTER_NEAREST
                )

                # compute centroid of the blob in the grid
                M = cv2.moments(binary)
                if M['m00'] > 0:
                    # grid‑cell centroid (cols=Y, rows=Z)
                    cx_cell = M['m10'] / M['m00']
                    cy_cell = M['m01'] / M['m00']

                    # pixel coords
                    px = int((cx_cell + 0.5) * self.scale)
                    py = int((cy_cell + 0.5) * self.scale)

                    disp_color = cv2.cvtColor(disp_gray, cv2.COLOR_GRAY2BGR)
                    # draw red dot at centroid
                    cv2.circle(disp_color, (px, py), radius=5,
                               color=(0, 0, 255), thickness=-1)
                    # annotate with XY‐depth
                    cv2.putText(disp_color,
                                f"{depth:.2f} m",
                                (px + 10, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)
                    cv2.imshow('Occupancy Grid YZ', disp_color)
                else:
                    cv2.imshow('Occupancy Grid YZ', disp_gray)

                cv2.waitKey(1)

            time.sleep(dt)

        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = OccupancyPublisherXZ()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
