#!/usr/bin/env python3
import threading
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
from scipy.ndimage import median_filter

# -----------------------------------------------------------------------------
# Globals for shared point-cloud data
# -----------------------------------------------------------------------------
latest_points = None
points_lock = threading.Lock()

# ignore radius around origin (drone body), in meters
ORIGIN_RADIUS = 0.5

# grid definition
GRID_BINS = 40
X_RANGE = (0.0, 4.0)
Y_RANGE = (0.0, 4.0)


def pc_callback(msg: PointCloud2):
    """Convert incoming PointCloud2 to Nx3 NumPy and store."""
    global latest_points
    pts = []
    for x, y, z in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
        pts.append((x, y, z))
    if pts:
        with points_lock:
            latest_points = np.array(pts, dtype=np.float32)


def ros_spin(node):
    """Background ROS2 spin."""
    rclpy.spin(node)


if __name__ == "__main__":
    # 1. Initialize ROS2
    rclpy.init()
    node = Node('pc_distance_printer')

    # 2. Define QoS matching publishers (BEST_EFFORT, small buffer)
    pc_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=5
    )

    # 3. Create subscriptions
    node.create_subscription(
        PointCloud2,
        '/voa_pc_out',
        pc_callback,
        qos_profile=pc_qos
    )

    # 4. Spin ROS in a daemon thread
    spin_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    # 5. Prepare grid edges
    x_edges = np.linspace(X_RANGE[0], X_RANGE[1], GRID_BINS + 1)
    y_edges = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_BINS + 1)

    try:
        while rclpy.ok():
            # grab the latest point cloud
            with points_lock:
                pts = latest_points.copy() if latest_points is not None else None

            if pts is not None and pts.size:
                # optional down‑sampling
                if pts.shape[0] > 50000:
                    idx = np.random.choice(pts.shape[0], 50000, replace=False)
                    pts = pts[idx]

                xs, ys = pts[:, 0], pts[:, 1]

                # filter out drone body points
                r = np.hypot(xs, ys)
                mask = r > ORIGIN_RADIUS
                fx, fy = xs[mask], ys[mask]

                # build raw occupancy grid
                hist, _, _ = np.histogram2d(fy, fx, bins=[y_edges, x_edges])

                # median‑filter the grid to remove speckle
                hist_filt = median_filter(hist, size=7)

                # threshold = mean count per cell
                thresh = hist_filt.mean()

                # find occupied cells
                occ_indices = np.argwhere(hist_filt > thresh)

                if occ_indices.size:
                    distances = []
                    for i, j in occ_indices:
                        # compute cell center
                        x_center = 0.5 * (x_edges[j] + x_edges[j+1])
                        y_center = 0.5 * (y_edges[i] + y_edges[i+1])
                        d = math.hypot(x_center, y_center)
                        distances.append(d)

                    # sort and print unique distances
                    distances = sorted(set(distances))
                    print(f"Detected {len(distances)} obstacle(s) at distances (m):")
                    print(", ".join(f"{d:.2f}" for d in distances))
                else:
                    print("No obstacles detected above threshold.")

            # small sleep to avoid spamming
            # time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join()
