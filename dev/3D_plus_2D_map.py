#!/usr/bin/env python3
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
from sklearn.cluster import DBSCAN
from scipy.ndimage import median_filter, gaussian_filter

# -----------------------------------------------------------------------------
# Globals for shared point-cloud data
# -----------------------------------------------------------------------------
latest_points = None
points_lock = threading.Lock()
# ignore radius around origin (drone body), in meters
ORIGIN_RADIUS = 0.5
# clustering parameters
EPS = 0.2
MIN_SAMPLES = 30
# grid definition
GRID_BINS = 40
X_RANGE = (0, 4)
Y_RANGE = (0, 4)


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
    node = Node('pc_viz_standalone')

    # 2. Define QoS matching publishers (BEST_EFFORT, small buffer)
    pc_qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=5
    )

    # 3. Create subscriptions
    for topic in ['/voa_pc_out']:#['/stereo_front_pc']:
        node.create_subscription(
            PointCloud2,
            topic,
            pc_callback,
            qos_profile=pc_qos
        )

    # 4. Spin ROS in a daemon thread
    spin_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    # 5. Set up interactive Matplotlib with 1x5 layout
    plt.ion()
    fig = plt.figure(figsize=(20, 6))
    ax3d     = fig.add_subplot(1, 5, 1, projection='3d')
    ax2d_raw = fig.add_subplot(1, 5, 2)
    ax2d_pre = fig.add_subplot(1, 5, 3)
    ax2d_post= fig.add_subplot(1, 5, 4)
    ax2d_dyn = fig.add_subplot(1, 5, 5)

    # Prepare grid edges and centers
    x_edges = np.linspace(X_RANGE[0], X_RANGE[1], GRID_BINS + 1)
    y_edges = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_BINS + 1)

    try:
        while rclpy.ok():
            with points_lock:
                pts = latest_points.copy() if latest_points is not None else None

            if pts is not None and pts.size:
                # Downsample for performance
                if pts.shape[0] > 50000:
                    idx = np.random.choice(pts.shape[0], 50000, replace=False)
                    pts = pts[idx]

                xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]

                # filter out drone body
                r = np.sqrt(xs**2 + ys**2)
                mask_outer = r > ORIGIN_RADIUS
                fpts = pts[mask_outer]
                fx, fy, fz = fpts[:,0], fpts[:,1], fpts[:,2]

                # --- 3D view ---
                ax3d.cla()
                ax3d.scatter(xs, ys, zs, s=1, c='gray', alpha=0.3)
                ax3d.scatter(fx, fy, fz, s=1, c='b', alpha=0.5)
                ax3d.set_title('3D Point Cloud')
                ax3d.set_xlabel('X')
                ax3d.set_ylabel('Y')
                ax3d.set_zlabel('Z')
                ax3d.set_xlim(X_RANGE)
                ax3d.set_ylim(Y_RANGE)
                ax3d.set_zlim(0, 10)

                # --- 2D raw scatter ---
                ax2d_raw.cla()
                ax2d_raw.scatter(fx, fy, s=1, c='b', alpha=0.5)
                ax2d_raw.set_title('2D Raw XY')
                ax2d_raw.set_xlim(X_RANGE)
                ax2d_raw.set_ylim(Y_RANGE)

                # --- occupancy grid before filter ---
                hist_pre, _, _ = np.histogram2d(fy, fx, bins=[y_edges, x_edges])
                ax2d_pre.cla()
                ax2d_pre.imshow(hist_pre, origin='lower',
                                 extent=(X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]),
                                 aspect='auto')
                ax2d_pre.set_title('Grid Before Filter')
                ax2d_pre.set_xlabel('X')
                ax2d_pre.set_ylabel('Y')

                # --- occupancy grid after median filter ---
                hist_post = median_filter(hist_pre, size=7)
                # hist_post = gaussian_filter(hist_pre, sigma=2)
                ax2d_post.cla()
                ax2d_post.imshow(hist_post, origin='lower',
                                  extent=(X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]),
                                  aspect='auto')
                ax2d_post.set_title('Grid After Median Filter')
                ax2d_post.set_xlabel('X')
                ax2d_post.set_ylabel('Y')

                # --- dynamic occupancy map ---
                # threshold based on mean
                thresh = hist_post.mean()
                ax2d_dyn.cla()
                for i in range(GRID_BINS):
                    for j in range(GRID_BINS):
                        count = hist_post[i, j]
                        color = 'red' if count > thresh else 'green'
                        xs0, xs1 = x_edges[j], x_edges[j+1]
                        ys0, ys1 = y_edges[i], y_edges[i+1]
                        ax2d_dyn.fill([xs0, xs1, xs1, xs0],
                                      [ys0, ys0, ys1, ys1],
                                      color=color, alpha=0.5)
                ax2d_dyn.set_xlim(X_RANGE)
                ax2d_dyn.set_ylim(Y_RANGE)
                ax2d_dyn.set_title('Dynamic Occupancy Map')
                ax2d_dyn.set_xlabel('X')
                ax2d_dyn.set_ylabel('Y')

                fig.canvas.draw()
                fig.canvas.flush_events()

            time.sleep(0.00001)

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        spin_thread.join()