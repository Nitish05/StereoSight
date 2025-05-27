#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import heapq
import cv2

class DijkstraPlanner(Node):
    def __init__(self):
        super().__init__('dijkstra_planner')

        self.scale = 10
        self.window_name = 'Dijkstra Path'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.drone_position = None 

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        # self.create_subscription(VehicleLocalPosition,
        #     '/fmu/out/vehicle_local_position',
        #     self.position_callback, qos_profile=px4_qos)
        self.create_subscription(Odometry,
            '/local_position_odom',
            self.position_callback, 10)
        self.create_subscription(OccupancyGrid, 'occupancy_grid', self.grid_callback, 10)
        
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)

    def position_callback(self, msg: Odometry):
        # Update the drone's current position
        self.drone_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )


    def grid_callback(self, msg: OccupancyGrid):
        h, w   = msg.info.height, msg.info.width
        res    = msg.info.resolution
        data   = np.array(msg.data, dtype=np.int8).reshape((h, w))

        cost   = np.where(data > 50, np.inf, 1.0)
        # if self.drone_position is not None:
        #     start = (int((self.drone_position[0] - msg.info.origin.position.x) / res * self.scale), int((self.drone_position[1] - msg.info.origin.position.y) / res * self.scale))
        # else:
        start  = (h//2, 0)
        goal   = (h//2, w-1)
        path   = self.compute_dijkstra(cost, start, goal)

        if path is None:
            self.get_logger().warn('No path found')
        else:
            # ---- PRINT THE PATH ----
            print(f"Grid indices (row, col): {path}")
            world_pts = [
                (
                    msg.info.origin.position.x + (c + 0.5) * res,
                    msg.info.origin.position.y + (r + 0.5) * res
                )
                for (r, c) in path
            ]
            print(f"World setpoints (x, y): {world_pts}")

            path_msg = Path()
            path_msg.header = msg.header
            for (r, c) in path:
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position.x = msg.info.origin.position.x + (c + 0.5) * res
                pose.pose.position.y = msg.info.origin.position.y + (r + 0.5) * res
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            self.path_pub.publish(path_msg)

        # --- OpenCV visualization (unchanged) ---
        occ_img = (data <= 50).astype(np.uint8) * 255
        disp    = cv2.resize(occ_img, (w*self.scale, h*self.scale),
                             interpolation=cv2.INTER_NEAREST)
        disp_color = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        if path:
            pts = np.array([((c+0.5)*self.scale, (r+0.5)*self.scale)
                            for (r, c) in path], dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(disp_color, [pts], False, (0,0,255), 2)

        # Draw the drone's position as a dot
        if self.drone_position is not None:
            drone_x = int((self.drone_position[0] - msg.info.origin.position.x) / res * self.scale)
            drone_y = int((self.drone_position[1] - msg.info.origin.position.y) / res * self.scale)
            cv2.circle(disp_color, (drone_x, drone_y), 5, (255, 0, 0), -1)

        cv2.imshow(self.window_name, disp_color)
        cv2.waitKey(1)

    def compute_dijkstra(self, cost, start, goal):
        h, w = cost.shape
        dist = np.full((h, w), np.inf, dtype=float)
        prev = {}
        dist[start] = 0.0
        pq = [(0.0, start)]
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]

        while pq:
            d, (r, c) = heapq.heappop(pq)
            if (r, c) == goal:
                break
            if d > dist[r, c]:
                continue
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and cost[nr, nc] < np.inf:
                    nd = d + cost[nr, nc]
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        prev[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (nd, (nr, nc)))

        if dist[goal] == np.inf:
            return None

        path = [goal]
        cur  = goal
        while cur != start:
            cur = prev[cur]
            path.append(cur)
        return list(reversed(path))


def main():
    rclpy.init()
    node = DijkstraPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
