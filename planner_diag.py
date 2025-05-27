#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleControlMode,
    OffboardControlMode,
    TrajectorySetpoint,
)
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import heapq

class DijkstraPlanner(Node):
    def __init__(self):
        super().__init__('dijkstra_planner')

        # ask user for goal in world coords
        try:
            s = input("Enter GOAL (x y) in world frame (meters): ")
            gx, gy = [float(v) for v in s.strip().split()]
        except Exception:
            self.get_logger().error("Invalid input; defaulting goal to (0,0)")
            gx, gy = 0.0, 0.0
        self.goal_world = (gx, gy)

        # parameters
        self.switch_radius = 0.5   # meters to consider "arrived"
        self.waypoints     = []    # list of (x,y,z)
        self.current_wp    = 0
        self.cost_grid     = None
        self.goal_cell     = None
        self.ox = self.oy = self.res = None

        # PX4 QoS
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.offboard_enabled = False
        self.drone_position   = None

        # subscriptions
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            px4_qos
        )
        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.control_mode_callback,
            px4_qos
        )
        self.create_subscription(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            lambda msg: None,
            px4_qos
        )
        self.create_subscription(
            OccupancyGrid,
            'occupancy_grid',
            self.grid_callback,
            10
        )

        # publishers
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
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)

        # heartbeat & waypoint timers
        self.create_timer(0.05, self.publish_offboard_heartbeat)
        self.create_timer(0.1, self.publish_next_setpoint)

        self.get_logger().info(f"Goal_world = x:{gx:.2f}, y:{gy:.2f}")

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offb_ctrl_pub.publish(msg)

    def position_callback(self, msg: VehicleLocalPosition):
        self.drone_position = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

    def control_mode_callback(self, msg: VehicleControlMode):
        self.offboard_enabled = bool(msg.flag_control_offboard_enabled)

    def grid_callback(self, msg: OccupancyGrid):
        # extract & store latest grid info
        h, w        = msg.info.height, msg.info.width
        self.res    = msg.info.resolution
        self.ox     = msg.info.origin.position.x
        self.oy     = msg.info.origin.position.y

        data        = np.array(msg.data, dtype=np.int8).reshape((h, w))
        self.cost_grid = np.where(data > 50, np.inf, 1.0)

        # compute goal in cell coords
        gx, gy      = self.goal_world
        gc = int((gx - self.ox)/self.res)
        gr = int((gy - self.oy)/self.res)
        self.goal_cell = (
            max(0, min(h-1, gr)),
            max(0, min(w-1, gc))
        )

        # initial plan only if we have no waypoints yet
        start = self._start_cell()
        if not self.waypoints:
            self.plan_path(start)

    def publish_next_setpoint(self):
        if not self.offboard_enabled or not self.waypoints:
            return

        if self.current_wp >= len(self.waypoints):
            return

        x, y, z = self.waypoints[self.current_wp]

        # if next waypoint is now blocked → replan
        if self._cell_blocked(x, y):
            self.get_logger().warn("Next waypoint blocked—replanning")
            self.plan_path(self._current_cell())
            return

        # publish the setpoint
        sp = TrajectorySetpoint()
        sp.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        sp.position     = [x, y, z]
        sp.velocity     = [0.0, 0.0, 0.0]
        sp.acceleration = [0.0, 0.0, 0.0]
        sp.jerk         = [0.0, 0.0, 0.0]

        # set yaw to face the waypoint
        if self.drone_position is not None:
            sp.yaw = math.atan2(y - self.drone_position[1], x - self.drone_position[0])
        else:
            sp.yaw = 0.0
        sp.yawspeed     = 0.0
        self.sp_pub.publish(sp)

        # advance if within switch radius
        if self.drone_position is not None:
            dist = np.linalg.norm(self.drone_position[:2] - np.array([x, y]))
            if dist < self.switch_radius:
                self.current_wp += 1
                self.get_logger().info(f"Switching to waypoint {self.current_wp}/{len(self.waypoints)}")

    def plan_path(self, start_cell):
        # run Dijkstra
        path_cells = self.compute_dijkstra(self.cost_grid, start_cell, self.goal_cell)
        if path_cells is None:
            self.get_logger().warn("No path found to goal")
            self.waypoints = []
            return

        # publish full Path for visualization
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for (r, c) in path_cells:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = self.ox + (c + 0.5)*self.res
            ps.pose.position.y = self.oy + (r + 0.5)*self.res
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        # append exact goal
        end = PoseStamped()
        end.header = path_msg.header
        end.pose.position.x = self.goal_world[0]
        end.pose.position.y = self.goal_world[1]
        end.pose.orientation.w = 1.0
        path_msg.poses.append(end)
        self.path_pub.publish(path_msg)

        # build waypoint list in world coords
        desired_z = float(self.drone_position[2]) if self.drone_position is not None else 0.0
        self.waypoints = [
            (
                self.ox + (c + 0.5)*self.res,
                self.oy + (r + 0.5)*self.res,
                desired_z
            )
            for (r, c) in path_cells
        ]
        self.current_wp = 0
        self.get_logger().info(f"Planned {len(self.waypoints)} waypoints")

    def compute_dijkstra(self, cost, start, goal):
        h, w = cost.shape
        dist = np.full((h, w), np.inf, dtype=float)
        dist[start] = 0.0
        prev = {}
        pq = [(0.0, start)]
        # include 8 motion directions
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        while pq:
            d, (r, c) = heapq.heappop(pq)
            if (r, c) == goal:
                break
            if d > dist[r, c]:
                continue
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and cost[nr, nc] < np.inf:
                    # diagonal movements cost sqrt(2)
                    step = cost[nr, nc] * (math.sqrt(2) if abs(dr) == 1 and abs(dc) == 1 else 1.0)
                    nd = d + step
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        prev[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (nd, (nr, nc)))

        if dist[goal] == np.inf:
            return None

        path = [goal]
        cur = goal
        while cur != start:
            cur = prev[cur]
            path.append(cur)
        return list(reversed(path))

    def _start_cell(self):
        h, w = self.cost_grid.shape
        if self.drone_position is None:
            return (h//2, 0)
        x, y = self.drone_position[:2]
        c = int((x - self.ox)/self.res)
        r = int((y - self.oy)/self.res)
        return (
            max(0, min(h-1, r)),
            max(0, min(w-1, c))
        )

    def _cell_blocked(self, x, y):
        r = int((y - self.oy)/self.res)
        c = int((x - self.ox)/self.res)
        h, w = self.cost_grid.shape
        return not (0 <= r < h and 0 <= c < w) or self.cost_grid[r, c] == np.inf

    def _current_cell(self):
        return self._start_cell()

def main():
    rclpy.init()
    node = DijkstraPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
