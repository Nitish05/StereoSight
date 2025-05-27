# Real-Time Obstacle Avoidance and Tracking

This project implements real-time obstacle avoidance and tracking for drones using the ModalAI VOXL 2 platform. The system leverages depth data obtained from a calibrated stereo camera, processed through the VOXL-DFS server, and converted into ROS 2 nodes using the `voxl_mpa_to_ros2` package. The provided scripts enable the drone to navigate while avoiding obstacles by generating occupancy grids and planning paths.

## Dependencies

To run this project, you need the following dependenciess:

1. **ROS 2**: Install ROS 2 (e.g., Humble or Foxy) on your system. Follow the [official installation guide](https://docs.ros.org/en/rolling/Installation.html).
2. **Python Packages**:
   - `numpy`
   - `opencv-python`
   - `scipy`
3. **VOXL 2 and VOXL-DFS**:
   - Ensure the VOXL 2 platform is set up and the stereo camera is calibrated.
   - Install the `voxl_mpa_to_ros2` package to bridge VOXL data to ROS 2.

## Setup Instructions

1. Inside the drone, navigate to the `voxl_mpa_to_ros2` folder:
   ```bash
   cd ~/colcon_ws/src/px4_ros_ws/src/px4_ros_com/src/examples
   ```

2. Create a directory for the obstacle avoidance scripts:
   ```bash
   mkdir obstacle_avoidance
   ```

3. Copy the following scripts into the `obstacle_avoidance` directory:
   - `planner.py`
   - `matrix.py`
   - `rotated.py`
   - `map_publisher.py`

   Example command:
   ```bash
   cp /path/to/scripts/*.py ~/colcon_ws/src/px4_ros_ws/src/px4_ros_com/src/examples/obstacle_avoidance/
   ```

4. Update the `CMakeLists.txt` file in `px4_ros_ws/src/px4_ros_com` to include the new scripts. Add the following lines:
   ```cmake
   install(PROGRAMS
     src/examples/obstacle_avoidance/map_publisher.py
     src/examples/obstacle_avoidance/matrix.py
     src/examples/obstacle_avoidance/rotated.py
     src/examples/obstacle_avoidance/planner.py
   DESTINATION lib/${PROJECT_NAME}
   )
   ```

5. Build the workspace:
   ```bash
   cd ~/colcon_ws
   colcon build
   source install/setup.bash
   ```

## Running the Scripts

### 1. Start the Map Publisher
The `map_publisher.py` script generates occupancy grids from the point cloud data and publishes them as ROS 2 messages.

Run the following command:
```bash
ros2 run px4_ros_com map_publisher.py
```

### 2. Start the Path Planner
The `planner.py` script subscribes to the occupancy grid and computes a path using Dijkstra's algorithm.

Run the following command:
```bash
ros2 run px4_ros_com planner.py
```

### 3. Visualize the Results
You can visualize the occupancy grid and planned path using tools like `rviz2`:
```bash
rviz2
```

Add the following topics to visualize:
- `/occupancy_grid` (OccupancyGrid)
- `/planned_path` (Path)

## Notes

- Ensure the VOXL 2 platform is running and publishing point cloud data to the `/voa_pc_out` topic.
- The stereo camera must be calibrated, and the depth data should be accurate for proper obstacle avoidance.
- Adjust parameters like grid resolution and thresholds in the scripts as needed for your specific environment.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.