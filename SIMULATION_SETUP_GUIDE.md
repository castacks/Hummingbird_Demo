# Hummingbird Demo PC Simulation Setup Guide

## Overview
The Hummingbird Demo is an autonomy system for VTOL drones that includes wire detection, tracking, and servoing capabilities. This guide covers running the full system in simulation on PC using Docker containers.

## Architecture Overview
The system consists of several key components:
- **ArduPilot SITL**: Software-in-the-loop simulation for drone dynamics
- **Autonomy Container**: ROS2 nodes for wire detection, tracking, and control
- **MAVROS**: Interface between ROS and ArduPilot
- **VINS**: Visual-inertial navigation system
- **ZED Camera Simulation**: Simulated stereo camera data

## Prerequisites
- Docker and Docker Compose installed
- X11 forwarding configured (for GUI applications)
- NVIDIA Docker runtime (if using GPU acceleration)

## File Structure
```
Hummingbird_Demo/
├── docker-compose.yaml              # Main orchestration file
├── docker/
│   ├── Dockerfile.pc               # PC-specific autonomy container
│   └── autonomy_entrypoint.sh      # Autonomy container startup script
├── ardupilot_SITL/
│   ├── docker-compose.yaml        # ArduPilot SITL container
│   └── sitl_entrypoint.sh         # SITL startup script
└── autonomy/src/
    └── system_bringup/
        └── launch/
            └── bringup.launch.py   # Main launch file
```

## Environment Variables

The system behavior is controlled by environment variables. Here's what each one does:

### Core Simulation Variables
| Variable | Values | Description |
|----------|--------|-------------|
| `SIMULATION` | `0` (real hardware) / `1` (simulation) | Enables simulation mode |
| `MAVROS` | `0` (disabled) / `1` (enabled) | Enables MAVROS for drone communication |

### Autonomy Feature Controls
| Variable | Values | Description |
|----------|--------|-------------|
| `WIRE_MODE` | `0` (off) / `1` (detection) / `2` (tracking) | Wire detection/tracking mode |
| `WIRE_VIZ` | `0` (disabled) / `1` (enabled) | Enable wire visualizations |
| `RVIZ` | `0` (disabled) / `1` (enabled) | Launch RViz for visualization |
| `VO` | `0` (disabled) / `1` (enabled) | Enable Visual Odometry (VINS) |
| `SERVO` | `0` (disabled) / `1` (enabled) | Enable servoing control |
| `RECORD` | `0` (disabled) / `1` (enabled) | Enable data recording |

### Additional Variables
| Variable | Description |
|----------|-------------|
| `ROS_DOMAIN_ID` | ROS2 domain ID (automatically set to 68) |
| `DISPLAY` | X11 display for GUI applications |
| `STORAGE_PATH` | Path for data collection (if recording) |

## System Components

### 1. system_bringup Launch File
Location: `/autonomy/src/system_bringup/launch/bringup.launch.py`

This is the main launch file that orchestrates all autonomy components:

#### Nodes Launched (conditionally based on environment variables):
- **Wire Tracking Node** (if WIRE_MODE=2)
- **Wire Detection Node** (if WIRE_MODE=1 or 2)
- **RViz Visualization** (if RVIZ=1)
- **Servoing Node** (if SERVO=1, delayed start)
- **MAVROS Real/SITL** (if MAVROS=1, different configs for real vs sim)
- **MAVROS Manager** (if MAVROS=1, delayed start)
- **VINS Node** (if VO=1)
- **Rosbag Recorder** (if RECORD=1)

#### Configuration Files Used:
- `/common_utils/interface_config.yaml` - ROS topic mappings and settings
- `/vins/config/zedx/zedx_stereo_config.yaml` - VINS configuration
- RViz configs: wire_tracking.rviz or wire_detection.rviz (based on WIRE_MODE)

### 2. Interface Configuration
Location: `/autonomy/src/wire_autonomy/common_utils/config/interface_config.yaml`

Key topic mappings:
- Camera info: `/wire_cam/zed_node/left/camera_info`
- RGB image: `/wire_cam/zed_node/left/image_rect_color`
- Depth image: `/wire_cam/zed_node/depth/depth_registered`
- Drone pose: `/mavros/local_position/pose`
- Velocity commands: `/mavros/setpoint_raw/local`
- Wire detections: `/pincer/wire_detections`
- Wire target: `/pincer/wire_target`

## Step-by-Step Setup Instructions

### Step 1: Prepare Environment
```bash
cd /home/exouser/Development/Hummingbird_Demo
export DISPLAY=:0  # Adjust for your X11 setup
```

### Step 2: Start ArduPilot SITL Simulation
```bash
cd ardupilot_SITL
docker-compose up -d
```
This starts:
- Gazebo with iris quadcopter model
- ArduPilot SITL flight controller
- Listens on UDP port 14550 for MAVROS connection

### Step 3: Configure Simulation Environment Variables
```bash
# Core simulation settings
export SIMULATION=1
export MAVROS=1

# Choose your autonomy features
export WIRE_MODE=1        # 1=detection, 2=tracking
export WIRE_VIZ=1         # Enable visualizations
export RVIZ=1             # Launch RViz
export VO=1               # Enable visual odometry
export SERVO=1            # Enable servoing control

# Optional settings
export RECORD=0           # Disable recording for simulation
```

### Step 4: Build and Run PC Autonomy Container
```bash
# Build the PC container (first time only)
docker-compose build autonomy

# Run the autonomy container
docker-compose up autonomy
```

Alternatively, run directly with environment variables:
```bash
SIMULATION=1 MAVROS=1 WIRE_MODE=1 WIRE_VIZ=1 RVIZ=1 VO=1 SERVO=1 RECORD=0 \
docker-compose up autonomy
```

### Step 5: Verify System Operation
Once running, you should see:
- MAVROS connected to ArduPilot SITL
- Wire detection node processing simulated camera data
- RViz showing drone pose and wire detections
- VINS providing visual odometry

## Manual Launch (Alternative)
If you prefer to launch components manually inside the container:

```bash
# Enter the autonomy container
docker-compose exec autonomy bash

# Set environment variables
export SIMULATION=1 MAVROS=1 WIRE_MODE=1 WIRE_VIZ=1 RVIZ=1 VO=1 SERVO=1 RECORD=0

# Launch manually
ros2 launch system_bringup bringup.launch.py
```

## Testing and Control

### Control Test Scripts
The system includes test nodes with convenient aliases:

```bash
# Test velocity control
ros2 launch control_test vel_control_test.launch.xml

# Test wire-following control  
ros2 launch control_test wire_control_test.launch.xml
```

### Key ROS Topics for Monitoring
```bash
# Monitor drone state
ros2 topic echo /mavros/local_position/pose

# Monitor wire detections
ros2 topic echo /pincer/wire_detections

# Monitor velocity commands
ros2 topic echo /mavros/setpoint_raw/local

# List all active topics
ros2 topic list
```

## Troubleshooting

### Common Issues

1. **MAVROS not connecting**
   - Check ArduPilot SITL is running: `docker-compose ps`
   - Verify port 14550 is available: `netstat -an | grep 14550`
   - Check SIMULATION=1 and MAVROS=1 are set

2. **No camera data**
   - ZED wrapper needs separate camera container or simulation data
   - For pure simulation, may need to provide mock camera topics

3. **GUI applications not showing**
   - Check X11 forwarding: `echo $DISPLAY`
   - Verify X11 permissions: `xauth list`
   - Try: `xhost +local:docker`

4. **Build failures**
   - Check Docker has enough memory allocated
   - Try building with fewer parallel workers: `--parallel-workers 2`

### Debugging Commands
```bash
# Check container logs
docker-compose logs autonomy

# Monitor ROS node status
ros2 node list
ros2 node info /wire_detection_node

# Check parameter values
ros2 param list /wire_detection_node
```

## Development Notes

- The correct launch command is: `ros2 launch system_bringup bringup.launch.py` (not "sustem_bringup")
- Environment variables control which nodes are launched
- Configuration files are in `/config/` directories within each package
- The system uses ROS_DOMAIN_ID=68 by default
- All timing-sensitive nodes use delayed starts via TimerAction

## Network Configuration
- ROS_DOMAIN_ID: 68
- MAVROS SITL connection: udp://:14550@
- ArduPilot master: tcp:127.0.0.1:5760
- SITL connection: 127.0.0.1:5501

This setup provides a complete autonomy simulation environment for developing and testing wire detection and following capabilities.