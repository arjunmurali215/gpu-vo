# gpu-vo
Stereo Visual Odometry in Python using CUDA and ROS2 Humble

## Dependencies:
ROS2 Humble
openCV4 built with CUDA support
numpy==1.26.4
cupy
rerun-sdk==0.21.0

## Setup:
#### 1. Clone the repo:

```bash
mkdir ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/arjunmurali215/gpu-vo.git
```

#### 2. Create the rosbag2 for KITTI Dataset
Refer to [kitti2rosbag2 readme](./kitti2rosbag2/README.md)

#### 3. Build packages

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

#### 4. Launch pipeline
```bash
ros2 launch frontend full-pipeline.launch.py
```

#### 5. Play bag
In a separate terminal:
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
ros2 bag play <path_to_rosbag2>
```


## To-Do
1. Add backend package with bundle adjustment node
2. Add loop closure node