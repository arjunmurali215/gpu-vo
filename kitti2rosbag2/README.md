`kitti2rosbag2` is designed to convert the KITTI Odometry dataset to ROS2 bag format, emphasizing manual control over message publishing and bag recording.

#### Note
* This is a modification of https://github.com/bharadwajsirigadi/kitti2rosbag2

## Usage

#### 1. Parameters Input

Open  [params file](./config/params.yaml) </br>

Update following tags.

```yaml
kitti_pub:
  ros__parameters:
    sequence: <sequence_no>  #Integer
    data_dir: '<dataset_dir>'
    odom_dir: '<data_odometry_poses_dir>'
    bag_dir : '<bag_dir>/<bag_name>'
    odom : [True/False] 
```

Example: Converts kitti dataset to rosbag2.

```yaml
kitti_rec:
  ros__parameters:
    sequence: 0
    data_dir: '/home/user_name/Download/data_odometry_gray/dataset/'
    odom_dir: '/home/user_name/Download/data_odometry_poses/dataset/' 
    bag_dir : '/home/user_name/Download/00_bag'
    odom : True
```

#### 2. Building Package

```python
cd ~/ros2_ws
colcon build --symlink-install
```

#### 3. Converting to bag

`
ros2 launch kitti2rosbag2 kitti2rosbag2.launch
`
