#!/usr/bin/env python3

import rclpy
import os
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from kitti2rosbag2.utils.kitti_utils import KITTIOdometryDataset
from kitti2rosbag2.utils.quaternion import Quaternion
import rosbag2_py


class KittiOdom(Node):
    def __init__(self):
        super().__init__("kitti_rec")
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('sequence', 0),
                ('data_dir', ''),
                ('odom', True),
                ('odom_dir', ''),
                ('bag_dir', '')
            ]
        )

        # Get parameters
        sequence = self.get_parameter('sequence').value
        data_dir = self.get_parameter('data_dir').value
        odom_enabled = self.get_parameter('odom').value
        odom_dir = self.get_parameter('odom_dir').value if odom_enabled else None
        bag_dir = self.get_parameter('bag_dir').value

        # Load KITTI data
        self.kitti_dataset = KITTIOdometryDataset(data_dir, sequence, odom_dir)
        self.left_imgs = self.kitti_dataset.left_images()
        self.right_imgs = self.kitti_dataset.right_images()
        self.times_file = self.kitti_dataset.times_file()
        self.counter_limit = len(self.left_imgs) - 1
        self.counter = 0
        self.odom_enabled = odom_enabled
        self.p_msg = Path()

        if odom_enabled:
            try:
                self.ground_truth = self.kitti_dataset.odom_pose()
                self.get_logger().info(f"Loaded {len(self.ground_truth)} odometry poses.")
            except FileNotFoundError as e:
                self.get_logger().error(f"Odometry file not found: {e}")
                rclpy.shutdown()
                return

        # ROS 2 bag setup
        self.writer = rosbag2_py.SequentialWriter()
        if os.path.exists(bag_dir):
            self.get_logger().info(f"The directory {bag_dir} already exists. Shutting down...")
            rclpy.shutdown()
            return

        storage_options = rosbag2_py._storage.StorageOptions(uri=bag_dir, storage_id='sqlite3')
        converter_options = rosbag2_py._storage.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        # Register topics
        self.writer.create_topic(rosbag2_py._storage.TopicMetadata(
            name='/camera_left/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr'))
        self.writer.create_topic(rosbag2_py._storage.TopicMetadata(
            name='/camera_right/image_raw', type='sensor_msgs/msg/Image', serialization_format='cdr'))
        self.writer.create_topic(rosbag2_py._storage.TopicMetadata(
            name='/camera_left/camera_info', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr'))
        self.writer.create_topic(rosbag2_py._storage.TopicMetadata(
            name='/camera_right/camera_info', type='sensor_msgs/msg/CameraInfo', serialization_format='cdr'))
        if odom_enabled:
            self.writer.create_topic(rosbag2_py._storage.TopicMetadata(
                name='/odom', type='nav_msgs/msg/Odometry', serialization_format='cdr'))
            self.writer.create_topic(rosbag2_py._storage.TopicMetadata(
                name='/odom_path', type='nav_msgs/msg/Path', serialization_format='cdr'))

        # Start timer
        self.timer = self.create_timer(0.05, self.rec_callback)  # 20 FPS

    def rec_callback(self):
        time = self.times_file[self.counter]
        timestamp_ns = int(time * 1e9)
        stamp_msg = rclpy.time.Time(nanoseconds=timestamp_ns).to_msg()
        frame_id = str(self.counter)

        # Load and write left image
        left_image = cv2.imread(self.left_imgs[self.counter], cv2.IMREAD_GRAYSCALE)
        left_msg = self.bridge.cv2_to_imgmsg(left_image, encoding='mono8')
        left_msg.header.stamp = stamp_msg
        left_msg.header.frame_id = frame_id
        self.writer.write('/camera_left/image_raw', serialize_message(left_msg), timestamp_ns)

        # Load and write right image
        right_image = cv2.imread(self.right_imgs[self.counter], cv2.IMREAD_GRAYSCALE)
        right_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='mono8')
        right_msg.header.stamp = stamp_msg
        right_msg.header.frame_id = frame_id
        self.writer.write('/camera_right/image_raw', serialize_message(right_msg), timestamp_ns)

        # CameraInfo messages
        self.write_camera_info(self.kitti_dataset.projection_matrix(0), '/camera_left/camera_info', frame_id, timestamp_ns, stamp_msg)
        self.write_camera_info(self.kitti_dataset.projection_matrix(1), '/camera_right/camera_info', frame_id, timestamp_ns, stamp_msg)

        # Odometry
        if self.odom_enabled:
            T = self.ground_truth[self.counter]
            translation = T[:3, 3]
            quaternion = Quaternion().rotationmtx_to_quaternion(T[:3, :3])
            self.write_odom_and_path(translation, quaternion, timestamp_ns, stamp_msg, frame_id)

        self.get_logger().info(f"{self.counter+1}/{self.counter_limit+1} - Images processed")

        if self.counter >= self.counter_limit:
            self.get_logger().info("All images and poses published. Stopping...")
            self.timer.cancel()
            rclpy.shutdown()
        self.counter += 1

    def write_camera_info(self, mtx, topic, frame_id, timestamp_ns, stamp_msg):
        cam_info = CameraInfo()
        cam_info.header.stamp = stamp_msg
        cam_info.header.frame_id = frame_id
        cam_info.p = mtx.flatten().tolist()
        self.writer.write(topic, serialize_message(cam_info), timestamp_ns)

    def write_odom_and_path(self, trans, quat, timestamp_ns, stamp_msg, frame_id):
        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp_msg
        odom.header.frame_id = frame_id
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = -trans[0]
        odom.pose.pose.position.y = -trans[2]
        odom.pose.pose.position.z = -trans[1]
        odom.pose.pose.orientation.x = quat[0]
        odom.pose.pose.orientation.y = quat[1]
        odom.pose.pose.orientation.z = quat[2]
        odom.pose.pose.orientation.w = quat[3]
        self.writer.write('/odom', serialize_message(odom), timestamp_ns)

        # Path
        pose = PoseStamped()
        pose.header.stamp = stamp_msg
        pose.header.frame_id = frame_id
        pose.pose = odom.pose.pose
        self.p_msg.poses.append(pose)
        self.p_msg.header.stamp = stamp_msg
        self.p_msg.header.frame_id = frame_id
        self.writer.write('/odom_path', serialize_message(self.p_msg), timestamp_ns)


def main(args=None):
    rclpy.init(args=args)
    node = KittiOdom()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
