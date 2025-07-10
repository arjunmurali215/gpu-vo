import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge
import numpy as np
import cupy as cp
import cv2 as cv
from collections import deque
from . import utils

fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157
intrinsic_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0,  0,  1]], dtype=np.float32)

class Triangulate(Node):
    def __init__(self):
        super().__init__('triangulate')
        self.bridge = CvBridge()
        self.pose = np.eye(4, dtype=np.float32)
        self.trajectory = []

        self.depth_buf = {}
        self.old_buf = {}
        self.new_buf = {}
        
        self.trans_pub = self.create_publisher(TransformStamped, '/trans', 10)

        # ROS 2 Subscriptions
        self.create_subscription(Image, '/depthmap', self.depth_cb, 50)
        self.create_subscription(PointCloud2, '/pts_2d/old', self.old_cb, 50)
        self.create_subscription(PointCloud2, '/pts_2d/new', self.new_cb, 50)


    def depth_cb(self, msg):
        fid = msg.header.frame_id
        self.depth_buf[fid] = msg
        self.try_process(fid)

    def old_cb(self, msg):
        fid = msg.header.frame_id
        self.old_buf[fid] = msg
        self.try_process(fid)

    def new_cb(self, msg):
        fid = msg.header.frame_id
        self.new_buf[fid] = msg
        self.try_process(fid)

    def try_process(self, fid):
        if fid in self.depth_buf and fid in self.old_buf and fid in self.new_buf:
            depth_msg = self.depth_buf.pop(fid)
            old_msg = self.old_buf.pop(fid)
            new_msg = self.new_buf.pop(fid)

            self.image_callback(depth_msg, old_msg, new_msg)

    def image_callback(self, depth_msg, old_msg, new_msg):
        try:
            depth = cp.array(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough'))
            good_old = utils.pointcloud2_to_cupy(old_msg)
            good_new = utils.pointcloud2_to_cupy(new_msg)

            pcl, tracked_old, tracked_new = utils.triangulate(depth, good_old, good_new)

            success, rvec, tvec, inliers = cv.solvePnPRansac(
                cp.asnumpy(pcl),
                cp.asnumpy(tracked_new),
                intrinsic_matrix,
                None
            )

            if not success:
                self.get_logger().warn("solvePnPRansac failed.")
                return

            # Compute rotation matrix from rotation vector
            R_mat, _ = cv.Rodrigues(rvec)

            # Convert rotation matrix to quaternion
            quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]

            # Create TransformStamped message
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "world"
            transform.child_frame_id = "camera"

            transform.transform.translation.x = float(tvec[0])
            transform.transform.translation.y = float(tvec[1])
            transform.transform.translation.z = float(tvec[2])

            transform.transform.rotation.x = float(quat[0])
            transform.transform.rotation.y = float(quat[1])
            transform.transform.rotation.z = float(quat[2])
            transform.transform.rotation.w = float(quat[3])

            self.trans_pub.publish(transform)

        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Triangulate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()
        plt.close()


if __name__ == '__main__':
    main()
