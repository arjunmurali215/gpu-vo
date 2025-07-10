import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2 as cv
from message_filters import Subscriber, ApproximateTimeSynchronizer

from . import utils
from collections import deque

left_image_queue = deque(maxlen=2)

l_img1, r_img1, l_img2, gpu_pts_l1, gpu_pts_l2, gpu_pts_l1r = utils.initializeImages()
gftt, klt, stream = utils.initializeFeatureTracking()


class FeatureTrack(Node):
    def __init__(self):
        super().__init__('featuretrack')
        self.bridge = CvBridge()

        self.old_pub = self.create_publisher(PointCloud2, '/pts_2d/old', 10)
        self.new_pub = self.create_publisher(PointCloud2, '/pts_2d/new', 10)
        
        # Create separate subscribers for left and right topics
        self.left_sub = self.create_subscription(Image, '/camera_left/image_raw', self.image_callback, 10)

    def image_callback(self, left_msg):
        try:
            left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
            left_image_queue.append(left_img)
            if(len(left_image_queue)>=2):
                l_img1.upload(left_image_queue[0])
                l_img2.upload(left_image_queue[1])
                good_old, good_new = utils.getFeaturesandTrack(gftt, klt, l_img1, l_img2, gpu_pts_l1, gpu_pts_l2, gpu_pts_l1r, stream)
                self.old_pub.publish(utils.create_pointcloud2(good_old, left_msg.header.frame_id))
                self.new_pub.publish(utils.create_pointcloud2(good_new, left_msg.header.frame_id))
                
        except Exception as e:
            self.get_logger().error(f"Error converting images: {e}")


def main(args=None):
    rclpy.init(args=args)
    feature_node = FeatureTrack()
    try:
        rclpy.spin(feature_node)
    except KeyboardInterrupt:
        pass
    finally:
        feature_node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
