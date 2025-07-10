import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from message_filters import Subscriber, ApproximateTimeSynchronizer
from . import utils
from collections import deque

l_img1, r_img1, l_img2, gpu_pts_l1, gpu_pts_l2, gpu_pts_l1r = utils.initializeImages()
disp = utils.initializeDisparity()


class Disparity(Node):
    def __init__(self):
        super().__init__('dispmap')
        self.bridge = CvBridge()

        self.disp_pub = self.create_publisher(Image, '/dispmap', 10)
        self.depth_pub = self.create_publisher(Image, '/depthmap', 10)
        
        # Create separate subscribers for left and right topics
        self.left_sub = Subscriber(self, Image, '/camera_left/image_raw')
        self.right_sub = Subscriber(self, Image, '/camera_right/image_raw')

        # ApproximateTimeSynchronizer(queue_size, slop) — slop in seconds
        self.ts = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

    def image_callback(self, left_msg, right_msg):
        try:
            left_img = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
            right_img = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='passthrough')
            if(left_msg.header.frame_id == right_msg.header.frame_id):
                l_img1.upload(left_img)
                r_img1.upload(right_img)
                dispmap, depthmap = utils.getDisparity(disp, l_img1, r_img1)

                disp_msg = self.bridge.cv2_to_imgmsg(dispmap, encoding='32FC1')
                disp_msg.header.stamp = self.get_clock().now().to_msg()
                disp_msg.header.frame_id = str(left_msg.header.frame_id)
                self.disp_pub.publish(disp_msg)
                
                depth_msg = self.bridge.cv2_to_imgmsg(depthmap, encoding='32FC1')
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = str(left_msg.header.frame_id)
                self.depth_pub.publish(depth_msg)

            else:
                self.get_logger().warn("Frame IDs do not match between left and right images.")

        except Exception as e:
            self.get_logger().error(f"Error converting images: {e}")


def main(args=None):
    rclpy.init(args=args)
    disp_node = Disparity()
    try:
        rclpy.spin(disp_node)
    except KeyboardInterrupt:
        pass
    finally:
        disp_node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()
