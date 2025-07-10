import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R

class TrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('rerun_trajectory_visualizer')
        self.subscription = self.create_subscription(
            TransformStamped,
            '/trans',
            self.transform_cb,
            50
        )

        # Initialize rerun
        rr.init("trajectory_visualizer", spawn=True)

        # Static marker for world origin
        rr.log("origin", rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0, 0, 1]], colors=[[255, 0, 0]]))

        self.pose = np.eye(4, dtype=np.float32)
        self.trajectory = []

    def transform_cb(self, msg: TransformStamped):
        # Extract translation
        t = msg.transform.translation
        translation = np.array([t.x, t.y, t.z], dtype=np.float32)

        # Extract quaternion rotation and build SE(3) transform
        r = msg.transform.rotation
        rotation = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = rotation
        T[:3, 3] = translation

        # Accumulate pose
        self.pose = self.pose @ T
        pos = self.pose[:3, 3]
        self.trajectory.append(pos.copy())

        # Log trajectory as a polyline
        rr.log("trajectory/path", rr.LineStrips3D([np.array(self.trajectory)]))

        # Optionally log camera orientation (omit if causing confusion)
        # rr.log("trajectory/pose", rr.Transform3D(translation=pos, rotation=rr.components.RotationMatrix(matrix=self.pose[:3, :3])))


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
