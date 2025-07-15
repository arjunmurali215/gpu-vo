import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import torch, theseus as th
import cupy as cp
import cv2 as cv
from collections import deque
from . import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_default_device(device)

fx, fy, cx, cy = 718.856, 718.856, 607.1928, 185.2157

intrinsics = th.Variable(torch.tensor([[[fx, 0, cx],[0, fy, cy],[0,  0,  1]]], dtype=torch.float32), name='intrinsics')

class BundleAdjustment(Node):
    def __init__(self):
        super().__init__('bundleAdjustment')
        # self.pose = np.eye(4, dtype=np.float32)
        # self.trajectory = []

        self.trans_buf = {}
        self.pcl_buf = {}
        self.old_buf = {}
        self.new_buf = {}
        
        self.trans_pub = self.create_publisher(TransformStamped, '/optim_T', 10)

        # ROS 2 Subscriptions
        self.create_subscription(TransformStamped, '/trans', self.trans_cb, 50)
        self.create_subscription(PointCloud2, '/pts_3d', self.pcl_cb, 50)
        self.create_subscription(PointCloud2, '/pts_2d/prev', self.old_cb, 50)
        self.create_subscription(PointCloud2, '/pts_2d/next', self.new_cb, 50)

    def trans_cb(self, msg):
        fid = int(msg.header.frame_id)
        self.trans_buf[fid] = msg
        self.try_process(fid)

    def pcl_cb(self, msg):
        fid = int(msg.header.frame_id)
        self.pcl_buf[fid] = msg
        self.try_process(fid)

    def old_cb(self, msg):
        fid = int(msg.header.frame_id)
        self.old_buf[fid] = msg
        self.try_process(fid)

    def new_cb(self, msg):
        fid = int(msg.header.frame_id)
        self.new_buf[fid] = msg
        self.try_process(fid)

    def try_process(self, fid):
        if fid>1 and fid-1 in self.trans_buf and fid-1 in self.pcl_buf and fid in self.pcl_buf and fid in self.old_buf and fid-1 in self.new_buf:
            trans_msg = self.trans_buf.pop(fid-1)
            old_pcl_msg = self.pcl_buf.pop(fid-1)
            new_pcl_msg = self.pcl_buf[fid]
            new_msg = self.old_buf[fid]
            old_msg = self.new_buf.pop(fid-1)

            self.image_callback(trans_msg, old_pcl_msg, new_pcl_msg, old_msg, new_msg)

    def image_callback(self, trans_msg, old_pcl_msg, new_pcl_msg, old_msg, new_msg):
        old_pcl = utils.pointcloud2_to_cupy(old_pcl_msg, is3D=True)
        new_pcl = utils.pointcloud2_to_cupy(new_pcl_msg, is3D=True)

        old_pts = utils.pointcloud2_to_cupy(old_msg)
        new_pts = utils.pointcloud2_to_cupy(new_msg)

        r, t = trans_msg.transform.rotation, trans_msg.transform.translation
        pose = th.SE3(x_y_z_quaternion = torch.tensor([t.x, t.y, t.z, r.x, r.y, r.z, r.w], dtype=torch.float32, device=device).unsqueeze(0), name="pose")


        tolerance = 10  # max allowed distance

        # Step 1: Compute pairwise distances (squared for speed)
        diff = old_pts[:, None, :] - new_pts[None, :, :]
        dists_sq = cp.sum(diff**2, axis=2)                

        # Step 2: Find closest new point for each old point
        min_dists_sq = cp.min(dists_sq, axis=1)           
        min_indices = cp.argmin(dists_sq, axis=1)         

        # Step 3: Filter by distance threshold
        valid_mask = min_dists_sq < (tolerance ** 2)     

        # Step 4: Get final matched index pairs
        i_old = cp.arange(old_pts.shape[0])[valid_mask]
        i_new = min_indices[valid_mask]

        # Result: (K, 2) array of (i_old, i_new)
        matches = cp.stack((i_old, i_new), axis=1)
        old_indices = matches[:, 0]
        new_indices = matches[:, 1]

        old_pcl = torch.tensor(old_pcl[old_indices], dtype=torch.float32, device=device).unsqueeze(0)
        new_pcl = torch.tensor(new_pcl[new_indices], dtype=torch.float32, device=device).unsqueeze(0)

        old_pcl = th.Variable(old_pcl, name="old_pcl")
        new_pcl = th.Variable(new_pcl, name="new_pcl")

        # self.get_logger().info(f"{old_pcl.shape[0], new_pcl.shape[0], (old_pts[old_indices] - new_pts[new_indices]).mean().item(), (old_pcl[old_indices] - new_pcl[new_indices]).mean().item()}")
        optim_vars = [pose]
        aux_vars = old_pcl, new_pcl
        cost_function = th.AutoDiffCostFunction(optim_vars, utils.compute_error, old_pcl.shape[1]*old_pcl.shape[2], aux_vars=aux_vars, name="reprojerr")

        objective = th.Objective().to(device)
        objective.add(cost_function)
        optimizer = th.LevenbergMarquardt(objective, max_iterations=100, damping=1e-2, diagonal_damping=True,)
        theseus_optim = th.TheseusLayer(optimizer)
        theseus_inputs = {"pose": pose, "obj_pcl": old_pcl, "new_pcl": new_pcl}

        with torch.no_grad():
            updated_inputs, info = theseus_optim.forward(theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":False})
        self.get_logger().info(f"Frame: {old_pcl_msg.header.frame_id}, npoints: {old_pcl.shape[1]}, {(info.best_solution['pose'][0]-pose[0]).mean().item()}")

        # Publish the optimized transform
        T = info.best_solution['pose']
        tvec = pose.translation().tensor.squeeze(0).cpu().numpy()
        R_mat = pose.rotation().tensor.squeeze(0).cpu().numpy()

        # Convert rotation matrix to quaternion
        quat = R.from_matrix(R_mat).as_quat()  # [x, y, z, w]

        # Create TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = old_msg.header.frame_id
        transform.child_frame_id = new_msg.header.frame_id

        transform.transform.translation.x = float(tvec[0])
        transform.transform.translation.y = float(tvec[1])
        transform.transform.translation.z = float(tvec[2])

        transform.transform.rotation.x = float(quat[0])
        transform.transform.rotation.y = float(quat[1])
        transform.transform.rotation.z = float(quat[2])
        transform.transform.rotation.w = float(quat[3])

        self.trans_pub.publish(transform)

        # except Exception as e:
        #     self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = BundleAdjustment()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()