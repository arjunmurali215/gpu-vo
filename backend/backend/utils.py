import cv2 as cv
import cupy as cp
import numpy as np
import rclpy
import ctypes
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import torch, theseus as th

def compute_error(optim_vars, aux_vars):
    pose = optim_vars[0]
    pcl1, pcl2 = aux_vars
    R, t = pose.rotation().tensor.squeeze(0), pose.translation().tensor.squeeze(0)
    error = (pcl2.tensor.squeeze(0) - ((R @ pcl1.tensor.squeeze(0).T).T + t))
    error = error.reshape(1, pcl1.shape[1]*pcl2.shape[2])
    return error


# Convert a CuPy point array to ROS PointCloud2 message
def create_pointcloud2(points_cp: cp.ndarray, frame_id="map") -> PointCloud2:
    """
    Convert an Nx2 or Nx3 CuPy array to a PointCloud2 ROS message.
    If input is Nx2, pads Z = 0.
    """
    if not isinstance(points_cp, cp.ndarray):
        raise TypeError("Input must be a CuPy array")

    if points_cp.ndim != 2 or points_cp.shape[1] not in (2, 3):
        raise ValueError("Input must be a Nx2 or Nx3 CuPy array")

    # Pad with Z=0 if 2D
    if points_cp.shape[1] == 2:
        z = cp.zeros((points_cp.shape[0], 1), dtype=cp.float32)
        points_cp = cp.hstack((points_cp, z))  # Convert to Nx3

    # Convert to raw bytes
    data_bytes = cp.asnumpy(points_cp.astype(cp.float32)).tobytes()

    # Define point fields for x, y, z
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]

    # Create ROS header
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    # Construct and return PointCloud2 message
    msg = PointCloud2(
        header=header,
        height=1,
        width=points_cp.shape[0],
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=12,
        row_step=12 * points_cp.shape[0],
        data=data_bytes
    )
    return msg

# Convert PointCloud2 ROS message to CuPy array
def pointcloud2_to_cupy(msg: PointCloud2, is3D = False) -> cp.ndarray:
    dtype_map = {
        PointField.FLOAT32: ('f', 4),
        PointField.FLOAT64: ('d', 8),
        PointField.UINT32: ('I', 4),
        PointField.INT32:  ('i', 4),
        PointField.UINT16: ('H', 2),
        PointField.INT16:  ('h', 2),
        PointField.UINT8:  ('B', 1),
        PointField.INT8:   ('b', 1),
    }

    # Build numpy dtype based on PointField
    offset = 0
    np_dtype = []
    for field in msg.fields:
        if field.datatype not in dtype_map:
            raise TypeError(f"Unsupported PointField datatype: {field.datatype}")
        name = field.name
        type_char, size = dtype_map[field.datatype]
        np_dtype.append((name, type_char))
        offset += size

    # Parse binary data into structured numpy array
    np_array = np.frombuffer(msg.data, dtype=np_dtype, count=msg.width * msg.height)

    # Extract x and y fields only
    if is3D:
        coords = np.vstack([np_array['x'], np_array['y'], np_array['z']]).T  # Shape: Nx3
    else:
        coords = np.vstack([np_array['x'], np_array['y']]).T  # Shape: Nx2

    # Convert to CuPy array
    cp_array = cp.asarray(coords, dtype=cp.float32)

    return cp_array
