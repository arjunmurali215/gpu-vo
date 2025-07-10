import cv2 as cv
import cupy as cp
import numpy as np
import torch, pypose as pp
import rclpy
import ctypes
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Initialize all GPU Mats used in the pipeline
def initializeImages():
    l_img1 = cv.cuda.GpuMat()
    r_img1 = cv.cuda.GpuMat()
    l_img2 = cv.cuda.GpuMat()
    gpu_pts_l1 = cv.cuda.GpuMat()
    gpu_pts_l2 = cv.cuda.GpuMat()
    gpu_pts_l1r = cv.cuda.GpuMat()
    return l_img1, r_img1, l_img2, gpu_pts_l1, gpu_pts_l2, gpu_pts_l1r

# Initialize CUDA feature detector and KLT optical flow
def initializeFeatureTracking():
    maxCorners = 7548
    qualityLevel = 0.0045
    minDistance = 49
    blockSize = 1
    winSize = 18
    maxLevel = 2
    gftt = cv.cuda.createGoodFeaturesToTrackDetector(srcType=cv.CV_8UC1, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize, useHarrisDetector=False, harrisK=0.04)
    klt = cv.cuda.SparsePyrLKOpticalFlow.create(winSize=(winSize,winSize), maxLevel=maxLevel)
    stream = cv.cuda.Stream()
    return gftt, klt, stream

# Create CUDA stereo disparity algorithm (StereoSGM)
def initializeDisparity():
    disp = cv.cuda.createStereoSGM(minDisparity=0, numDisparities=64, uniquenessRatio=0)
    return disp

# Detect and track features using forward-backward KLT and consistency check
def getFeaturesandTrack(gftt, klt, l_img1, l_img2, gpu_pts_l1, gpu_pts_l2, gpu_pts_l1r, stream):
    keypoints_l = gftt.detect(l_img1, stream=stream).download()
    pts_l1 = cp.array(keypoints_l.reshape(-1, 2), dtype=cp.float32)

    # Upload keypoints to GPU
    gpu_pts_l1.upload(cp.asnumpy(pts_l1.reshape(1, -1, 2))) 

    # Forward KLT tracking
    gpu_pts_l2, status_fwd, _ = klt.calc(l_img1, l_img2, gpu_pts_l1, None, stream=stream)

    # Backward tracking to original frame
    gpu_pts_l1r, status_bwd, _ = klt.calc(l_img2, l_img1, gpu_pts_l2, None, stream=stream)
    stream.waitForCompletion()

    # Download and compute forward-backward error
    pts_l1r = cp.asarray(gpu_pts_l1r.download()).reshape(-1, 2)
    status_fwd = cp.asarray(status_fwd.download()).flatten()
    status_bwd = cp.asarray(status_bwd.download()).flatten()
    fb_error = cp.linalg.norm(pts_l1r - pts_l1.reshape(-1, 2), axis=1)

    # Keep only consistent and valid tracks
    valid = (status_fwd == 1) & (status_bwd == 1) & (fb_error < 1.0)
    good_old = pts_l1[valid]
    good_new = cp.asarray(gpu_pts_l2.download()).reshape(-1, 2)[valid]

    return good_old, good_new

# Compute disparity and corresponding depth map
def getDisparity(disp, l_img1, r_img1, fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, b=0.537):
    disp_map = disp.compute(l_img1, r_img1).download().astype(cp.float32)/16
    disp_map[disp_map <= 0] = cp.nan  # Mask invalid disparities
    depth = fx * b / disp_map  # Convert disparity to depth using stereo formula

    return disp_map, depth

# Triangulate 3D points from depth and 2D tracked keypoints
def triangulate(depth, good_old, good_new, fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, b=0.537, max_depth=4000):
    u = good_old[:, 0].astype(cp.int32)
    v = good_old[:, 1].astype(cp.int32)
    z = depth[v, u]  # Sample depth at 2D keypoint locations
    z = cp.asarray(z)
    valid_mask = ~cp.isnan(z) & (z > 0) & (z < max_depth)

    # Filter valid 3D points and reproject
    z = z[valid_mask]
    good_new = good_new[valid_mask]
    good_old = good_old[valid_mask]
    x = (good_old[:, 0] - cx) * z / fx
    y = (good_old[:, 1] - cy) * z / fy
    pcl = cp.vstack((x, y, z)).T  # Shape: Nx3

    return pcl, good_old, good_new

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
def pointcloud2_to_cupy(msg: PointCloud2) -> cp.ndarray:
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
    coords = np.vstack([np_array['x'], np_array['y']]).T  # Shape: Nx2

    # Convert to CuPy array
    cp_array = cp.asarray(coords, dtype=cp.float32)

    return cp_array
