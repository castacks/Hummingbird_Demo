import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Transform, Pose, Quaternion

def transform_to_homogeneous(transform):
    """
    Converts a Transform message into a 4x4 homogeneous transformation matrix.

    Args:
        Transform (Transform): The ROS Transform message to convert.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    assert isinstance(transform, Transform) , "Input must be a Transform message, got %s" % type(Transform)
    # Extract translation components
    translation = transform.translation
    translation_vector = np.array([translation.x, translation.y, translation.z])

    # Extract rotation components
    rotation = transform.rotation
    quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
    rot_matrix = Rotation.from_quat(quaternion).as_matrix()

    # Combine into a homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rot_matrix[:3, :3]
    homogeneous_matrix[:3, 3] = translation_vector
    return homogeneous_matrix

def pose_to_homogeneous(pose):
    """
    Converts a Pose message into a 4x4 homogeneous transformation matrix.

    Args:
        Pose (Pose): The ROS Pose message to convert.

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    assert isinstance(pose, Pose) , "Input must be a Pose message, got %s" % type(Pose)
    # Extract translation components
    translation = pose.position
    translation_vector = np.array([translation.x, translation.y, translation.z])

    # Extract rotation components
    rotation = pose.orientation
    quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
    rot_matrix = Rotation.from_quat(quaternion).as_matrix()

    # Combine into a homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rot_matrix[:3, :3]
    homogeneous_matrix[:3, 3] = translation_vector
    return homogeneous_matrix

def homogeneous_to_pose(homogeneous_matrix):
    """
    Converts a 4x4 homogeneous transformation matrix into a Pose message.

    Args:
        homogeneous_matrix (np.ndarray): The 4x4 homogeneous transformation matrix to convert.

    Returns:
        Pose: A ROS Pose message.
    """
    assert isinstance(homogeneous_matrix, np.ndarray) and homogeneous_matrix.shape == (4, 4), "Input must be a 4x4 numpy array, got %s" % type(homogeneous_matrix)
    # Extract translation components
    translation_vector = homogeneous_matrix[:3, 3]
    translation = translation_vector

    # Extract rotation components
    rot_matrix = homogeneous_matrix[:3, :3]
    quaternion = Rotation.from_matrix(rot_matrix).as_quat()
    rotation = quaternion

    # Combine into a Pose message
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = translation
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = rotation
    return pose
    
def image_to_world_tf(points, depth, tf_camera_to_world, camera_vector):
        x_c, y_c, z_c = image_to_camera(points, depth.reshape(-1, 1), camera_vector)

        H_cam_to_world = transform_to_homogeneous(tf_camera_to_world)

        # Convert the point to a numpy array
        point_vec = np.hstack((x_c, y_c, z_c, np.ones_like(x_c)))

        # Apply the transform: Rotate then translate
        with np.errstate(invalid='ignore'):
            try:
                world_points = H_cam_to_world @ point_vec.T
            except Exception as e:
                return None
        return world_points[:3].T

def image_to_world_pose(points, depth, pose_in_world, camera_vector):
    x_c, y_c, z_c = image_to_camera(points, depth.reshape(-1, 1), camera_vector)

    # transforming from camera frame to world convention
    x_c_w = z_c
    y_c_w = -x_c
    z_c_w = -y_c

    H_cam_to_world = pose_to_homogeneous(pose_in_world)

    # Convert the point to a numpy array
    # point_vec = np.hstack((x_c, y_c, z_c, np.ones_like(x_c)))
    point_vec = np.hstack((x_c_w, y_c_w, z_c_w, np.ones_like(x_c)))

    # Apply the transform: Rotate then translate
    with np.errstate(invalid='ignore'):
        try:
            world_points = H_cam_to_world @ point_vec.T
        except Exception as e:
            return None

    return world_points[:3].T

def world_to_image_tf(world_x, world_y, world_z, tf_camera_to_world, camera_vector):

    H_cam_to_world = transform_to_homogeneous(tf_camera_to_world)
    H_cam_to_world_inv = np.linalg.inv(H_cam_to_world)

    # Convert the point to a numpy array
    point_vec = np.array([world_x, world_y, world_z])

    # Apply the transform: Rotate then translate
    cam_point = H_cam_to_world_inv @ np.append(point_vec, 1)

    return camera_to_image(cam_point[0], cam_point[1], cam_point[2], camera_vector)

def world_to_image_pose(world_x, world_y, world_z, pose_in_world, camera_vector):
    H_world_to_cam = pose_to_homogeneous(pose_in_world)
    H_cam_to_world = np.linalg.inv(H_world_to_cam)

    # Convert the point to a numpy array
    world_point = np.array([world_x, world_y, world_z])

    # Apply the transform: Rotate then translate
    cam_in_world_point = H_cam_to_world @ np.append(world_point, 1)
    cam_in_cam_point = np.array([-cam_in_world_point[1], -cam_in_world_point[2], cam_in_world_point[0]])

    return camera_to_image(cam_in_cam_point[0], cam_in_cam_point[1], cam_in_cam_point[2], camera_vector)


def image_to_camera(image_points, depth, camera_vector):
    fx, fy, cx, cy = camera_vector
    camera_x = (image_points[:,0] - cx).reshape(-1, 1) * depth / fx
    camera_y = (image_points[:,1] - cy).reshape(-1, 1) * depth / fy
    return camera_x, camera_y, depth

def camera_to_image(camera_x, camera_y, camera_z, camera_vector):
    fx, fy, cx, cy = camera_vector
    image_x = (camera_x * fx / camera_z) + cx
    image_y = (camera_y * fy / camera_z) + cy
    return image_x, image_y

def get_yaw_z_from_quaternion(x, y, z, w):
    """
    Calculate the yaw (rotation around the z-axis) from a quaternion.
    
    Parameters:
    x, y, z, w -- components of the quaternion
    
    Returns:
    yaw -- the yaw angle in radians
    """
    # Calculate the yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def get_yaw_x_from_quaternion(x, y, z, w):
    """
    Calculate the yaw (rotation around the x-axis) from a quaternion.
    
    Parameters:
    x, y, z, w -- components of the quaternion
    
    Returns:
    yaw -- the yaw angle in radians
    """
    # Calculate the yaw (z-axis rotation)
    sinx_cosp = 2 * (w * x + y * z)
    cosx_cosp = 1 - 2 * (x**2 + y**2)
    yaw = np.arctan2(sinx_cosp, cosx_cosp)
    return yaw

def get_distance_between_3D_point(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.
    
    Parameters:
    point1, point2 -- the 3D points
    
    Returns:
    distance -- the Euclidean distance between the points
    """
    # Calculate the Euclidean distance
    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)
    return distance
