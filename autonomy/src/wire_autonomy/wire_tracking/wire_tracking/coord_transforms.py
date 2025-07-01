import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Transform, Pose, Quaternion

def get_relative_transform(pose1, pose2):
    """
    Get the relative transformation from pose1 to pose2.
    
    Parameters:
    pose1, pose2 -- the poses in the world frame

    Returns:
    relative_transform -- the relative transformation from pose1 to pose2
    """
    assert isinstance(pose1, Pose) and isinstance(pose2, Pose), "Both inputs must be Pose messages"
    
    # Convert poses to homogeneous matrices
    H1 = pose_to_homogeneous(pose1)
    H2 = pose_to_homogeneous(pose2)
    
    # Calculate the relative transformation
    relative_transform = np.linalg.inv(H1) @ H2
    
    return relative_transform

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
    quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    rot_matrix = Rotation.from_quat(quaternion).as_matrix()

    # Combine into a homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rot_matrix[:3, :3]
    homogeneous_matrix[:3, 3] = translation_vector
    assert homogeneous_matrix.shape == (4, 4), "Output must be a 4x4 numpy array, got %s" % homogeneous_matrix.shape
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

def points_in_cam_to_world(points_in_cam, H_cam_to_world):
    """
    Converts points in camera frame to world frame using a pose.

    Args:
        points_in_cam (np.ndarray): Points in camera frame (Nx3).
        pose_cam_to_world (Pose): The pose of the camera in the world frame.

    Returns:
        np.ndarray: Points in world frame (Nx3).
    """
    assert points_in_cam.shape[1] == 3, f"Points must be Nx3, got {points_in_cam.shape}"
    assert len(points_in_cam.shape) == 2, f"Points must be Nx3, got {points_in_cam.shape}"
    assert H_cam_to_world.shape == (4, 4), f"Homogeneous transformation matrix must be 4x4, got {H_cam_to_world.shape}"


    # Convert the point to a numpy array
    points_in_cam_homogeneous = np.hstack((points_in_cam, np.ones((points_in_cam.shape[0], 1))))
    
    # Apply the transform: Rotate then translate
    points_in_world = ((H_cam_to_world @ points_in_cam_homogeneous.T).T)[:,:3]
    
    return points_in_world

def world_to_image_pose(world_points, pose_in_world, camera_vector):
    '''
    Convert world points to image points using a pose

    Parameters:
    world_points -- the world points in the world frame (Nx3)
    pose_in_world -- the pose of the camera in the world frame
    camera_vector -- the camera vector (fx, fy, cx, cy)

    Returns:
    image_points_x -- the x-coordinates of the image points, (Nx1)
    image_points_y -- the y-coordinates of the image points, (Nx1)
    '''
    assert world_points.shape[1] == 3, f"World points must be Nx3, got {world_points.shape}"
    assert len(world_points.shape) == 2, f"World points must be Nx3, got {world_points.shape}"

    H_cam_to_world = pose_to_homogeneous(pose_in_world)

    # Convert the point to a numpy array
    world_points_homogeneous = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
    assert world_points_homogeneous.shape[1] == 4, f"World point must be Nx3, got {world_points_homogeneous.shape}"

    # Apply the transform: Rotate then translate
    cam_points_in_world = ((H_cam_to_world @ world_points_homogeneous.T).T)[:,:3]
    assert cam_points_in_world.shape[1] == 3, f"World point must be Nx3, got {cam_points_in_world.shape}"
    assert len(cam_points_in_world.shape) == 2, f"World point must be Nx3, got {cam_points_in_world.shape}"

    # changing world convention to camera convention
    cam_points_in_cam = np.vstack([cam_points_in_world[:,1], - cam_points_in_world[:,0], cam_points_in_world[:,2]]).T

    img_xs, img_ys = camera_to_image(cam_points_in_cam, camera_vector)
    return img_xs, img_ys

def image_to_camera(image_points, depth, camera_vector):
    '''
    Convert image points to camera points

    Parameters:
    image_points -- the image points in the image frame (Nx2)
    depth -- the depth of the image points (N)
    camera_vector -- the camera vector (fx, fy, cx, cy)

    Returns:
    camera_points_x -- the x-coordinates of the camera points
    camera_points_y -- the y-coordinates of the camera points
    camera_points_z -- the z-coordinates of the camera points
    '''
    assert len(image_points) == len(depth), "The number of image points must match the number of depth values"
    assert image_points.shape[1] == 2 and image_points.shape[0] == depth.shape[0], "Image points must be Nx2 and depth must be Nx1"

    fx, fy, cx, cy = camera_vector
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    image_points = np.hstack((image_points, np.ones((image_points.shape[0], 1))))

    camera_points = np.dot(inv_camera_matrix, image_points.T).T * depth.reshape(-1, 1)
    return camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]

def camera_to_image(camera_points, camera_vector):
    '''
    Convert camera points to image points

    Parameters:
    camera_points -- the camera points in the camera frame (Nx3)
    camera_vector -- the camera vector (fx, fy, cx, cy)

    Returns:
    image_points_x -- the x-coordinates of the image points
    image_points_y -- the y-coordinates of the image points
    '''
    assert camera_points.shape[1] == 3, f"Camera points must be Nx3, got {camera_points.shape}"
    fx, fy, cx, cy = camera_vector
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    image_points = np.dot(camera_matrix, camera_points.T).T
    image_points /= image_points[:, 2].reshape(-1, 1)
    image_points = image_points[:, :2]

    return image_points[:,0], image_points[:,1]

def get_distance_between_3D_points(point1, point2):
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
