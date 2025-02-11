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

def rotvec_to_homogeneous(rotvec, x, y, z):
    """
    Converts a rotation vector and a displacement into a 4x4 homogeneous transformation matrix.

    Args:
        rotvec: a 3x1 rotation vector listing rotation in x, y, z
        x: displacement in meteres
        y: displacement in meteres
        z: displacement in meteres

    Returns:
        np.ndarray: A 4x4 homogeneous transformation matrix.
    """
    translation_vector = np.array([x, y, z])
    rot_matrix = Rotation.from_rotvec(rotvec).as_matrix()

    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rot_matrix[:3, :3]
    homogeneous_matrix[:3, 3] = translation_vector

    assert homogeneous_matrix.shape == (4, 4), "Output must be a 4x4 numpy array, got %s" % homogeneous_matrix.shape
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
    # quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w]) * np.array([1, -1, -1, 1])
    quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
    rot_matrix = Rotation.from_quat(quaternion).as_matrix()
    # assert False, f"rot_vector: {Rotation.from_quat(quaternion).as_rotvec()}"

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
    
def image_to_world_tf(image_points, depth, tf_camera_to_world, camera_vector):
    ''' 
    Convert image points to world points using a pose
    
    Parameters:
    image_points -- the image points in the image frame (Nx2)
    depth -- the depth of the image points (Nx1)
    pose_in_world -- the pose of the camera in the world frame
    
    Returns:
    world_points -- the world points in the world frame (Nx3)
    '''
    camera_points_x, camera_points_y, camera_points_z = image_to_camera(image_points, depth.reshape(-1, 1), camera_vector)

    # transforming from camera frame to world convention
    camera_points_in_world_x = camera_points_z
    camera_points_in_world_y = - camera_points_x
    camera_points_in_world_z = - camera_points_y

    H_cam_to_world = transform_to_homogeneous(tf_camera_to_world)

    # Convert the point to a numpy array
    # point_vec = np.hstack((x_c, y_c, z_c, np.ones_like(x_c)))
    point_vec = np.hstack((camera_points_in_world_x, camera_points_in_world_y, camera_points_in_world_z, np.ones_like(camera_points_in_world_x)))

    # Apply the transform: Rotate then translate
    with np.errstate(invalid='ignore'):
        try:
            world_points = H_cam_to_world @ point_vec.T
        except Exception as e:
            return None

    return world_points

def image_to_world_pose(image_points, depth, pose_in_world, camera_vector):
    ''' 
    Convert image points to world points using a pose
    
    Parameters:
    image_points -- the image points in the image frame (Nx2)
    depth -- the depth of the image points (Nx1)
    pose_in_world -- the pose of the camera in the world frame
    
    Returns:
    world_points -- the world points in the world frame (Nx3)
    '''
    assert len(image_points) == len(depth), "The number of image points must match the number of depth values"
    assert image_points.shape[1] == 2 and image_points.shape[0] == depth.shape[0], "Image points must be Nx2 and depth must be Nx1"
    camera_points_x, camera_points_y, camera_points_z = image_to_camera(image_points, depth.reshape(-1, 1), camera_vector)

    # transforming from camera convention to world convention
    camera_points_in_world_x = - camera_points_y
    camera_points_in_world_y = camera_points_x
    camera_points_in_world_z = camera_points_z

    # zed gives pose of camera going to world
    H_cam_to_world = pose_to_homogeneous(pose_in_world)
    H_world_to_cam = np.linalg.inv(H_cam_to_world)

    # Convert the point to a numpy array
    point_vec = np.vstack((camera_points_in_world_x, camera_points_in_world_y, camera_points_in_world_z, np.ones_like(camera_points_x))).T
    assert len(point_vec.shape) == 2, f"Point vector must be Nx4, got {point_vec.shape}"
    assert point_vec.shape[1] == 4 and point_vec.shape[0] == len(image_points), f"Point vector must be Nx4, got {point_vec.shape}"

    # Apply the transform: Rotate then translate
    world_points = ((H_world_to_cam @ point_vec.T).T)[:, :3]

    return world_points

def world_to_image_tf(world_points, tf_camera_to_world, camera_vector):
    '''
    Convert world points to image points using a transform

    Parameters:
    world_points -- the world points in the world frame (Nx3)
    pose_in_world -- the pose of the camera in the world frame
    camera_vector -- the camera vector (fx, fy, cx, cy)

    Returns:
    image_points_x -- the x-coordinates of the image points
    image_points_y -- the y-coordinates of the image points
    '''
    
    H_world_to_cam = transform_to_homogeneous(tf_camera_to_world)
    H_cam_to_world = np.linalg.inv(H_world_to_cam)

    # Convert the point to a numpy array
    world_point = np.hstack((world_points, np.ones((world_points.shape[0], 1))))

    # Apply the transform: Rotate then translate
    cam_in_world_points = H_cam_to_world @ world_point.T
    # cam_in_cam_point = np.array([-cam_in_world_point[1], -cam_in_world_point[2], cam_in_world_point[0]])
    cam_in_cam_points = np.array([cam_in_world_points[:,1], -cam_in_world_points[:,0], -cam_in_world_points[:,2]])

    return camera_to_image(cam_in_cam_points, camera_vector)

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
