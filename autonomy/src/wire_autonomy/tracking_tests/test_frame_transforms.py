import numpy as np
from scipy.spatial.transform import Rotation as R

ZED_BASELINE_M = 0.12 / 2 # ZED baseline in meters

def initialize_transforms():

    # Create rotation matrices
    Rz = R.from_rotvec(-np.pi / 2 * np.array([0, 0, 1])).as_matrix()
    Ry = R.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()

    # Compose transformation matrix
    H_pose_to_wire = np.eye(4)
    H_pose_to_wire[:3, :3] = Rz @ Ry
    H_pose_to_wire[:3, 3] = np.array([ZED_BASELINE_M, 0.0, -0.216])

    # Inverse transform
    H_wire_to_pose = np.linalg.inv(H_pose_to_wire)
    return H_pose_to_wire, H_wire_to_pose
    
def test_frame_transforms():
    H_pose_to_wire, H_wire_to_pose = initialize_transforms()

    pose_relative_transform = np.eye(4)
    pose_relative_transform[:3, :3] = R.from_rotvec(np.array([0.0, 0.0, np.pi / 2])).as_matrix()
    pose_relative_transform[:3, 3] = np.array([0.0, 0.0, 0.0])

    wire_relative_transform = H_pose_to_wire @ pose_relative_transform @ H_wire_to_pose
    wire_relative_transform = wire_relative_transform.astype(np.float32)

    print("H_pose_to_wire:\n", H_pose_to_wire)
    print("H_wire_to_pose:\n", H_wire_to_pose)
    print("Pose relative transform:\n", pose_relative_transform)
    print("Wire relative transform:\n", wire_relative_transform)

if __name__ == "__main__":
    test_frame_transforms()