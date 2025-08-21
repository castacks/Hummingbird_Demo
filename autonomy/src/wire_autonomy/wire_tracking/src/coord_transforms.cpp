#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>

#include "wire_tracking/coord_transforms.h"

using Eigen::Matrix4d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

// Convert a ROS Pose message into a 4×4 homogeneous transform
Matrix4d poseToHomogeneous2(const Eigen::Matrix4d &H_cam_to_fc_, const geometry_msgs::msg::Pose &pose_fc_to_w)
{
  Matrix4d H_fc_to_w = Matrix4d::Identity();
  Matrix4d H_cam_to_world = Matrix4d::Identity();
  // Translation
  H_fc_to_w.block<3, 1>(0, 3) = Vector3d(
      pose.position.x,
      pose.position.y,
      pose.position.z);

  // Rotation: Eigen::Quaterniond(w, x, y, z)
  Quaterniond q{
      pose_fc_to_w.orientation.w,
      pose_fc_to_w.orientation.x,
      pose_fc_to_w.orientation.y,
      pose_fc_to_w.orientation.z};
  H_fc_to_w.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();

  H_cam_to_world = H_cam_to_fc_ * H_fc_to_w;
  return H_cam_to_world;
}

// Compute the relative transform from pose1 to pose2
Eigen::Matrix4d getRelativeTransform(
    const Eigen::Matrix4d &from_transform,
    const Eigen::Matrix4d &to_transform)
{
  // Relative: H_rel = H1⁻¹ * H2
  // return {from_transform.inverse() * to_transform, to_transform};
  return to_transform.inverse() * from_transform;
}

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransformInCam(
    const Eigen::Matrix4d &H_cam_to_fc_,
    const Eigen::Matrix4d &H_cam_to_w1,
    const geometry_msgs::msg::Pose &pose_fc_to_w2)
{
  Eigen::Matrix4d H_cam_to_w2 = poseToHomogeneous2(H_cam_to_fc_, pose_fc_to_w2);

  Eigen::Matrix4d H_relative_cam = getRelativeTransform(H_cam_to_w1, H_cam_to_w2);

  // Transform the relative frame transform to the new frame
  return {H_relative_cam, H_cam_to_w2};
}