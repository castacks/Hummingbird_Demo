#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>

#include "coord_transforms.h"

using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::Quaterniond;

// Convert a ROS Pose message into a 4×4 homogeneous transform
Matrix4d poseToHomogeneous(const geometry_msgs::msg::Pose & pose) {
  Matrix4d H = Matrix4d::Identity();

  // Translation
  H.block<3,1>(0,3) = Vector3d(
    pose.position.x,
    pose.position.y,
    pose.position.z
  );

  // Rotation: Eigen::Quaterniond(w, x, y, z)
  Quaterniond q{
    pose.orientation.w,
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z
  };
  H.block<3,3>(0,0) = q.normalized().toRotationMatrix();

  return H;
}

// Compute the relative transform from pose1 to pose2
std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransform(
  const Eigen::Matrix4d & from_transform,
  const geometry_msgs::msg::Pose & to_pose)
{
  // Build homogeneous matrices
  Matrix4d to_transform = poseToHomogeneous(to_pose);

  // Relative: H_rel = H1⁻¹ * H2
  return {from_transform.inverse() * to_transform, to_transform};
}

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransformInAnotherFrame(const Eigen::Matrix4d &frame_transform, const Eigen::Matrix4d &from_transform, const geometry_msgs::msg::Pose &to_pose)
{
    auto relative_transforms = getRelativeTransform(from_transform, to_pose);

    // Transform the relative frame transform to the new frame
    return {frame_transform * relative_transforms.first * frame_transform.inverse(), relative_transforms.second};
}
