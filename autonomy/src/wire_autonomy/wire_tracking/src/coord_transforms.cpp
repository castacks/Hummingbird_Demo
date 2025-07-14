#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>

#include "wire_tracking/coord_transforms.h"

using Eigen::Matrix4d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

// Convert a ROS Pose message into a 4×4 homogeneous transform
Matrix4d poseToHomogeneous(const geometry_msgs::msg::Pose &pose)
{
  Matrix4d H = Matrix4d::Identity();

  // Translation
  H.block<3, 1>(0, 3) = Vector3d(
      pose.position.x,
      pose.position.y,
      pose.position.z);

  // Rotation: Eigen::Quaterniond(w, x, y, z)
  Quaterniond q{
      pose.orientation.w,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z};
  H.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();

  return H;
}

// Compute the relative transform from pose1 to pose2
std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransform(
    const Eigen::Matrix4d &from_transform,
    const geometry_msgs::msg::Pose &to_pose)
{
  // Build homogeneous matrices
  Matrix4d to_transform = poseToHomogeneous(to_pose);

  // Relative: H_rel = H1⁻¹ * H2
  // return {from_transform.inverse() * to_transform, to_transform};
  return {to_transform.inverse() * from_transform, to_transform};
}

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransformInAnotherFrame(const Eigen::Matrix4d &to_frame_transform,
                                                                               const Eigen::Matrix4d &from_frame_transform,
                                                                               const Eigen::Matrix4d &from_transform,
                                                                               const geometry_msgs::msg::Pose &to_pose)
{
  Eigen::Matrix4d relative_transform;
  Eigen::Matrix4d to_pose_transform;
  std::tie(relative_transform, to_pose_transform) = getRelativeTransform(from_transform, to_pose);

  // Transform the relative frame transform to the new frame
  return {to_frame_transform * relative_transform * from_frame_transform, to_pose_transform};
}

Eigen::Matrix4d getRelativeTransformInAnotherFrame(const Eigen::Matrix4d &to_frame_transform,
                                                   const Eigen::Matrix4d &from_frame_transform,
                                                   const Eigen::Matrix4d &relative_transform)
{
  // Transform the relative frame transform to the new frame
  // TODO: check the H_to_pose math
  return to_frame_transform * relative_transform * from_frame_transform;
}

Eigen::Matrix4d getRelativeTransformFromVelocity(
    const Eigen::Vector3d &lin_vel,
    const Eigen::Vector3d &ang_vel,
    double dt)
{
  // 1) Compute rotation ΔR from ω * dt
  double omega_norm = ang_vel.norm();
  Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();
  if (omega_norm > 1e-8)
  {
    double theta = omega_norm * dt;              // total rotation angle
    Eigen::Vector3d axis = ang_vel / omega_norm; // unit rotation axis
    R_delta = Eigen::AngleAxisd(theta, axis).toRotationMatrix();
  }

  // 2) Compute translation Δp = v * dt
  Eigen::Vector3d delta_p = lin_vel * dt;

  // 3) Assemble into homogeneous matrix
  Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
  H.block<3, 3>(0, 0) = R_delta;
  H.block<3, 1>(0, 3) = delta_p;
  // return H;
  return H.inverse(); // Return the inverse to get the relative transform
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> getVelocityFromTransforms(
    const Eigen::Matrix4d &from_transform,
    const Eigen::Matrix4d &to_transform,
    double dt)
{
  // Compute the relative translation and rotation
  Eigen::Vector3d translation = to_transform.block<3, 1>(0, 3) - from_transform.block<3, 1>(0, 3);
  Eigen::Vector3d linear_velocity = translation / dt;

  Eigen::Quaterniond rotation_from(from_transform.block<3, 3>(0, 0));
  Eigen::Quaterniond rotation_to(to_transform.block<3, 3>(0, 0));
  Eigen::Quaterniond relative_rotation = rotation_to * rotation_from.inverse();

  Eigen::AngleAxisd aa(relative_rotation);
  double angle = aa.angle();                     // θ in radians
  Eigen::Vector3d axis = aa.axis().normalized(); // unit axis u

  // 4) Angular velocity ω = u * (θ / dt)
  Eigen::Vector3d angular_velocity = axis * (angle / dt);

  // Compute linear velocity

  return {linear_velocity, angular_velocity};
}
