// transforms.hpp
#ifndef TRANSFORMS_HPP_
#define TRANSFORMS_HPP_
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>

Eigen::Matrix4d poseToHomogeneous(const geometry_msgs::msg::Pose & pose);

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransform(
  const Eigen::Matrix4d & from_transform,
  const geometry_msgs::msg::Pose & to_pose);

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransformInAnotherFrame(
  const Eigen::Matrix4d & to_frame_transform,
  const Eigen::Matrix4d & from_frame_transform,
  const Eigen::Matrix4d & from_transform,
  const geometry_msgs::msg::Pose & to_pose);

#endif  // TRANSFORMS_HPP_