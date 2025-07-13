// transforms.hpp
#ifndef TRANSFORMS_HPP_
#define TRANSFORMS_HPP_
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>

Eigen::Matrix4d poseToHomogeneous(const geometry_msgs::msg::Pose &pose);

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransform(
    const Eigen::Matrix4d &from_transform,
    const geometry_msgs::msg::Pose &to_pose);

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransformInAnotherFrame(
    const Eigen::Matrix4d &to_frame_transform,
    const Eigen::Matrix4d &from_frame_transform,
    const Eigen::Matrix4d &from_transform,
    const geometry_msgs::msg::Pose &to_pose);

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransformInAnotherFrame(
    const Eigen::Matrix4d &to_frame_transform,
    const Eigen::Matrix4d &from_frame_transform,
    const Eigen::Matrix4d &relative_transform,
    const Eigen::Matrix4d &from_transform);

Eigen::Matrix4d getVelocityRelativeTransform(const Eigen::Vector3d &lin_vel, const Eigen::Vector3d &ang_vel, double dt);

std::pair<Eigen::Vector3d, Eigen::Vector3d> getVelocityFromTransforms(
    const Eigen::Matrix4d &from_transform,
    const Eigen::Matrix4d &to_transform,
    double dt);

#endif // TRANSFORMS_HPP_