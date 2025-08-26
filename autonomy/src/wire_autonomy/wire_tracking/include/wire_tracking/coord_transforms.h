// transforms.hpp
#ifndef TRANSFORMS_HPP_
#define TRANSFORMS_HPP_
#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>

Eigen::Matrix4d poseToHomogeneous2(const Eigen::Matrix4d &H_cam_to_fc_, const geometry_msgs::msg::Pose &pose_fc_to_w);

Eigen::Matrix4d getRelativeTransform(
    const Eigen::Matrix4d &from_transform,
    const Eigen::Matrix4d &to_transform);

std::pair<Eigen::Matrix4d, Eigen::Matrix4d> getRelativeTransformInCam(
    const Eigen::Matrix4d &H_cam_to_fc_,
    const Eigen::Matrix4d &H_cam_to_w1,
    const geometry_msgs::msg::Pose &pose_fc_to_w2);

#endif // TRANSFORMS_HPP_