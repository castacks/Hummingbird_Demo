#ifndef WIRE_TRACKING__WIRE_TRACKING_NODE_HPP_
#define WIRE_TRACKING__WIRE_TRACKING_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <wire_interfaces/msg/wire_detections.hpp>
#include <wire_interfaces/msg/wire_target.hpp>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include "direction_kf.h"
#include "position_kf.h"

class WireTrackingNode : public rclcpp::Node
{
public:
  explicit WireTrackingNode();

private:
  // ROS callbacks
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void poseCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void wireDetectionCallback(const wire_interfaces::msg::WireDetections::SharedPtr msg);
  void targetTimerCallback();
  void rgbCallback(const sensor_msgs::msg::Image::SharedPtr rgb_msg);

  // helper loaders
  void loadConfig();
  void publishKFsViz(cv::Mat img, double stamp, const std::vector<WireDetection> *wire_detections);
  void visualize3DWires();
  void predict_kfs_up_to_timestamp(double input_stamp);

  // members
  std::string camera_info_topic_, pose_topic_, detections_topic_, target_topic_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
  rclcpp::Subscription<wire_interfaces::msg::WireDetections>::SharedPtr detection_sub_;
  rclcpp::Publisher<wire_interfaces::msg::WireTarget>::SharedPtr target_pub_;
  rclcpp::TimerBase::SharedPtr target_timer_;

  // Kalman filters
  DirectionKalmanFilter direction_kf_;
  PositionKalmanFilter position_kf_;
  int tracked_wire_id_{-1};
  int total_iterations_{0};
  int iteration_start_threshold_{0}; // Number of iterations before starting to track
  int min_valid_kf_count_threshold_{0}; // Minimum number of valid Kalman filters to consider tracking

  // Queues
  std::vector<double> relative_transform_timestamps_;
  std::vector<Eigen::Matrix4d> relative_transforms_;
  Eigen::Matrix4d previous_relative_transform_;
  std::vector<double> rgb_timestamps_;
  std::vector<cv::Mat> rgb_images_;
  

  Eigen::Matrix3d camera_matrix_;
  Eigen::Matrix4d H_pose_to_wire_, H_wire_to_pose_;
  bool initialized_{false};
  int image_height_, image_width_;
  double fx_, fy_, cx_, cy_;
  YAML::Node config_;
  bool wire_viz_{false};
  int wire_mode_{0}; // 0: off, 1: wire_detection, 2: wire_tracking
};

#endif // WIRE_TRACKING__WIRE_TRACKING_NODE_HPP_
