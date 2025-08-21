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
  
  // helper functions
  void loadConfig();
  void predict_kfs_up_to_timestamp(double input_stamp);
  void debugPrintKFs();
  void visualizeWireTracking(cv::Mat img, double stamp, const std::vector<wire_interfaces::msg::WireDetection> *wire_detections);

  // topics
  std::string camera_info_sub_topic_;
  std::string pose_sub_topic_;
  std::string wire_detections_topic_;
  std::string wire_target_topic_;
  std::string rgb_image_sub_topic_;
  std::string tracking_2d_viz_topic_;
  std::string tracking_3d_viz_topic_;

  // members
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
  rclcpp::Subscription<wire_interfaces::msg::WireDetections>::SharedPtr detection_sub_;

  rclcpp::Publisher<wire_interfaces::msg::WireTarget>::SharedPtr target_pub_;
  rclcpp::TimerBase::SharedPtr target_timer_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr tracking_2d_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr tracking_3d_pub_;
  

  // Kalman filters
  std::optional<DirectionKalmanFilter> direction_kalman_filter_;
  std::optional<PositionKalmanFilters> position_kalman_filters_;
  int tracked_wire_id_{-1};
  int total_iterations_{0};
  int iteration_start_threshold_{0}; // Number of iterations before starting to track
  int min_valid_kf_count_threshold_{0}; // Minimum number of valid Kalman filters to consider tracking

  double linear_translation_dropout_{0.5}; // Dropout threshold for linear translation in meters
  double angular_translation_dropout_{0.25}; // Dropout threshold for angular translation in radians
  double linear_velocity_dropout_{3.0}; // Dropout threshold for linear velocity in meters/second
  double angular_velocity_dropout_{0.5}; // Dropout threshold for angular velocity in radians/second
  double linear_acceleration_dropout_{5.0}; // Dropout threshold for linear acceleration in meters/second^2
  double angular_acceleration_dropout_{8.0}; // Dropout threshold for angular acceleration in radians/second^2
  bool using_velocity_transform_{false}; // Flag to indicate if velocity transform is being used

  // Queues
  std::vector<double> relative_transform_timestamps_;
  std::vector<Eigen::Matrix4d> relative_transforms_;
  Eigen::Matrix4d previous_transform_ = Eigen::Matrix4d::Zero();
  double previous_transform_stamp_{0.0};
  Eigen::Vector3d last_linear_velocity_{0,0,0};
  Eigen::Vector3d last_angular_velocity_{0,0,0};

  std::vector<double> rgb_timestamps_;
  std::vector<cv::Mat> rgb_images_;

  Eigen::Matrix3d camera_matrix_;
  Eigen::Matrix4d H_fc_to_cam_, H_cam_to_fc_;
  bool initialized_{false};
  int image_height_, image_width_;
  double line_length_{0.0}; // Length of the line to be drawn in pixels
  double fx_, fy_, cx_, cy_;
  YAML::Node config_;
  bool wire_viz_{false};
  int wire_mode_{0}; // 0: off, 1: wire_detection, 2: wire_tracking
  bool wire_viz_2d_{false}; // 2D visualization flag
  bool wire_viz_3d_{false}; // 3D visualization flag
};

#endif // WIRE_TRACKING__WIRE_TRACKING_NODE_HPP_
