#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <wire_interfaces/msg/wire_detections.hpp>
#include <wire_interfaces/msg/wire_target.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

#include "wire_tracking_node.h"
#include "coord_transforms.h"
#include "direction_kf.h"
#include "position_kf.h"
#include "wire_tracking_node..h"

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::placeholders::_1;

WireTrackingNode::WireTrackingNode() : Node("wire_tracking_node")
{
    // Load parameters and config
    this->declare_parameter<std::string>("camera_info_topic", "/camera/info");
    this->get_parameter("camera_info_topic", camera_info_topic_);
    loadConfig();

    // Initialize subscriptions
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, 1,
        std::bind(&WireTrackingNode::cameraInfoCallback, this, _1));

    detection_sub_ = this->create_subscription<wire_interfaces::msg::WireDetections>(
        detections_topic_, 1,
        std::bind(&WireTrackingNode::wireDetectionCallback, this, _1));

    target_pub_ = this->create_publisher<wire_interfaces::msg::WireTarget>(target_topic_, 10);
    target_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / config_["target_publish_frequency_hz"].as<double>()),
        std::bind(&WireTrackingNode::targetTimerCallback, this));

    // Precompute transforms
    double baseline = config_["zed_baseline"].as<double>() / 2.0;
    Matrix3d Rz;
    Rz = Eigen::AngleAxisd(-M_PI / 2, Vector3d::UnitZ());
    Matrix3d Ry;
    Ry = Eigen::AngleAxisd(M_PI, Vector3d::UnitY());
    H_pose_to_wire_.setIdentity();
    H_pose_to_wire_.topLeftCorner<3, 3>() = Rz * Ry;
    H_pose_to_wire_.block<3, 1>(0, 3) = Vector3d(baseline, 0.0, -0.216);
    H_wire_to_pose_ = H_pose_to_wire_.inverse();

    // Populate Kalman filters
    direction_kf_ = DirectionKalmanFilter(config_);

    RCLCPP_INFO(this->get_logger(), "Wire Tracking Node initialized");
}

void WireTrackingNode::loadConfig()
{
    std::string pkg_share = ament_index_cpp::get_package_share_directory("wire_tracking");
    YAML::Node cfg = YAML::LoadFile(pkg_share + "/config/wire_tracking_config.yaml");
    config_ = cfg;
    pose_topic_ = cfg["pose_sub_topic"].as<std::string>();
    detections_topic_ = cfg["wire_detections_pub_topic"].as<std::string>();
    target_topic_ = cfg["wire_target_pub_topic"].as<std::string>();

    const char *wire_viz_raw = std::getenv("WIRE_VIZ");
    const char *wire_mode_raw = std::getenv("WIRE_MODE");
    if (wire_viz_raw)
    {
        std::string wire_viz_str(wire_viz_raw);
        std::transform(wire_viz_str.begin(), wire_viz_str.end(), wire_viz_str.begin(), ::tolower);
        std::string wire_mode_str(wire_mode_raw);
        std::transform(wire_mode_str.begin(), wire_mode_str.end(), wire_mode_str.begin(), ::tolower);
        if (wire_viz_str == "true" || wire_viz_str == "1")
        {
            wire_viz_ = true;
        }
        wire_mode_ = std::stoi(wire_mode_str);
    }
}

void WireTrackingNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (initialized_)
        return;
    fx_ = msg->k[0];
    fy_ = msg->k[4];
    cx_ = msg->k[2];
    cy_ = msg->k[5];
    camera_matrix_ = Matrix3d();
    camera_matrix_ << fx_, 0, cx_,
        0, fy_, cy_,
        0, 0, 1;
    image_height_ = msg->height;
    image_width_ = msg->width;
    // Init the position Kalman filter
    position_kf_ = PositionKalmanFilter(config_, camera_matrix_, {image_width_, image_height_});
    initialized_ = true;
    camera_info_sub_.reset();
    // creating the pose subscription after camera info is received
    pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        pose_topic_, rclcpp::SensorDataQoS(),
        std::bind(&WireTrackingNode::poseCallback, this, _1));
    RCLCPP_INFO(this->get_logger(), "Wire Tracking Node initialized, Camera info received.");
}

void WireTrackingNode::poseCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (!initialized_)
        return;
    // if last relative transform is not set, initialize it
    if (previous_relative_transform_.isZero())
    {
        previous_relative_transform_ = poseToHomogeneous(msg->pose.pose.pose);
    }
    else
    {
        // Compute the relative transform from the previous pose to the current pose
        auto [H_relative_in_wire_cam, H_to_pose] = getRelativeTransformInAnotherFrame(
            H_pose_to_wire_, previous_relative_transform_, msg->pose.pose.pose);
        // Update the previous relative transform
        double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        relative_transform_timestamps_.push_back(stamp);
        relative_transforms_.push_back(H_relative_in_wire_cam);
        previous_relative_transform_ = H_to_pose;
    }
}

void WireTrackingNode::predict_kfs_up_to_timestamp(double input_stamp)
{
    if (!initialized_ || !position_kalman_filters_.initialized())
    {
        RCLCPP_INFO(this->get_logger(), "Wire Tracking Node not initialized yet. Waiting for camera info.");
        return;
    }

    if (relative_transform_timestamps_.empty())
    {
        RCLCPP_INFO(this->get_logger(), "No relative pose transforms available. Waiting for pose updates.");
        return;
    }

    // Find index for latest pose before input_stamp
    auto it = std::upper_bound(relative_transform_timestamps_.begin(),
                               relative_transform_timestamps_.end(),
                               input_stamp);
    int idx = static_cast<int>(std::distance(relative_transform_timestamps_.begin(), it)) - 1;

    if (idx < 0)
    {
        RCLCPP_INFO(this->get_logger(), "No relative pose transforms available for prediction.");
        return;
    }

    // Iterate over poses up to idx
    for (int i = 0; i < idx; ++i)
    {
        const Eigen::Matrix4d &relative_pose_transform = relative_transforms_[i];

        if (direction_kalman_filter_.initialized())
        {
            double previous_yaw = direction_kalman_filter_.getYaw();
            double curr_yaw = direction_kalman_filter_.predict(relative_pose_transform.block<3, 3>(0, 0));

            if (position_kalman_filters_.initialized())
            {
                position_kalman_filters_.predict(relative_pose_transform, previous_yaw, curr_yaw);
            }
        }
    }

    // Remove old transforms
    relative_transforms_.erase(relative_transforms_.begin(), relative_transforms_.begin() + idx);
    relative_transform_timestamps_.erase(relative_transform_timestamps_.begin(), relative_transform_timestamps_.begin() + idx);
}

void WireTrackingNode::wireDetectionCallback(const wire_interfaces::msg::WireDetections::SharedPtr msg)
{
    if (!initialized_)
        return;
    double wire_detection_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    if (!relative_transforms_.empty())
    {
        // Use the most recent relative transform
        predict_kfs_up_to_timestamp(wire_detection_stamp);
    }

    if (msg->avg_angle < 0.0 || msg->avg_angle > M_PI || msg->wire_detections.empty())
    {
        return;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3> wire_points_xyz(N, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 3> wire_directions(N, 3);

    Eigen::Vector3d reference_dir;

    for (std::size_t i = 0; i < N; ++i)
    {
        const auto &detection = detections[i];

        // start and end as Eigen vectors
        Eigen::Vector3d start(detection.start.x, detection.start.y, detection.start.z);
        Eigen::Vector3d end(detection.end.x, detection.end.y, detection.end.z);

        // midpoint
        wire_points_xyz.row(i) = 0.5 * (start + end);

        // direction from line endpoints (your C++ API)
        if (direction_kalman_filter.initialized())
        {
            wire_directions.row(i) = direction_kalman_filter.get_direction_from_line_end_points(start, end);
        }
        else if (i == 0)
        {
            reference_dir = direction_kalman_filter.get_direction_from_line_end_points(start, end);
            wire_directions.row(i) = reference_dir;
        }
        else
        {
            wire_directions.row(i) = direction_kalman_filter.get_direction_from_line_end_points(start, end, reference_dir);
        }
    }

    // compute average direction and normalize
    Eigen::Vector3d avg_dir = wire_directions.colwise().mean();
    avg_dir.normalize();

    // update or initialize your direction filter
    if (direction_kalman_filter.initialized())
    {
        direction_kalman_filter.update(avg_dir);
        double measured_yaw = std::atan2(avg_dir.y(), avg_dir.x()) * 180.0 / M_PI;
        double current_yaw = direction_kalman_filter.get_yaw() * 180.0 / M_PI;
        RCLCPP_INFO(this->get_logger(), "Average measured yaw: %.2f degrees", measured_yaw);
        RCLCPP_INFO(this->get_logger(), "Current wire yaw: %.2f degrees", current_yaw);
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Direction filter with new wire direction.");
        direction_kalman_filter.initialize(avg_dir);
    }

    // update or initialize your position filters
    double yaw = direction_kalman_filter.get_yaw();
    if (position_kalman_filters.initialized())
    {
        position_kalman_filters.update(wire_points_xyz, yaw);
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Position Filters with new wire locations.");
        position_kalman_filters.initialize_kfs(wire_points_xyz, yaw);
    }
    total_iterations_++;

    if (wire_viz_2d_)
    {
        // Find the closest timestamp (just before or equal to the wire detection stamp)
        auto it = std::upper_bound(rgb_timestamps_.begin(), rgb_timestamps_.end(), wire_detection_stamp) - 1;

        if (it >= rgb_timestamps_.begin() && it < rgb_timestamps_.end())
        {
            size_t rgb_index = std::distance(rgb_timestamps_.begin(), it);
            double rgb_stamp = rgb_timestamps_[rgb_index];
            cv::Mat rgb_image = rgb_imgs_[rgb_index];

            // Draw Kalman filter visualization on the image
            publishKFsViz(rgb_image, rgb_stamp, wire_detections_msg.wire_detections);
        }
    }

    if (wire_viz_3d_)
    {
        visualize3DWires();
    }
}

void WireTrackingNode::targetTimerCallback()
{
    if (!initialized_)
        return;
    if (total_iterations_ < iteration_start_threshold_)
    {
        return;
    }
    if (tracked_wire_id_ == -1)
    {
        tracked_wire_id_ = position_kalman_filters.get_target_id();
        if (tracked_wire_id_ == -1)
        {
            RCLCPP_WARN(this->get_logger(), "No valid wire Kalman filters found.");
            return;
        }
    }
    auto [wire_dh, kf_index] = position_kalman_filters.get_kf_by_id(tracked_wire_id_);
    if (kf_index == -1)
    {
        RCLCPP_WARN(this->get_logger(), "No valid wire Kalman filter found.");
        return;
    }

    Eigen::Vector3d wire_direction = direction_kalman_filter_.getDirection();
    Eigen::Vector3d kf_xyz = position_kalman_filters_.getKfXYZs(direction_kalman_filter_.getYaw(), {kf_index});
    Eigen::Vector3d start_point = kf_xyz + 20.0 * wire_direction;
    Eigen::Vector3d end_point = kf_xyz - 20.0 * wire_direction;

    Eigen::Vector3d start_proj = camera_matrix_ * start_point;
    Eigen::Vector3d end_proj = camera_matrix_ * end_point;
    Eigen::Vector3d kf_proj = camera_matrix_ * kf_xyz;

    cv::Point start_point_2d(start_proj(0) / start_proj(2), start_proj(1) / start_proj(2));
    cv::Point end_point_2d(end_proj(0) / end_proj(2), end_proj(1) / end_proj(2));
    std::tuple<int, int, double> kf_point_2d(
        static_cast<int>(kf_proj(0) / kf_proj(2)),
        static_cast<int>(kf_proj(1) / kf_proj(2)),
        kf_dh[1] // Assuming kf_dh is something like Eigen::Vector2d
    );

    // Calculate pitch
    double dz = start_point(2) - end_point(2);
    double dxdy = (start_point_2d - end_point_2d).norm();
    double pitch = std::atan2(dz, dxdy);

    // Image plane angle theta (from end to start)
    double theta = std::atan2(
        start_point_2d.y - end_point_2d.y,
        start_point_2d.x - end_point_2d.x);

    // Distance of midpoint from origin
    cv::Point2d mid_point = 0.5 * (start_point_2d + end_point_2d);
    double rho = std::hypot(mid_point.x, mid_point.y);

    wire_interfaces::msg::WireTarget target_msg;
    target_msg.header.stamp = this->now();
    target_msg.header.frame_id = "wire_cam";
    target_msg.position.x = kf_point_2d.first;
    target_msg.position.y = kf_point_2d.second;
    target_msg.position.z = kf_point_2d.third; // Assuming kf_dh[1] is the height
    target_msg.target_angle = theta;
    target_msg.target_distance = rho;
    target_msg.target_pitch = pitch;

    wire_target_pub_->publish(target_msg);
}

void WireTrackingNode::visualize3DWires()
{
    if (!position_kalman_filters_ || !initialized_)
    {
        return;
    }

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "/map";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "lines";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;

    float line_scale = 0.05f;
    marker.scale.x = line_scale;
    marker.scale.y = line_scale;
    marker.scale.z = line_scale;

    std::vector<int> kf_ids = position_kalman_filters_->getKFIDs();
    Eigen::MatrixXd kf_xyzs = position_kalman_filters_->getKFXYZs(direction_kalman_filter_->getYaw());

    if (kf_xyzs.rows() != static_cast<int>(kf_ids.size()))
    {
        RCLCPP_ERROR(this->get_logger(), "Kalman filter IDs and XYZs do not match in length.");
        return;
    }

    Eigen::Vector3d wire_direction = direction_kalman_filter_->getDirection();
    Eigen::MatrixXd start_points = kf_xyzs.rowwise() + wire_direction.transpose();
    Eigen::MatrixXd end_points = kf_xyzs.rowwise() - wire_direction.transpose();

    for (size_t i = 0; i < kf_ids.size(); ++i)
    {
        Eigen::Vector3d start = start_points.row(i);
        Eigen::Vector3d end = end_points.row(i);

        geometry_msgs::msg::Point p1, p2;
        p1.x = start(0);
        p1.y = start(1);
        p1.z = start(2);
        p2.x = end(0);
        p2.y = end(1);
        p2.z = end(2);

        marker.points.push_back(p1);
        marker.points.push_back(p2);

        std_msgs::msg::ColorRGBA color;
        if (kf_ids[i] == tracked_wire_id_)
        {
            color.r = 0.0f;
            color.g = 1.0f;
            color.b = 0.0f;
            color.a = 1.0f;
        }
        else
        {
            color.r = 0.0f;
            color.g = 0.0f;
            color.b = 1.0f;
            color.a = 1.0f;
        }
        marker.colors.push_back(color);
        marker.colors.push_back(color);
    }

    tracking_3d_pub_->publish(marker);
}

void WireTrackingNode::publishKFsViz(cv::Mat img, double stamp, const std::vector<WireDetection> *wire_detections = nullptr)
{
    // Filter valid KF indices
    Eigen::VectorXi valid_counts = position_kalman_filters_->valid_counts();
    std::vector<int> valid_kf_ids;
    for (int i = 0; i < valid_counts.size(); ++i)
    {
        if (valid_counts(i) > min_valid_kf_count_threshold_)
        {
            valid_kf_ids.push_back(i);
        }
    }

    if (!valid_kf_ids.empty())
    {
        // Gather valid KF colors and positions
        Eigen::MatrixXf valid_colors(valid_kf_ids.size(), 3);
        for (size_t i = 0; i < valid_kf_ids.size(); ++i)
        {
            valid_colors.row(i) = position_kalman_filters_->kf_colors().row(valid_kf_ids[i]);
        }

        Eigen::Vector3f wire_direction_estimate = direction_kalman_filter_->getDirection().cast<float>();
        float wire_yaw = direction_kalman_filter_->getYaw();

        Eigen::MatrixXf valid_xyz_points = position_kalman_filters_->getKFXYZs(wire_yaw, valid_kf_ids);

        Eigen::MatrixXf start_points = valid_xyz_points.rowwise() + 20 * wire_direction_estimate.transpose();
        Eigen::MatrixXf end_points = valid_xyz_points.rowwise() - 20 * wire_direction_estimate.transpose();

        Eigen::MatrixXf image_start_points = camera_matrix_ * start_points.transpose();
        Eigen::MatrixXf image_end_points = camera_matrix_ * end_points.transpose();

        int target_kf_idx = -1;
        const auto &kf_ids = position_kalman_filters_->kf_ids();
        for (size_t i = 0; i < kf_ids.size(); ++i)
        {
            if (kf_ids[i] == tracked_wire_id_)
            {
                target_kf_idx = static_cast<int>(i);
                break;
            }
        }

        for (size_t i = 0; i < valid_kf_ids.size(); ++i)
        {
            Eigen::Vector3f start_img = image_start_points.col(i);
            Eigen::Vector3f end_img = image_end_points.col(i);

            cv::Point start_pt(start_img(0) / start_img(2), start_img(1) / start_img(2));
            cv::Point end_pt(end_img(0) / end_img(2), end_img(1) / end_img(2));

            cv::Scalar color(valid_colors(i, 0), valid_colors(i, 1), valid_colors(i, 2));
            cv::line(img, start_pt, end_pt, color, 2);

            cv::Point center((start_pt.x + end_pt.x) / 2, (start_pt.y + end_pt.y) / 2);
            cv::circle(img, center, 5, color, -1);

            if (valid_kf_ids[i] == target_kf_idx)
            {
                cv::circle(img, center, 10, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    if (wire_detections != nullptr)
    {
        for (const auto &wire_detection : *wire_detections)
        {
            Eigen::Vector3f start_vec(wire_detection.start.x, wire_detection.start.y, wire_detection.start.z);
            Eigen::Vector3f end_vec(wire_detection.end.x, wire_detection.end.y, wire_detection.end.z);

            Eigen::Vector3f img_start = camera_matrix_ * start_vec;
            Eigen::Vector3f img_end = camera_matrix_ * end_vec;

            cv::Point pt_start(img_start(0) / img_start(2), img_start(1) / img_start(2));
            cv::Point pt_end(img_end(0) / img_end(2), img_end(1) / img_end(2));
            cv::line(img, pt_start, pt_end, cv::Scalar(255, 0, 0), 2);
        }
    }

    sensor_msgs::msg::Image img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();

    img_msg.header.stamp.sec = static_cast<int32_t>(stamp);
    img_msg.header.stamp.nanosec = static_cast<uint32_t>((stamp - static_cast<int32_t>(stamp)) * 1e9);

    tracking_2d_pub_->publish(img_msg);
}

void WireTrackingNode::rgbCallback(const sensor_msgs::msg::Image::SharedPtr rgb_msg)
{
    if (!initialized_)
    {
        return;
    }

    // Convert timestamp to double (seconds)
    double stamp = rgb_msg->header.stamp.sec + rgb_msg->header.stamp.nanosec * 1e-9;

    // Convert ROS image to OpenCV image (BGR)
    cv::Mat rgb;
    try
    {
        rgb = cv_bridge::toCvCopy(rgb_msg, "bgr8")->image;
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Maintain only the latest 10 timestamps/images
    if (rgb_timestamps_.size() >= 10)
    {
        rgb_timestamps_.erase(rgb_timestamps_.begin());
        rgb_imgs_.erase(rgb_imgs_.begin());
    }

    // Insert while keeping the timestamps sorted
    rgb_timestamps_.push_back(stamp);
    rgb_imgs_.push_back(rgb);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<WireTrackingNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
