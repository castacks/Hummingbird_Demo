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

#include "wire_tracking/wire_tracking_node.h"
#include "wire_tracking/coord_transforms.h"
#include "wire_tracking/direction_kf.h"
#include "wire_tracking/position_kf.h"

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::placeholders::_1;

WireTrackingNode::WireTrackingNode() : rclcpp::Node("wire_tracking_node")
{
    // Load parameters and config
    loadConfig();

    // Initialize subscriptions
    this->declare_parameter<std::string>("camera_info_sub_topic", "/default/topic");
    this->get_parameter("camera_info_sub_topic", camera_info_sub_topic_);
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_sub_topic_, 1,
        std::bind(&WireTrackingNode::cameraInfoCallback, this, _1));

    this->declare_parameter<std::string>("pose_sub_topic", "/default/pose_topic");
    this->get_parameter("pose_sub_topic", pose_sub_topic_);
    pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        pose_sub_topic_, rclcpp::SensorDataQoS(),
        std::bind(&WireTrackingNode::poseCallback, this, _1));

    this->declare_parameter<std::string>("wire_detections_topic", "/default/wire_detections");
    this->get_parameter("wire_detections_topic", wire_detections_topic_);
    detection_sub_ = this->create_subscription<wire_interfaces::msg::WireDetections>(
        wire_detections_topic_, 1,
        std::bind(&WireTrackingNode::wireDetectionCallback, this, _1));

    this->declare_parameter<std::string>("rgb_image_sub_topic", "/default/rgb_image");
    this->get_parameter("rgb_image_sub_topic", rgb_image_sub_topic_);
    rgb_image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        rgb_image_sub_topic_, 1,
        std::bind(&WireTrackingNode::rgbCallback, this, _1));

    this->declare_parameter<std::string>("wire_target_topic", "/default/wire_target");
    this->get_parameter("wire_target_topic", wire_target_topic_);
    target_pub_ = this->create_publisher<wire_interfaces::msg::WireTarget>(wire_target_topic_, 10);
    target_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / config_["target_publish_frequency_hz"].as<double>()),
        std::bind(&WireTrackingNode::targetTimerCallback, this));

    this->declare_parameter<std::string>("tracking_2d_viz_topic", "/default/tracking_2d_viz_topic_");
    this->get_parameter("tracking_2d_viz_topic", tracking_2d_viz_topic_);
    tracking_2d_pub_ = this->create_publisher<sensor_msgs::msg::Image>(tracking_2d_viz_topic_, 10);

    this->declare_parameter<std::string>("tracking_3d_viz_topic", "/default/tracking_3d_viz_topic_");
    this->get_parameter("tracking_3d_viz_topic", tracking_3d_viz_topic_);
    tracking_3d_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(tracking_3d_viz_topic_, 10);

    iteration_start_threshold_ = config_["iteration_start_threshold"].as<int>();
    vtol_payload_ = config_["vtol_payload"].as<bool>();
    use_mavros_ = config_["use_mavros"].as<bool>();

    linear_translation_dropout_ = config_["linear_translation_dropout"].as<double>();
    angular_translation_dropout_ = config_["angular_translation_dropout"].as<double>();
    linear_velocity_dropout_ = config_["linear_velocity_dropout"].as<double>();
    angular_velocity_dropout_ = config_["angular_velocity_dropout"].as<double>();
    linear_acceleration_dropout_ = config_["linear_acceleration_dropout"].as<double>();
    angular_acceleration_dropout_ = config_["angular_acceleration_dropout"].as<double>();

    // Precompute transforms
    double baseline = config_["zed_baseline"].as<double>();
    H_pose_to_cam_.setIdentity();
    if (!vtol_payload_) // zeds are in same orientation
    {
        Matrix3d Ry;
        Ry = Eigen::AngleAxisd(M_PI, Vector3d::UnitY());
        H_pose_to_cam_.topLeftCorner<3, 3>() = Ry; // Only rotation around Y-axis
        H_pose_to_cam_.block<3, 1>(0, 3) = Vector3d(-baseline, 0.0, 0.18415);
    }
    else
    {
        if (!use_mavros_)
        {
            Matrix3d Rz;
            Rz = Eigen::AngleAxisd(M_PI / 2, Vector3d::UnitZ());
            Matrix3d Rx;
            Rx = Eigen::AngleAxisd(M_PI, Vector3d::UnitX());
            H_pose_to_cam_.topLeftCorner<3, 3>() = Rz * Rx;
            H_pose_to_cam_.block<3, 1>(0, 3) = Vector3d(-0.035, baseline / 2.0, 0.18415);
        }
        else // mavros uses FLU
        {
            Matrix3d Rz;
            Rz = Eigen::AngleAxisd(M_PI, Vector3d::UnitZ());
            H_pose_to_cam_.topLeftCorner<3, 3>() = Rz;
            H_pose_to_cam_.block<3, 1>(0, 3) = Vector3d(0.15875, 0.0, 0.0635)
        }
    }
    H_cam_to_pose_ = H_pose_to_cam_.inverse();

    // Populate Kalman filters
    direction_kalman_filter_ = DirectionKalmanFilter(config_);

    RCLCPP_INFO(this->get_logger(), "Wire Tracking Node initialized");
}

void WireTrackingNode::loadConfig()
{
    std::string pkg_share = ament_index_cpp::get_package_share_directory("wire_tracking");
    YAML::Node cfg = YAML::LoadFile(pkg_share + "/config/wire_tracking_config.yaml");
    config_ = cfg;

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
        if (wire_viz_)
        {
            wire_viz_3d_ = true;
            if (wire_mode_ == 2)
            {
                wire_viz_2d_ = true;
            }
        }
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
    line_length_ = std::max(image_height_, image_width_) * 1.5;
    // Init the position Kalman filter
    position_kalman_filters_ = PositionKalmanFilters(config_, camera_matrix_, {image_width_, image_height_});
    initialized_ = true;
    camera_info_sub_.reset();
    RCLCPP_INFO(this->get_logger(), "Wire Tracking Node initialized, Camera info received.");
}

void WireTrackingNode::poseCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (!initialized_)
    {
        return;
    }

    // if last relative transform is not set, initialize it
    if (previous_transform_.isZero())
    {
        previous_transform_ = poseToHomogeneous(msg->pose.pose);
        previous_transform_stamp_ = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    }
    else
    {
        Eigen::Matrix4d H_relative_in_wire_cam;
        Eigen::Matrix4d H_to_pose = poseToHomogeneous(msg->pose.pose);

        // Compute the relative transform from the previous pose to the current pose
        double curr_stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        double delta_time = curr_stamp - previous_transform_stamp_;
        double delta_translation = (H_to_pose.block<3, 1>(0, 3) - previous_transform_.block<3, 1>(0, 3)).norm();

        std::tie(H_relative_in_wire_cam, H_to_pose) = getRelativeTransformInAnotherFrame(H_pose_to_cam_, H_cam_tod, previous_transform_, msg->pose.pose);
        RCLCPP_INFO(this->get_logger(), "Received in bounds pose at timestamp: %.2f, with relative translation [%.2f, %.2f, %.2f]", curr_stamp, H_relative_in_wire_cam(0, 3), H_relative_in_wire_cam(1, 3), H_relative_in_wire_cam(2, 3));
        relative_transform_timestamps_.push_back(curr_stamp);
        relative_transforms_.push_back(H_relative_in_wire_cam);
        last_linear_velocity_ = curr_linear_velocity;
        last_angular_velocity_ = curr_angular_velocity;
        using_velocity_transform_ = false;

        previous_transform_ = H_to_pose;
        previous_transform_stamp_ = curr_stamp;
    }
}

void WireTrackingNode::predict_kfs_up_to_timestamp(double input_stamp)
{
    if (!initialized_)
    {
        RCLCPP_INFO(this->get_logger(), "Wire Tracking Node not initialized yet. Waiting for camera info.");
        return;
    }

    if (!direction_kalman_filter_ || !position_kalman_filters_)
    {
        RCLCPP_INFO(this->get_logger(), "Kalman filters not initialized yet. Waiting for measurment initialization.");
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
    else
    {
        RCLCPP_INFO(this->get_logger(), "Predicting Kalman filters up to timestamp: %.2f, using %d transforms.", input_stamp, idx + 1);
    }

    // Iterate over poses up to idx
    for (int i = 0; i <= idx; ++i)
    {
        const Eigen::Matrix4d &relative_pose_transform = relative_transforms_[i];
        RCLCPP_INFO_STREAM(this->get_logger(), "Predicting Transform: " << relative_pose_transform);

        if (direction_kalman_filter_->isInitialized())
        {
            double previous_yaw = direction_kalman_filter_->getYaw();
            double curr_yaw = direction_kalman_filter_->predict(relative_pose_transform.block<3, 3>(0, 0));

            if (position_kalman_filters_->isInitialized())
            {
                RCLCPP_INFO(this->get_logger(), "Linear translation of relative transform: [%.2f, %.2f, %.2f], yaw change: %.2f degrees",
                            relative_pose_transform(0, 3), relative_pose_transform(1, 3), relative_pose_transform(2, 3),
                            (curr_yaw - previous_yaw) * 180.0 / M_PI);
                position_kalman_filters_->predict(relative_pose_transform, previous_yaw, curr_yaw);
            }
        }
    }

    // Remove old transforms
    relative_transforms_.erase(relative_transforms_.begin(), relative_transforms_.begin() + idx + 1);
    relative_transform_timestamps_.erase(relative_transform_timestamps_.begin(), relative_transform_timestamps_.begin() + idx + 1);
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
        RCLCPP_INFO(this->get_logger(), "Kalman Debug After Predict:");
        debugPrintKFs();
    }

    if (msg->avg_angle > 0.0 && msg->avg_angle < M_PI && !msg->wire_detections.empty())
    {
        int N = static_cast<int>(msg->wire_detections.size());
        Eigen::Matrix3Xd wire_points_xyz(3, N);
        Eigen::Matrix3Xd wire_directions(3, N);

        Eigen::Vector3d reference_dir;

        for (std::size_t i = 0; i < N; ++i)
        {
            const auto &detection = msg->wire_detections[i];

            // start and end as Eigen vectors
            Eigen::Vector3d start(detection.start.x, detection.start.y, detection.start.z);
            Eigen::Vector3d end(detection.end.x, detection.end.y, detection.end.z);

            // midpoint
            wire_points_xyz.col(i) = 0.5 * (start + end);

            // direction from line endpoints (your C++ API)
            if (direction_kalman_filter_->isInitialized())
            {
                wire_directions.col(i) = direction_kalman_filter_->getDirectionFromLineEndPoints(start, end);
            }
            else if (i == 0)
            {
                reference_dir = direction_kalman_filter_->getDirectionFromLineEndPoints(start, end);
                wire_directions.col(i) = reference_dir;
            }
            else
            {
                wire_directions.col(i) = direction_kalman_filter_->getDirectionFromLineEndPoints(start, end, &reference_dir);
            }
        }

        // compute average direction and normalize
        Eigen::Vector3d avg_dir = wire_directions.rowwise().mean();
        avg_dir.normalize();
        if (std::isnan(avg_dir.x()) || std::isnan(avg_dir.y()) || std::isnan(avg_dir.z()))
        {
            RCLCPP_INFO(this->get_logger(), "Wire direction has a nan, printing %d directions", wire_directions.cols());
            for (int i = 0; i < wire_directions.cols(); ++i)
            {
                RCLCPP_INFO(this->get_logger(), "Direction %d: [%.6f, %.6f, %.6f]", i,
                            wire_directions(0, i), wire_directions(1, i), wire_directions(2, i));
            }
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Measured Average wire direction: [%.6f, %.6f, %.6f]", avg_dir.x(), avg_dir.y(), avg_dir.z());
        }

        // update or initialize your direction filter
        if (direction_kalman_filter_->isInitialized())
        {
            direction_kalman_filter_->update(avg_dir);
        }
        else
        {
            direction_kalman_filter_->initialize(avg_dir);
            RCLCPP_INFO(this->get_logger(), "Direction filter initialized with yaw: %.2f degrees",
                        direction_kalman_filter_->getYaw() * 180.0 / M_PI);
        }

        // update or initialize your position filters
        double yaw = direction_kalman_filter_->getYaw();
        if (position_kalman_filters_->isInitialized())
        {
            position_kalman_filters_->update(wire_points_xyz, yaw);
        }
        else
        {
            position_kalman_filters_->initializeKFs(wire_points_xyz, yaw);
            relative_transforms_.clear();
            relative_transform_timestamps_.clear();
            RCLCPP_INFO(this->get_logger(), "Position filters initialized with %d Kalman filters.",
                        position_kalman_filters_->getNumKFs());
        }
        total_iterations_++;
    }

    RCLCPP_INFO(this->get_logger(), "Kalman Debug After Update:");
    debugPrintKFs();

    if (wire_viz_2d_ || wire_viz_3d_)
    {
        if (!rgb_timestamps_.empty())
        {
            auto it = std::upper_bound(rgb_timestamps_.begin(), rgb_timestamps_.end(), wire_detection_stamp);
            if (it != rgb_timestamps_.begin())
            {
                --it; // Safe now
                size_t rgb_index = std::distance(rgb_timestamps_.begin(), it);
                double rgb_stamp = rgb_timestamps_[rgb_index];

                if (rgb_index < rgb_images_.size())
                {
                    cv::Mat rgb_image = rgb_images_[rgb_index];
                    visualizeWireTracking(rgb_image, rgb_stamp, &msg->wire_detections);
                }
            }
        }
    }
}

void WireTrackingNode::targetTimerCallback()
{
    // TODO: Check this logic of defining the target
    if (!initialized_)
        return;
    if (total_iterations_ < iteration_start_threshold_)
    {
        return;
    }
    if (tracked_wire_id_ == -1)
    {
        tracked_wire_id_ = position_kalman_filters_->getTargetID();
        if (tracked_wire_id_ == -1)
        {
            RCLCPP_WARN(this->get_logger(), "No valid wire Kalman filters found.");
            return;
        }
    }
    auto [wire_dh, kf_index] = position_kalman_filters_->getKFByID(tracked_wire_id_);
    if (kf_index == -1)
    {
        RCLCPP_WARN(this->get_logger(), "No valid wire Kalman filter found.");
        return;
    }

    double wire_yaw = direction_kalman_filter_->getYaw();
    Eigen::Vector3d kf_xyz = position_kalman_filters_->getKFXYZs(wire_yaw, {kf_index});

    wire_interfaces::msg::WireTarget target_msg;
    target_msg.header.stamp = this->now();
    target_msg.header.frame_id = "wire_cam";
    target_msg.target_x = kf_xyz(0);
    target_msg.target_y = kf_xyz(1);
    target_msg.target_z = kf_xyz(2);
    target_msg.target_yaw = wire_yaw;

    target_pub_->publish(target_msg);
}

void WireTrackingNode::visualizeWireTracking(cv::Mat img = cv::Mat(), double stamp = 0.0, const std::vector<wire_interfaces::msg::WireDetection> *wire_detections = nullptr)
{
    if (!initialized_ || !position_kalman_filters_->isInitialized())
    {
        return;
    }

    // Marker initialization
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "/map";
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "lines";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.05f;
    marker.scale.y = 0.05f;
    marker.scale.z = 0.05f;

    std::vector<int> valid_kf_indices = position_kalman_filters_->getValidKFIndices();
    if (!valid_kf_indices.empty())
    {
        Eigen::Vector3d wire_direction = direction_kalman_filter_->getDirection();
        float wire_yaw = direction_kalman_filter_->getYaw();
        Eigen::Matrix3Xd valid_xyzs = position_kalman_filters_->getKFXYZs(wire_yaw, valid_kf_indices);
        Eigen::Matrix3Xd start_points = valid_xyzs.colwise() + wire_direction;
        Eigen::Matrix3Xd end_points = valid_xyzs.colwise() - wire_direction;

        Eigen::Vector2d tracked_dh;
        int target_idx = -1;
        if (tracked_wire_id_ != -1)
        {
            std::tie(tracked_dh, target_idx) = position_kalman_filters_->getKFByID(tracked_wire_id_);
        }

        cv::Rect img_bounds(0, 0, img.cols, img.rows);

        for (size_t i = 0; i < valid_kf_indices.size(); ++i)
        {
            Eigen::Vector3d start = start_points.col(i);
            Eigen::Vector3d end = end_points.col(i);
            Eigen::Vector3d color_vec = position_kalman_filters_->getKFColor(valid_kf_indices[i]);

            // 3D Marker visualization
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
            Eigen::Vector3d color_marker = color_vec / 255.0; // Normalize to [0, 1]
            color.r = static_cast<float>(color_marker(2));
            color.g = static_cast<float>(color_marker(1));
            color.b = static_cast<float>(color_marker(0));
            color.a = 1.0f;
            marker.colors.push_back(color);
            marker.colors.push_back(color);

            if (img.empty())
            {
                continue; // Skip 2D visualization if no image is provided
            }
            // 2D Image projection
            Eigen::Vector3d center_img_h = camera_matrix_ * valid_xyzs.col(i);
            Eigen::Vector3d dir_img_h = camera_matrix_ * (valid_xyzs.col(i) + wire_direction);

            // Convert to 2D pixel coordinates
            Eigen::Vector2d p0(center_img_h(0) / center_img_h(2), center_img_h(1) / center_img_h(2));
            Eigen::Vector2d dir(dir_img_h(0) / dir_img_h(2), dir_img_h(1) / dir_img_h(2));
            dir = (dir - p0).normalized();

            // Extend the line far in both directions
            Eigen::Vector2d p1_px = p0 + line_length_ * dir;
            Eigen::Vector2d p2_px = p0 - line_length_ * dir;

            cv::Point start_px(static_cast<int>(p1_px.x()), static_cast<int>(p1_px.y()));
            cv::Point end_px(static_cast<int>(p2_px.x()), static_cast<int>(p2_px.y()));

            cv::Scalar cv_color(color_vec(0), color_vec(1), color_vec(2));

            if (cv::clipLine(img_bounds, start_px, end_px))
            {
                cv::line(img, start_px, end_px, cv_color, 2);
            }

            // Draw center circle
            cv::Point center_px(p0.x(), p0.y());
            cv::circle(img, center_px, 5, cv_color, -1);

            if (valid_kf_indices[i] == target_idx && tracked_wire_id_ != -1)
            {
                cv::circle(img, center_px, 10, cv::Scalar(0, 255, 0), 2);
            }

            tracking_3d_pub_->publish(marker);
        }
    }
    // Visualize detections if there are some
    if (wire_detections != nullptr && !img.empty())
    {
        for (const auto &wire_detection : *wire_detections)
        {
            Eigen::Vector3d start_vec(wire_detection.start.x, wire_detection.start.y, wire_detection.start.z);
            Eigen::Vector3d end_vec(wire_detection.end.x, wire_detection.end.y, wire_detection.end.z);

            Eigen::Vector3d img_start = camera_matrix_ * start_vec;
            Eigen::Vector3d img_end = camera_matrix_ * end_vec;

            cv::Point pt_start(img_start(0) / img_start(2), img_start(1) / img_start(2));
            cv::Point pt_end(img_end(0) / img_end(2), img_end(1) / img_end(2));
            cv::line(img, pt_start, pt_end, cv::Scalar(0, 0, 255), 1);
        }
    }
    if (!img.empty())
    {
        sensor_msgs::msg::Image::SharedPtr img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
        img_msg->header.stamp.sec = static_cast<int32_t>(stamp);
        img_msg->header.stamp.nanosec = static_cast<uint32_t>((stamp - static_cast<int32_t>(stamp)) * 1e9);
        tracking_2d_pub_->publish(*img_msg);
    }
    tracking_3d_pub_->publish(marker);
}

void WireTrackingNode::debugPrintKFs()
{
    if (!position_kalman_filters_->isInitialized() || !initialized_)
    {
        RCLCPP_INFO(this->get_logger(), "Position Kalman Filters not initialized.");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "DEBUG ----------------------------------------------------");
    if (using_velocity_transform_)
    {
        // throw std::runtime_error("Using velocity transform, stopping to see afftect");
        RCLCPP_INFO(this->get_logger(), "Using velocity transform for pose update !!!");
    }

    auto kf_direction = direction_kalman_filter_->getDirection();
    RCLCPP_INFO(this->get_logger(), "Direction: [%.6f, %.6f, %.6f], Yaw: %.6f degrees",
                kf_direction.x(), kf_direction.y(), kf_direction.z(),
                direction_kalman_filter_->getYaw() * 180.0 / M_PI);
    Eigen::VectorXi kf_ids = position_kalman_filters_->getKFIDs();
    Eigen::Matrix3Xd kf_points = position_kalman_filters_->getKFXYZs(direction_kalman_filter_->getYaw());
    Eigen::Matrix2Xd kf_dhs = position_kalman_filters_->getKFDHs();
    Eigen::VectorXi valid_counts = position_kalman_filters_->getValidCounts();
    for (size_t i = 0; i < kf_ids.size(); ++i)
    {
        RCLCPP_INFO(this->get_logger(), "KF ID: %d, DH: (%.4f, %.4f), Point: (%.4f, %.4f, %.4f), Valid Count: %d",
                    kf_ids[i],
                    kf_dhs(0, i), kf_dhs(1, i),
                    kf_points(0, i), kf_points(1, i), kf_points(2, i),
                    valid_counts[i]);
    }
    RCLCPP_INFO(this->get_logger(), "----------------------------------------------------");
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
        rgb_images_.erase(rgb_images_.begin());
    }

    // Insert while keeping the timestamps sorted
    rgb_timestamps_.push_back(stamp);
    rgb_images_.push_back(rgb);
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
