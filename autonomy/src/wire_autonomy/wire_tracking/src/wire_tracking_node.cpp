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
    // Init the position Kalman filter
    position_kalman_filters_ = PositionKalmanFilters(config_, camera_matrix_, {image_width_, image_height_});
    initialized_ = true;
    camera_info_sub_.reset();
    RCLCPP_INFO(this->get_logger(), "Wire Tracking Node initialized, Camera info received.");
}

void WireTrackingNode::poseCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (!initialized_)
        return;
    // if last relative transform is not set, initialize it
    if (previous_relative_transform_.isZero())
    {
        previous_relative_transform_ = poseToHomogeneous(msg->pose.pose);
    }
    else
    {
        // Compute the relative transform from the previous pose to the current pose
        auto [H_relative_in_wire_cam, H_to_pose] = getRelativeTransformInAnotherFrame(
            H_pose_to_wire_, previous_relative_transform_, msg->pose.pose);
        // Update the previous relative transform
        double stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        relative_transform_timestamps_.push_back(stamp);
        relative_transforms_.push_back(H_relative_in_wire_cam);
        previous_relative_transform_ = H_to_pose;
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

    // Iterate over poses up to idx
    for (int i = 0; i < idx; ++i)
    {
        const Eigen::Matrix4d &relative_pose_transform = relative_transforms_[i];

        if (direction_kalman_filter_->isInitialized())
        {
            double previous_yaw = direction_kalman_filter_->getYaw();
            double curr_yaw = direction_kalman_filter_->predict(relative_pose_transform.block<3, 3>(0, 0));

            if (position_kalman_filters_->isInitialized())
            {
                position_kalman_filters_->predict(relative_pose_transform, previous_yaw, curr_yaw);
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
    Eigen::Vector3d avg_dir = wire_directions.colwise().mean();
    avg_dir.normalize();

    // update or initialize your direction filter
    if (direction_kalman_filter_->isInitialized())
    {
        RCLCPP_INFO(this->get_logger(), "Updating Direction Kalman Filter with new average direction.");
        direction_kalman_filter_->update(avg_dir);
        double measured_yaw = std::atan2(avg_dir.y(), avg_dir.x()) * 180.0 / M_PI;
        double current_yaw = direction_kalman_filter_->getYaw() * 180.0 / M_PI;
        RCLCPP_INFO(this->get_logger(), "Direction filter updated with yaw: %.2f degrees (measured: %.2f degrees)",
                    current_yaw, measured_yaw);
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Direction Kalman Filter with new average direction.");
        direction_kalman_filter_->initialize(avg_dir);
        RCLCPP_INFO(this->get_logger(), "Direction filter initialized with yaw: %.2f degrees",
                    direction_kalman_filter_->getYaw() * 180.0 / M_PI);
    }

    // update or initialize your position filters
    double yaw = direction_kalman_filter_->getYaw();
    if (position_kalman_filters_->isInitialized())
    {
        RCLCPP_INFO(this->get_logger(), "Updating Position Kalman Filters with new wire locations.");
        position_kalman_filters_->update(wire_points_xyz, yaw);
        RCLCPP_INFO(this->get_logger(), "Position filters updated with %d Kalman filters.", position_kalman_filters_->getNumKFs());
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Position Filters with new wire locations.");
        position_kalman_filters_->initializeKFs(wire_points_xyz, yaw);
        relative_transforms_.clear();
        relative_transform_timestamps_.clear();
        RCLCPP_INFO(this->get_logger(), "Position filters initialized with %d Kalman filters.",
                    position_kalman_filters_->getNumKFs());
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
            cv::Mat rgb_image = rgb_images_[rgb_index];

            // Draw Kalman filter visualization on the image
            publishKFsViz(rgb_image, rgb_stamp, &msg->wire_detections);
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

    Eigen::Vector3d wire_direction = direction_kalman_filter_->getDirection();
    Eigen::Vector3d kf_xyz = position_kalman_filters_->getKFXYZs(direction_kalman_filter_->getYaw(), {kf_index});
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
        kf_xyz[2] // Assuming kf_xyz is in the form [x, y, z], and the z is the height away from camera
    );

    // Calculate pitch
    double dz = start_point(2) - end_point(2);
    double dxdy = cv::norm(start_point_2d - end_point_2d);
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
    target_msg.target_image_x = std::get<0>(kf_point_2d);
    target_msg.target_image_y = std::get<1>(kf_point_2d);
    target_msg.target_height = std::get<2>(kf_point_2d); // Assuming kf_dh[1] is the height
    target_msg.target_angle = theta;
    target_msg.target_distance = rho;
    target_msg.target_pitch = pitch;

    target_pub_->publish(target_msg);
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

    Eigen::MatrixXd kf_xyzs = position_kalman_filters_->getKFXYZs(direction_kalman_filter_->getYaw());

    Eigen::Vector3d wire_direction = direction_kalman_filter_->getDirection();
    Eigen::MatrixXd start_points = kf_xyzs.rowwise() + wire_direction.transpose();
    Eigen::MatrixXd end_points = kf_xyzs.rowwise() - wire_direction.transpose();
    
    bool mark_target = false;
    int kf_index = -1;
    if (tracked_wire_id_ != -1)
    {
        auto [wire_dh, kf_index] = position_kalman_filters_->getKFByID(tracked_wire_id_);
        mark_target = true;
    }

    for (size_t i = 0; i < kf_xyzs.rows(); ++i)
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
        if (mark_target && i == kf_index)
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

void WireTrackingNode::publishKFsViz(cv::Mat img, double stamp, const std::vector<wire_interfaces::msg::WireDetection> *wire_detections = nullptr)
{
    // Filter valid KF indices
    std::vector<int> valid_kf_indices = position_kalman_filters_->getValidKFIndices();

    if (!valid_kf_indices.empty())
    {
        Eigen::Vector3d wire_direction_estimate = direction_kalman_filter_->getDirection();
        float wire_yaw = direction_kalman_filter_->getYaw();

        Eigen::MatrixXd valid_xyz_points = position_kalman_filters_->getKFXYZs(wire_yaw, valid_kf_indices);

        Eigen::MatrixXd start_points = valid_xyz_points.rowwise() + 20 * wire_direction_estimate.transpose();
        Eigen::MatrixXd end_points = valid_xyz_points.rowwise() - 20 * wire_direction_estimate.transpose();

        Eigen::MatrixXd image_start_points = camera_matrix_ * start_points.transpose();
        Eigen::MatrixXd image_end_points = camera_matrix_ * end_points.transpose();

        int tracked_wire_id_ = position_kalman_filters_->getTargetID();
        auto [wire_dh, target_kf_idx] = position_kalman_filters_->getKFByID(tracked_wire_id_);
        for (size_t i = 0; i < valid_kf_indices.size(); ++i)
        {
            Eigen::Vector3d start_img = image_start_points.col(i);
            Eigen::Vector3d end_img = image_end_points.col(i);

            cv::Point start_pt(start_img(0) / start_img(2), start_img(1) / start_img(2));
            cv::Point end_pt(end_img(0) / end_img(2), end_img(1) / end_img(2));

            Eigen::Vector3d color_vec = position_kalman_filters_->getKFColor(valid_kf_indices[i]);
            cv::Scalar color(color_vec(0), color_vec(1), color_vec(2));
            cv::line(img, start_pt, end_pt, color, 2);

            cv::Point center((start_pt.x + end_pt.x) / 2, (start_pt.y + end_pt.y) / 2);
            cv::circle(img, center, 5, color, -1);

            if (valid_kf_indices[i] == target_kf_idx)
            {
                cv::circle(img, center, 10, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    if (wire_detections != nullptr)
    {
        for (const auto &wire_detection : *wire_detections)
        {
            Eigen::Vector3d start_vec(wire_detection.start.x, wire_detection.start.y, wire_detection.start.z);
            Eigen::Vector3d end_vec(wire_detection.end.x, wire_detection.end.y, wire_detection.end.z);

            Eigen::Vector3d img_start = camera_matrix_ * start_vec;
            Eigen::Vector3d img_end = camera_matrix_ * end_vec;

            cv::Point pt_start(img_start(0) / img_start(2), img_start(1) / img_start(2));
            cv::Point pt_end(img_end(0) / img_end(2), img_end(1) / img_end(2));
            cv::line(img, pt_start, pt_end, cv::Scalar(255, 0, 0), 2);
        }
    }

    sensor_msgs::msg::Image::SharedPtr img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();

    img_msg->header.stamp.sec = static_cast<int32_t>(stamp);
    img_msg->header.stamp.nanosec = static_cast<uint32_t>((stamp - static_cast<int32_t>(stamp)) * 1e9);

    tracking_2d_pub_->publish(*img_msg);
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
