#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <vector>

#include "wire_tracking/direction_kf.h"

DirectionKalmanFilter::DirectionKalmanFilter(const YAML::Node &config)
{
    try
    {
        yaw_predict_covariance_ = config["yaw_predict_covariance"].as<double>();
        yaw_measurement_covariance_ = config["yaw_measurement_covariance"].as<double>();
        double yaw_max_cov = config["yaw_max_covariance"].as<double>();
        yaw_max_covariance_ = yaw_max_cov * yaw_max_cov; // Store squared value for internal use
        Q_val_ = yaw_predict_covariance_ * yaw_predict_covariance_;
        R_val_ = yaw_measurement_covariance_ * yaw_measurement_covariance_;
        initial_cov_multiplier_ = config["initial_covariance_multiplier"].as<double>();

        initialized_ = false;
        curr_direction_ = Eigen::Vector3d::Zero();

        Q_ = Eigen::Matrix3d::Identity() * Q_val_;
        R_ = Eigen::Matrix3d::Identity() * R_val_;
        P_init_ = R_ * initial_cov_multiplier_;
    }
    catch (const YAML::Exception &e)
    {
        throw std::runtime_error("Error parsing DirectionKalmanFilters configuration: " + std::string(e.what()));
    }
}

void DirectionKalmanFilter::initialize(const Eigen::Vector3d &direction0)
{
    // Sanity check
    assert(direction0.size() == 3 && "direction0 must have 3 elements (vx, vy, vz)");

    // Set the initial direction estimate
    curr_direction_ = direction0;

    // Reset the covariance matrix P = initial_cov_multiplier_ * R
    P_ = initial_cov_multiplier_ * R_;

    // Mark as initialized
    initialized_ = true;
}

double DirectionKalmanFilter::predict(const Eigen::Matrix3d &relative_rotation)
{
    assert(relative_rotation.rows() == 3 && relative_rotation.cols() == 3 && "relative_rotation must be 3x3");

    // Predict new direction
    curr_direction_ = relative_rotation * curr_direction_;
    curr_direction_.normalize(); // Normalize the vector

    // Predict new covariance
    P_ = relative_rotation * P_ * relative_rotation.transpose() + Q_;

    return getYaw(); // Return predicted yaw
}

void DirectionKalmanFilter::update(const Eigen::Vector3d &measured_direction_input)
{
    Eigen::Vector3d measured_direction = measured_direction_input.normalized(); // Ensure unit vector

    Eigen::Matrix3d S = P_ + R_;
    Eigen::Matrix3d K = P_ * S.inverse(); // Kalman gain

    Eigen::Vector3d y = measured_direction - curr_direction_; // Innovation

    curr_direction_ = curr_direction_ + K * y;
    curr_direction_.normalize(); // Normalize updated direction

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    P_ = (I - K) * P_;
}

Eigen::Vector3d DirectionKalmanFilter::getDirectionFromLineEndPoints(
    const Eigen::Vector3d &start,
    const Eigen::Vector3d &end,
    const Eigen::Vector3d *reference_direction) const
{
    // Check input shapes (Eigen vectors already have fixed size)
    assert(start.size() == 3 && end.size() == 3 && "start and end must be 3D points");

    Eigen::Vector3d direction = end - start;
    direction.normalize();

    // Use current direction if initialized
    if (initialized_)
    {
        if (direction.dot(curr_direction_) < 0)
        {
            direction = -direction;
        }
    }
    else if (reference_direction != nullptr)
    {
        // No current direction, but reference provided
        if (direction.dot(*reference_direction) < 0)
        {
            direction = -direction;
        }
    }

    return direction;
}

double DirectionKalmanFilter::getYaw() const
{
    return std::atan2(curr_direction_(1), curr_direction_(0));
}

Eigen::Vector3d DirectionKalmanFilter::getDirection() const
{
    if (!initialized_)
    {
        throw std::runtime_error("Kalman filter is not initialized. Call initialize() first.");
    }
    return curr_direction_; // Already an Eigen::Vector3d (3x1 vector)
}