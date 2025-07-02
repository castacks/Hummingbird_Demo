#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <vector>
#include <utility>

class DirectionKalmanFilter {
public:
    DirectionKalmanFilter(const YAML::Node& config);

    void initialize(const Eigen::Vector3d& direction0);

    double getYaw() const;

    double predict(const Eigen::Matrix3d& relative_rotation);

    void update(const Eigen::Vector3d& measured_direction);

    Eigen::Vector3d getDirectionFromLineEndPoints(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end,
    const Eigen::Vector3d* reference_direction = nullptr) const;

    Eigen::Vector3d getDirection() const;

    bool isInitialized() const { return initialized_; }

private:
    double yaw_predict_covariance_;
    double yaw_measurement_covariance_;
    double yaw_max_covariance_;

    double Q_val_;
    double R_val_;
    double max_yaw_covariance_;
    double initial_cov_multiplier_;

    Eigen::Matrix3d Q_;  // Process noise covariance matrix
    Eigen::Matrix3d R_;  // Measurement noise covariance matrix
    Eigen::Matrix3d P_init_;  // Initial covariance matrix

    bool initialized_;
    Eigen::Vector3d curr_direction_;  // Current direction estimate
    Eigen::Matrix3d P_;  // State covariance matrix

};
