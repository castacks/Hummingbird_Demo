#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

class PositionKalmanFilters
{
public:
    /**
     * Constructor: Initializes Kalman filter parameters from YAML.
     *
     * @param config YAML::Node containing wire tracking configuration.
     * @param camera_intrinsics 3x3 camera intrinsic matrix.
     * @param image_size Pair of (width, height).
     */
    PositionKalmanFilters(const YAML::Node &config,
                          const Eigen::Matrix3d &camera_intrinsics,
                          const std::pair<int, int> &image_size);

private:
    // Config values (squared for internal use)
    double z_predict_cov_;
    double z_measurement_cov_;
    double z_max_covariance_;

    double y_predict_cov_;
    double y_measurement_cov_;
    double y_max_covariance_;

    double initial_cov_multiplier_;
    double wire_matching_min_threshold_m_;
    int valid_count_buffer_;
    int min_valid_kf_count_threshold_;

    // State tracking
    bool initialized_;
    int max_kf_id_ = 0; // Maximum Kalman filter ID assigned so far
    int target_kf_id_;

    Eigen::VectorXi kf_ids_;       // 1×N
    Eigen::Matrix2Xd kf_points_;                         // 2×N
    Eigen::MatrixXd kf_covariances_;                     // 4 × M, vec(P) in each col
    Eigen::Matrix3Xd kf_colors_;                         // 3×N
    Eigen::VectorXi valid_counts_; // 

    // Kalman filter matrices
    Eigen::Matrix2d Q_;      // Process noise covariance
    Eigen::Matrix2d R_;      // Measurement noise covariance
    Eigen::Matrix2d P_init_; // Initial estimate covariance

    // Camera properties
    Eigen::Matrix3d camera_intrinsics_;
    std::pair<int, int> image_size_;

    void addKFs(const Eigen::Matrix2Xd &dhs0);
    int removeStaleKFs();
    std::vector<bool> checkKFsInFrame(const std::vector<int> &kf_indices, double camera_yaw) const;
    static Eigen::Matrix3Xd generateVizColor(int num_colors = 1);

public:
    void initializeKFs(const Eigen::Matrix3Xd &camera_points, double wire_yaw);
    static Eigen::Matrix2Xd getDHFromXYZs(const Eigen::Matrix3Xd &xyz_points, double wire_yaw);
    Eigen::Matrix3Xd getKFXYZs(double wire_yaw, const std::vector<int> &inds = {}) const;
    void predict(const Eigen::Matrix4d &relative_H, double yaw_prev, double yaw_curr);
    void update(const Eigen::Matrix3Xd &cam_points, double wire_yaw);
    std::pair<Eigen::Vector2d, int> getKFByID(int kf_id) const;
    int getTargetID();
    std::vector<int> getValidKFIndices() const;
    Eigen::VectorXd getKFColor(int kf_index) const;
    bool isInitialized() const { return initialized_; }
    int getNumKFs() const { return kf_points_.cols(); }
    Eigen::MatrixXd getKFCovariances() const { return kf_covariances_; }
    Eigen::VectorXi getKFIDs() const { return kf_ids_; }
    Eigen::Matrix2Xd getKFDHs() const { return kf_points_; }
    Eigen::VectorXi getValidCounts() const { return valid_counts_; }
};
