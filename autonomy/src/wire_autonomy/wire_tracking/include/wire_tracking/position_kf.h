#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <vector>
#include <utility>

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
    int max_kf_id_;
    int target_kf_id_;

    Eigen::Matrix<int, 1, Eigen::Dynamic> kf_ids_;       // 1×N
    Eigen::Matrix2Xd kf_points_;                         // 2×N
    Eigen::Matrix2Xd kf_Ps_flat_;                        // (2*2)×N  each column is P_init_.reshaped()
    Eigen::Matrix3Xd kf_colors_;                         // 3×N
    Eigen::Matrix<int, 1, Eigen::Dynamic> valid_counts_; // 1×N

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
    static std::vector<cv::Vec3b> generateVizColor(int num_colors = 1);

public:
    void initializeKFs(const Eigen::Matrix3Xd &camera_points, double wire_yaw);
    static Eigen::Matrix2Xd getDHFromXYZs(const Eigen::Matrix3Xd &xyz_points, double wire_yaw);
    Eigen::Matrix3Xd getKFXYZs(double wire_yaw, const std::vector<int> &inds = {}) const;
    void predict(const Eigen::Matrix4d &relative_H, double yaw_prev, double yaw_curr);
    void update(const Eigen::Matrix3Xd &cam_points, double wire_yaw);
    std::pair<Eigen::Vector2d, int> getKFByID(int kf_id) const;
    int getTargetID();
};
