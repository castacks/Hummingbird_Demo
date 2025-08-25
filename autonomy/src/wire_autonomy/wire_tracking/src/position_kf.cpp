#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <vector>
#include <unsupported/Eigen/KroneckerProduct>
#include "rclcpp/rclcpp.hpp"

#include "wire_tracking/position_kf.h"

PositionKalmanFilters::PositionKalmanFilters(const YAML::Node &config,
                                             const Eigen::Matrix3d &camera_intrinsics,
                                             const std::pair<int, int> &image_size)
    : camera_intrinsics_(camera_intrinsics), image_size_(image_size)
{
    try
    {
        z_predict_cov_ = config["z_predict_covariance"].as<double>();
        z_measurement_cov_ = config["z_measurement_covariance"].as<double>();
        double z_max_cov = config["z_max_covariance"].as<double>();
        z_max_covariance_ = z_max_cov * z_max_cov; // Store squared value for internal use

        y_predict_cov_ = config["y_predict_covariance"].as<double>();
        y_measurement_cov_ = config["y_measurement_covariance"].as<double>();
        double y_max_cov = config["y_max_covariance"].as<double>();
        y_max_covariance_ = y_max_cov * y_max_cov; // Store squared value for internal use

        initial_cov_multiplier_ = config["initial_covariance_multiplier"].as<double>();
        wire_matching_min_threshold_m_ = config["wire_matching_min_threshold_m"].as<double>();
        valid_count_buffer_ = config["valid_count_buffer"].as<int>();
        min_valid_kf_count_threshold_ = config["min_valid_kf_count_threshold"].as<int>();

        initialized_ = false;
        max_kf_id_ = 0;
        target_kf_id_ = -1;

        Q_ = Eigen::Matrix2d::Zero();
        R_ = Eigen::Matrix2d::Zero();
        Q_(0, 0) = z_predict_cov_ * z_predict_cov_;
        Q_(1, 1) = y_predict_cov_ * y_predict_cov_;
        R_(0, 0) = z_measurement_cov_ * z_measurement_cov_;
        R_(1, 1) = y_measurement_cov_ * y_measurement_cov_;

        Eigen::Matrix2d P_init_matrix = R_ * initial_cov_multiplier_;
        P_init_ << P_init_matrix(0, 0), P_init_matrix(1, 0), P_init_matrix(0, 1), P_init_matrix(1, 1);
    }
    catch (const YAML::Exception &e)
    {
        throw std::runtime_error("Error parsing PositionKalmanFilters configuration: " + std::string(e.what()));
    }
}

void PositionKalmanFilters::addKFs(const Eigen::Matrix2Xd &dhs0)
{
    const int oldN = kf_points_.cols();
    const int newN = dhs0.cols();
    if (newN == 0)
        return;

    // 1) Resize all the matrices in one go:
    kf_ids_.conservativeResize(oldN + newN);
    kf_points_.conservativeResize(Eigen::NoChange, oldN + newN);
    kf_colors_.conservativeResize(Eigen::NoChange, oldN + newN);
    valid_counts_.conservativeResize(oldN + newN);

    // 2) Fill the new-id block with a linspace:
    kf_ids_.segment(oldN, newN) = Eigen::VectorXi::LinSpaced(newN, max_kf_id_, max_kf_id_ + newN - 1);

    // 3) Copy in the new points:
    kf_points_.block(0, oldN, 2, newN) = dhs0;

    // 4) Tile P_init_ into each new column:
    kf_covariances_.conservativeResize(4, oldN + newN);
    kf_covariances_.block(0, oldN, 4, newN) = P_init_.replicate(1, newN);

    // 5) Generate and copy colors in one call:
    Eigen::Matrix3Xd cols = generateVizColor(newN); // returns 3×newN Eigen::Matrix3Xd
    kf_colors_.block(0, oldN, 3, newN) = cols;

    // 6) Fill valid_counts_ with a constant:
    valid_counts_.segment(oldN, newN).setConstant(valid_count_buffer_);

    max_kf_id_ += newN;
}

void PositionKalmanFilters::initializeKFs(const Eigen::Matrix3Xd &camera_points, double wire_yaw)
{
    // 1) Validate input shape: must be 3×N
    if (camera_points.rows() != 3)
    {
        throw std::invalid_argument(
            "initializeKFs: camera_points must have 3 rows (got " + std::to_string(camera_points.rows()) + ")");
    }

    Eigen::Matrix2Xd dhs = getDHFromXYZs(camera_points, wire_yaw);

    // 3) Batch‑add these new filters
    addKFs(dhs);

    // 4) Mark initialized
    initialized_ = true;
}

void PositionKalmanFilters::predict(
    const Eigen::Matrix4d &relative_H,
    double yaw_prev,
    double yaw_curr)
{
    // 1) Validate
    if (relative_H.rows() != 4 || relative_H.cols() != 4)
    {
        throw std::invalid_argument("predict: relative_H must be 4×4");
    }
    // 2) Get current KF points in XYZ (3×N)
    Eigen::Matrix3Xd xyz = getKFXYZs(yaw_prev);

    // 3) Apply transform: R * xyz + t
    Eigen::Matrix3d R = relative_H.block<3, 3>(0, 0);
    Eigen::Vector3d t = relative_H.block<3, 1>(0, 3);
    Eigen::Matrix3Xd xyz_t = (R * xyz).colwise() + t;

    // 4) Reproject into (distance, height)
    kf_points_ = getDHFromXYZs(xyz_t, yaw_curr);

    // 5) Build the 2×2 Jacobian J
    double c1 = std::cos(yaw_prev), s1 = std::sin(yaw_prev);
    double c2 = std::cos(yaw_curr), s2 = std::sin(yaw_curr);

    Eigen::Matrix2d J;
    J(0, 0) = s2 * (R(0, 0) * s1 - R(0, 1) * c1) + c2 * (-R(1, 0) * s1 + R(1, 1) * c1);
    J(0, 1) = c2 * R(1, 2) - s2 * R(0, 2);
    J(1, 0) = -s1 * R(2, 0) + c1 * R(2, 1);
    J(1, 1) = R(2, 2);

    // 6) Update each covariance: P_new = J * P_old * Jᵀ + Q_
    //    We store covariances as a 4×N flat matrix kf_covariances
    Eigen::Matrix<double, 4, 4> KJJ =
        Eigen::kroneckerProduct(J, J);

    // Flatten Q_ (2×2) column‑wise into a 4×1 vector
    Eigen::Vector4d q_flat;
    q_flat << Q_(0, 0), Q_(1, 0),
        Q_(0, 1), Q_(1, 1);

    // Now kf_covariances is a (4 × N) matrix, each column is vec(P_old).
    // We can do all at once:
    kf_covariances_ = (KJJ * kf_covariances_).colwise() + q_flat;
}

void PositionKalmanFilters::update(const Eigen::Matrix3Xd &cam_points, double wire_yaw)
{
    if (cam_points.rows() != 3)
    {
        throw std::invalid_argument(
            "update: cam_points must be 3×M");
    }
    int N = cam_points.cols();
    int M = kf_points_.cols();

    if (N == 0)
    {
        std::cout << "No camera points to update Kalman filters.\n";
        return;
    }

    if (!initialized_)
    {
        throw std::runtime_error("PositionKalmanFilters not initialized. Call initializeKFs first.");
    }

    if (valid_counts_.size() != kf_points_.cols() || kf_ids_.size() != kf_points_.cols())
    {
        throw std::runtime_error("valid_counts size mismatch with KF points");
    }

    // Convert cam_points (3 x N) to dhs_measured (2 x N)
    Eigen::Matrix2Xd dhs_measured = getDHFromXYZs(cam_points, wire_yaw);

    // using Euclidean distance formula:
    // dists_sq = ||a||² + ||b||² - 2 * aᵀ * b
    // where a = dhs_measured, b = kf_points_
    Eigen::VectorXd dh_norms = dhs_measured.colwise().squaredNorm(); // 1 x N vector
    Eigen::VectorXd kf_norms = kf_points_.colwise().squaredNorm();   // 1 x M vector

    // Compute squared distance matrix using broadcasting
    Eigen::MatrixXd dists_sq = dh_norms.replicate(1, M)                         // N x M
                               + kf_norms.transpose().replicate(N, 1)           // N x M
                               - 2.0 * (dhs_measured.transpose() * kf_points_); // N x M

    // Clamp any negative values to zero (due to floating-point errors)
    dists_sq = dists_sq.array().max(0.0);

    // Final Euclidean distances
    Eigen::MatrixXd comp_dists = dists_sq.array().sqrt();

    // dists_masked: boolean N x M, true if distance < threshold
    Eigen::ArrayXX<bool> dists_masked = (comp_dists.array() < wire_matching_min_threshold_m_);

    // Find unmatched kfs (no measured point matched to them)
    Eigen::Array<bool, 1, Eigen::Dynamic> any_matches = dists_masked.colwise().any(); // 1 x M

    // Step 2: Find indices where any_matches is false (no matches)
    std::vector<int> unmatched_kfs;
    for (int j = 0; j < any_matches.size(); ++j)
    {
        if (!any_matches(j))
        {
            unmatched_kfs.push_back(j);
        }
    }

    if (!unmatched_kfs.empty())
    {
        std::vector<bool> kfs_in_frame = checkKFsInFrame(unmatched_kfs, wire_yaw);
        for (size_t idx = 0; idx < unmatched_kfs.size(); ++idx)
        {
            if (kfs_in_frame[idx])
            {
                int kf_idx = unmatched_kfs[idx];
                valid_counts_[kf_idx]--;
            }
        }
    }

    Eigen::Matrix2Xd dhs_to_add(2, 0);
    for (int i = 0; i < N; ++i)
    {
        // Find matched KF indices for this measured point
        std::vector<int> matched_indices;
        for (int j = 0; j < M; ++j)
        {
            if (dists_masked(i, j))
            {
                matched_indices.push_back(j);
            }
        }

        if (matched_indices.size() >= 1)
        {
            if (matched_indices.size() != 1) // More than one match found for the measured point
            {
                // Pick the closest KF point
                double min_dist = std::numeric_limits<double>::max();
                int closest_index = -1;
                for (int j : matched_indices)
                {
                    double dist = comp_dists(i, j);
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        closest_index = j;
                    }
                }
                // Update the closest KF point
                matched_indices.clear();
                matched_indices.push_back(closest_index);
            }
            int matched_index = matched_indices[0];

            Eigen::Matrix2d P;
            P(0, 0) = kf_covariances_(0, matched_index);
            P(1, 0) = kf_covariances_(1, matched_index);
            P(0, 1) = kf_covariances_(2, matched_index);
            P(1, 1) = kf_covariances_(3, matched_index);
            Eigen::Matrix2d S = P + R_;
            Eigen::Matrix2d K = P * S.inverse();

            // Update kf_points
            Eigen::Vector2d y = dhs_measured.col(i).transpose() - kf_points_.col(matched_index).transpose();
            Eigen::Vector2d correction = K * y;
            kf_points_.col(matched_index) += correction.transpose();

            // Update covariance
            Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
            Eigen::Matrix2d P_new = (I - K) * P;
            kf_covariances_.col(matched_index) << P_new(0, 0), P_new(1, 0), P_new(0, 1), P_new(1, 1);

            // Increment valid count
            valid_counts_[matched_index]++;
        }
        else
        {
            // No matches, add measured point as a new Kalman filter
            dhs_to_add.conservativeResize(2, dhs_to_add.cols() + 1);
            dhs_to_add.col(dhs_to_add.cols() - 1) = dhs_measured.col(i);
        }
    }

    if (dhs_to_add.cols() > 0)
    {
        addKFs(dhs_to_add);
    }
    if (kf_ids_.maxCoeff() > max_kf_id_)
    {
        throw std::runtime_error("4 max kf_id max exceeds max_kf_id_: " + std::to_string(valid_counts_.maxCoeff()));
    }
    if (kf_points_.maxCoeff() > 100.0 || kf_points_.minCoeff() < -100.0)
    {
        throw std::runtime_error("4 kf_points out of bounds: " + std::to_string(kf_points_.maxCoeff()) + ", " + std::to_string(kf_points_.minCoeff()));
    }

    int num_removed = removeStaleKFs();
    if (num_removed > 0)
    {
        RCLCPP_INFO(rclcpp::get_logger("PositionKalmanFilters"), "Removed %d stale Kalman filters", num_removed);
    }

    // Final validation checks
    if (valid_counts_.size() != (size_t)kf_points_.cols()
        || kf_ids_.size() != (size_t)kf_points_.cols()
        || kf_covariances_.cols() != (size_t)kf_points_.cols()
        || kf_colors_.cols() != (size_t)kf_points_.cols())
    {
        throw std::runtime_error("Size mismatch after removal");
    }
    if (kf_ids_.maxCoeff() > max_kf_id_)
    {
        throw std::runtime_error("5 max kf_id max exceeds max_kf_id_: " + std::to_string(valid_counts_.maxCoeff()));
    }
    if (kf_points_.maxCoeff() > 100.0 || kf_points_.minCoeff() < -100.0)
    {
        throw std::runtime_error("5 kf_points out of bounds: " + std::to_string(kf_points_.maxCoeff()) + ", " + std::to_string(kf_points_.minCoeff()));
    }
}

int PositionKalmanFilters::removeStaleKFs()
{
    if (valid_counts_.size() != kf_points_.cols()
        || kf_ids_.size() != kf_points_.cols()
        || kf_covariances_.cols() != kf_points_.cols()
        || kf_colors_.cols() != kf_points_.cols())
    {
        throw std::runtime_error("Size mismatch before removal");
    }
    // 1) Build a 1×N Boolean mask: keepMask[j] = (valid_counts_[j] >= 0)
    Eigen::Array<bool, 1, Eigen::Dynamic> keepMask =
        (valid_counts_.array() >= 0);

    int oldN = valid_counts_.size();
    int newN = static_cast<int>(keepMask.count());
    int removed = oldN - newN;

    // 2) If target_kf_id_ was among the invalid ones, clear it:
    if (target_kf_id_ >= 0)
    {
        // Build a mask for where kf_ids_ == target_kf_id_
        Eigen::Array<bool, 1, Eigen::Dynamic> targetMask = (kf_ids_.array() == target_kf_id_);
        // If any of those columns are now invalid, reset:
        if ((targetMask && (keepMask == false)).any())
        {
            target_kf_id_ = -1;
        }
    }

    // 3) Early out if nothing to remove
    if (removed == 0)
    {
        return 0;
    }

    // Allocate new, smaller matrices
    Eigen::VectorXi new_ids(newN);
    Eigen::Matrix2Xd new_points(2, newN);
    Eigen::MatrixXd new_Ps_flat(4, newN);
    Eigen::Matrix3Xd new_colors(3, newN);
    Eigen::VectorXi new_valid(newN);

    for (int srcCol = 0, dstCol = 0; srcCol < oldN; ++srcCol)
    {
        if (keepMask(srcCol))
        {
            new_ids(dstCol) = kf_ids_(srcCol);
            new_points.col(dstCol) = kf_points_.col(srcCol);
            new_Ps_flat.col(dstCol) = kf_covariances_.col(srcCol);
            new_colors.col(dstCol) = kf_colors_.col(srcCol);
            new_valid(dstCol) = valid_counts_(srcCol);
            ++dstCol;
        }
    }

    // 6) Swap in the compacted matrices
    kf_ids_ = std::move(new_ids);
    kf_points_ = std::move(new_points);
    kf_covariances_ = std::move(new_Ps_flat);
    kf_colors_ = std::move(new_colors);
    valid_counts_ = std::move(new_valid);

    return removed;
}

int PositionKalmanFilters::getTargetID()
{
    // If target_kf_id_ is still valid, return it
    if (target_kf_id_ != -1)
    {
        // Check if it's in the current list
        if ((kf_ids_.array() == target_kf_id_).any())
        {
            return target_kf_id_;
        }
    }

    // No Kalman filter points
    if (kf_ids_.size() == 0)
    {
        return -1;
    }

    // Create mask for valid counts
    Eigen::Array<bool, 1, Eigen::Dynamic> valid_mask = (valid_counts_.array() > min_valid_kf_count_threshold_);

    if (!valid_mask.any())
    {
        return -1;
    }

    // Filter heights and find index of min height among valid
    Eigen::ArrayXd heights = kf_points_.row(1).array(); // heights
    double min_height = std::numeric_limits<double>::max();
    int closest_index = -1;

    for (int i = 0; i < heights.size(); ++i)
    {
        if (valid_mask(i) && heights(i) < min_height)
        {
            min_height = heights(i);
            closest_index = i;
        }
    }

    if (closest_index != -1)
    {
        target_kf_id_ = kf_ids_(closest_index);
        return target_kf_id_;
    }

    return -1;
}

std::vector<int> PositionKalmanFilters::getValidKFIndices() const
{
    std::vector<int> valid_indices;
    for (int i = 0; i < valid_counts_.size(); ++i)
    {
        if (valid_counts_(i) >= min_valid_kf_count_threshold_)
        {
            valid_indices.push_back(i);
        }
    }
    return valid_indices;
}
Eigen::VectorXd PositionKalmanFilters::getKFColor(int kf_index) const
{
    if (kf_index < 0 || kf_index >= kf_colors_.cols())
    {
        throw std::out_of_range("getKFColor: index out of range");
    }
    // Return the color for the specified KF index
    return kf_colors_.col(kf_index);
}
std::pair<Eigen::Vector2d, int> PositionKalmanFilters::getKFByID(int kf_id) const
{
    // Find all matching indices
    std::vector<int> matches;
    matches.reserve(kf_ids_.cols());
    Eigen::Vector2d state;
    int idx = -1;
    for (int j = 0; j < kf_ids_.cols(); ++j)
    {
        if (kf_ids_(j) == kf_id)
        {
            matches.push_back(j);
        }
    }

    if (matches.empty())
    {
        // throw std::out_of_range("getKFByID: ID " + std::to_string(kf_id) + " not found");
        idx = -1;
    }
    else if (matches.size() > 1)
    {
        throw std::out_of_range("getKFByID: multiple entries for ID " + std::to_string(kf_id));
    }
    else {
        idx = matches[0];
        // Extract the (distance, height) column
        state = kf_points_.col(idx);
    }
    return {state, idx};
}
std::vector<bool> PositionKalmanFilters::checkKFsInFrame(
    const std::vector<int> &kf_indices,
    double camera_yaw) const
{
    // 1) Get 3×M XYZs for the requested indices
    Eigen::Matrix3Xd xyz = getKFXYZs(camera_yaw, kf_indices);
    int M = xyz.cols();
    if (M == 0)
    {
        return {};
    }

    // 2) Project into homogeneous image coordinates: 3×M
    Eigen::Matrix3Xd img_h = camera_intrinsics_ * xyz;

    // 3) Vectorized normalize:
    //    Divide row 0 and row 1 by row 2, element-wise
    Eigen::ArrayXd inv_z = img_h.row(2).array().inverse(); // 1×M
    Eigen::ArrayXd px = img_h.row(0).array() * inv_z;      // 1×M
    Eigen::ArrayXd py = img_h.row(1).array() * inv_z;      // 1×M

    // 4) Bounds check in one shot:
    //    0 <= px < width, 0 <= py < height
    Eigen::Array<bool, 1, Eigen::Dynamic> in_frame_mask =
        (px >= 0.0) && (px < image_size_.first) && (py >= 0.0) && (py < image_size_.second);

    // 5) Convert mask to std::vector<bool>
    std::vector<bool> in_frame(M);
    for (int i = 0; i < M; ++i)
    {
        in_frame[i] = in_frame_mask(i);
    }
    return in_frame;
}

Eigen::Matrix2Xd PositionKalmanFilters::getDHFromXYZs(
    const Eigen::Matrix3Xd &xyz_points,
    double wire_yaw)
{
    const int N = xyz_points.cols();
    Eigen::Matrix2Xd dhs(2, N);

    // Precompute sin/cos
    double s = std::sin(wire_yaw);
    double c = std::cos(wire_yaw);

    // Compute distances: -sin(yaw)*x + cos(yaw)*y
    // xyz_points.row(0) is x‑row, row(1) is y‑row, row(2) is z‑row
    dhs.row(0) = (-s * xyz_points.row(0).array() +
                  c * xyz_points.row(1).array())
                     .matrix();

    // Heights = z coordinate
    dhs.row(1) = xyz_points.row(2);

    return dhs;
}

Eigen::Matrix3Xd PositionKalmanFilters::getKFXYZs(
    double wire_yaw,
    const std::vector<int> &inds) const
{
    const int N = static_cast<int>(kf_points_.cols());
    if (N == 0)
    {
        throw std::runtime_error("getKFXYZs: no points");
    }

    // Precompute
    double s = std::sin(wire_yaw);
    double c = std::cos(wire_yaw);

    if (inds.empty())
    {
        // vectorized for all points:
        Eigen::Matrix3Xd xyz(3, N);
        // row 0: x = -sin(yaw) * d
        xyz.row(0) = (-s * kf_points_.row(0).array()).matrix();
        // row 1: y =  cos(yaw) * d
        xyz.row(1) = (c * kf_points_.row(0).array()).matrix();
        // row 2: z = h
        xyz.row(2) = kf_points_.row(1);
        return xyz;
    }
    else
    {
        // fall back to the indexed version if you really need inds
        Eigen::Matrix3Xd xyz(3, inds.size());
        for (int j = 0; j < (int)inds.size(); ++j)
        {
            int i = inds[j];
            double d = kf_points_(0, i);
            xyz(0, j) = -s * d;
            xyz(1, j) = c * d;
            xyz(2, j) = kf_points_(1, i);
        }
        return xyz;
    }
}

Eigen::Matrix3Xd PositionKalmanFilters::generateVizColor(int num_colors)
{
    // 1) Create an Nx1 HSV Mat
    cv::Mat hsv(num_colors, 1, CV_8UC3);

    // 2) Fill H channel ∈ [0,180), S channel ∈ [128,256), V channel = 255
    cv::Scalar lowerb(0, 128, 255);
    cv::Scalar upperb(180, 256, 256);
    cv::randu(hsv, lowerb, upperb);

    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // Allocate output Eigen matrix (3 rows, num_colors cols), double for example
    Eigen::Matrix3Xd colors(3, num_colors);

    // rgb.ptr<cv::Vec3b>(i) points to the i-th pixel (Vec3b = uchar[3])
    // Loop through each pixel and fill columns of Eigen matrix
    for (int i = 0; i < num_colors; ++i)
    {
        cv::Vec3b pixel = rgb.at<cv::Vec3b>(i, 0);
        colors(0, i) = static_cast<double>(pixel[0]); // R
        colors(1, i) = static_cast<double>(pixel[1]); // G
        colors(2, i) = static_cast<double>(pixel[2]); // B
    }

    return colors;
}