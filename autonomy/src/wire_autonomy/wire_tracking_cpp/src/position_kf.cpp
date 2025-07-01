#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <vector>

#include "kalman_filters.h"
#include "position_kf.h"

PositionKalmanFilters::PositionKalmanFilters(const YAML::Node &config,
                                             const Eigen::Matrix3d &camera_intrinsics,
                                             const std::pair<int, int> &image_size)
    : camera_intrinsics_(camera_intrinsics), image_size_(image_size)
{

    z_predict_cov_ = config["z_predict_covariance"].as<double>();
    z_measurement_cov_ = config["z_measurement_covariance"].as<double>();
    z_max_covariance_ = config["z_max_covariance"].as<double>();

    y_predict_cov_ = config["y_predict_covariance"].as<double>();
    y_measurement_cov_ = config["y_measurement_covariance"].as<double>();
    y_max_covariance_ = config["y_max_covariance"].as<double>();

    initial_cov_multiplier_ = config["initial_yaw_covariance_multiplier"].as<double>();
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
    yaw_max_covariance_ = yaw_max_covariance_ * yaw_max_covariance_;

    P_init_ = R_ * initial_cov_multiplier_;
}

void PositionKalmanFilters::addKFs(const Eigen::Matrix2Xd &dhs0)
{
    const int oldN = kf_points_.cols();
    const int newN = dhs0.cols();
    if (newN == 0)
        return;

    // 1) Resize all the matrices in one go:
    kf_ids_.conservativeResize(Eigen::NoChange, oldN + newN);
    kf_points_.conservativeResize(Eigen::NoChange, oldN + newN);
    kf_Ps_flat_.conservativeResize(Eigen::NoChange, oldN + newN);
    kf_colors_.conservativeResize(Eigen::NoChange, oldN + newN);
    valid_counts_.conservativeResize(Eigen::NoChange, oldN + newN);

    // 2) Fill the new-id block with a linspace:
    kf_ids_.segment(oldN, newN) = Eigen::ArrayXi::LinSpaced(newN, max_kf_id_, max_kf_id_ + newN - 1).transpose();

    // 3) Copy in the new points:
    kf_points_.block(0, oldN, 2, newN) = dhs0;

    // 4) Tile P_init_ into each new column:
    //    First flatten P_init_ column‑wise to size (4×1):
    Eigen::Vector4d Pcol;
    Eigen::Map<Eigen::Matrix2d>(Pcol.data(), 2, 2) = P_init_;
    //    Then replicate across newN columns:
    kf_Ps_flat_.block(0, oldN, 4, newN) = Pcol.replicate(1, newN);

    // 5) Generate and copy colors in one call:
    auto cols = generateVizColor(newN); // returns 3×newN Eigen::Matrix3Xd
    kf_colors_.block(0, oldN, 3, newN) = cols;

    // 6) Fill valid_counts_ with a constant:
    valid_counts_.segment(oldN, newN).setConstant(valid_count_buffer_);

    max_kf_id_ += newN;
}

int PositionKalmanFilters::removeStaleKFs()
{
    // 1) Build a 1×N Boolean mask: keepMask[j] = (valid_counts_[j] >= 0)
    Eigen::Array<bool, 1, Eigen::Dynamic> keepMask =
        (valid_counts_.array() >= 0);

    int oldN = valid_counts_.cols();
    int newN = static_cast<int>(keepMask.count());
    int removed = oldN - newN;

    // 2) If target_kf_id_ was among the invalid ones, clear it:
    if (target_kf_id_ >= 0)
    {
        // Build a mask for where kf_ids_ == target_kf_id_
        Eigen::Array<bool, 1, Eigen::Dynamic> targetMask =
            (kf_ids_.array() == target_kf_id_);
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

    // 4) Allocate new, smaller matrices
    Eigen::Matrix<int, 1, Eigen::Dynamic> new_ids(1, newN);
    Eigen::Matrix2Xd new_points(2, newN);
    Eigen::Matrix<double, 4, Eigen::Dynamic> new_Ps_flat(4, newN);
    Eigen::Matrix3Xd new_colors(3, newN);
    Eigen::Matrix<int, 1, Eigen::Dynamic> new_valid(1, newN);

    // 5) Copy columns *in one shot* by iterating only over the kept indices
    //    (the only small loop we need)
    for (int srcCol = 0, dstCol = 0; srcCol < oldN; ++srcCol)
    {
        if (keepMask(srcCol))
        {
            new_ids(0, dstCol) = kf_ids_(0, srcCol);
            new_points.col(dstCol) = kf_points_.col(srcCol);
            new_Ps_flat.col(dstCol) = kf_Ps_flat_.col(srcCol);
            new_colors.col(dstCol) = kf_colors_.col(srcCol);
            new_valid(0, dstCol) = valid_counts_(0, srcCol);
            ++dstCol;
        }
    }

    // 6) Swap in the compacted matrices
    kf_ids_ = std::move(new_ids);
    kf_points_ = std::move(new_points);
    kf_Ps_flat_ = std::move(new_Ps_flat);
    kf_colors_ = std::move(new_colors);
    valid_counts_ = std::move(new_valid);

    return removed;
}

void PositionKalmanFilters::initializeKFs(const Eigen::Matrix3Xd &camera_points, double wire_yaw)
{
    // 1) Validate input shape: must be 3×N
    if (camera_points.rows() != 3)
    {
        throw std::invalid_argument(
            "initializeKFs: camera_points must have 3 rows (got " + std::to_string(camera_points.rows()) + ")");
    }

    // 2) Compute (distance, height) pairs from 3D points
    //    Assumes you already have a method with this signature:
    //      Eigen::Matrix2Xd getDHFromXYZs(const Eigen::Matrix3Xd&, double);
    Eigen::Matrix2Xd dhs = getDHFromXYZs(camera_points, wire_yaw);

    // 3) Batch‑add these new filters
    add_kfs(dhs);

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
    int N = xyz.cols();

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
    //    We store covariances as a 4×N flat matrix kf_Ps_flat_
    Eigen::Matrix<double, 4, 4> KJJ =
        Eigen::kroneckerProduct(J, J);

    // Flatten Q_ (2×2) column‑wise into a 4×1 vector
    Eigen::Vector4d q_flat;
    q_flat << Q_(0, 0), Q_(1, 0),
        Q_(0, 1), Q_(1, 1);

    // Now kf_Ps_flat_ is a (4 × N) matrix, each column is vec(P_old).
    // We can do all at once:
    kf_Ps_flat_ = (KJJ * kf_Ps_flat_).colwise() + q_flat;
}
void PositionKalmanFilters::update(
    const Eigen::Matrix3Xd &cam_points,
    double wire_yaw)
{
    // 1) Validate input
    if (cam_points.rows() != 3)
    {
        throw std::invalid_argument(
            "update: cam_points must be 3×M");
    }
    int M = int(dhs_meas.cols());
    int N = int(kf_points_.cols());

    // 1) compute squared norms of columns
    Eigen::RowVectorXd meas_sqnorm = dhs_meas.colwise().squaredNorm();   // 1×M
    Eigen::RowVectorXd filt_sqnorm = kf_points_.colwise().squaredNorm(); // 1×N

    // 2) compute cross-term: (dhs_meas^T * kf_points_) is M×N
    Eigen::MatrixXd cross = dhs_meas.transpose() * kf_points_; // M×N

    // 3) build M×N matrix of squared distances
    Eigen::MatrixXd d2 =
        meas_sqnorm.transpose().replicate(1, N) // each row i = meas_sqnorm(i)
        + filt_sqnorm.replicate(M, 1)           // each col j = filt_sqnorm(j)
        - 2.0 * cross;

    // 4) sqrt → actual distances
    Eigen::MatrixXd comp_dists = d2.array().max(0.0).sqrt();

    // 5) boolean match matrix
    Eigen::ArrayXX<bool> matched = (comp_dists.array() < wire_matching_min_threshold_m_);

    // 6) vectorized “unmatched filter” mask: true for filters j with no matches
    //    (~matched) flips each element, then .colwise().all() checks rows over M
    Eigen::Array<bool, 1, Eigen::Dynamic> unmatched_filter_mask =
        (~matched).colwise().all(); // length N

    // 7) decrement valid_counts_ for those filters in‑frame
    //    assume checkKFsInFrame returns a Array<bool,1,N> in_frame_mask
    Eigen::Array<bool, 1, Eigen::Dynamic> in_frame_mask = checkKFsInFrameMask(wire_yaw);
    valid_counts_.array() -= (unmatched_filter_mask && in_frame_mask).cast<int>();
}

int PositionKalmanFilters::getTargetID() {
    // If target_kf_id_ is still valid, return it
    if (target_kf_id_ != -1) {
        // Check if it's in the current list
        if ((kf_ids_.array() == target_kf_id_).any()) {
            return target_kf_id_;
        }
    }

    // No Kalman filter points
    if (kf_ids_.size() == 0) {
        return -1;
    }

    // Create mask for valid counts
    Eigen::Array<bool, 1, Eigen::Dynamic> valid_mask = (valid_counts_.array() > valid_count_buffer_);

    if (!valid_mask.any()) {
        return -1;
    }

    // Filter heights and find index of min height among valid
    Eigen::ArrayXd heights = kf_points_.row(1).array();  // heights
    double min_height = std::numeric_limits<double>::max();
    int closest_index = -1;

    for (int i = 0; i < heights.size(); ++i) {
        if (valid_mask(i) && heights(i) < min_height) {
            min_height = heights(i);
            closest_index = i;
        }
    }

    if (closest_index != -1) {
        target_kf_id_ = kf_ids_(0, closest_index);
        return target_kf_id_;
    }

    return -1;
}


std::pair<Eigen::Vector2d, int> PositionKalmanFilters::getKFByID(int kf_id) const
{
    // Find all matching indices
    std::vector<int> matches;
    matches.reserve(kf_ids_.cols());
    for (int j = 0; j < kf_ids_.cols(); ++j) {
        if (kf_ids_(0, j) == kf_id) {
            matches.push_back(j);
        }
    }

    if (matches.empty()) {
        throw std::out_of_range("getKFByID: ID " + std::to_string(kf_id) + " not found");
    }
    if (matches.size() > 1) {
        throw std::out_of_range("getKFByID: multiple entries for ID " + std::to_string(kf_id));
    }

    int idx = matches[0];
    // Extract the (distance, height) column
    Eigen::Vector2d state = kf_points_.col(idx);
    return {state, idx};
}
std::vector<bool> PositionKalmanFilters::checkKFsInFrame(
    const std::vector<int> &kf_indices,
    double camera_yaw) const
{
    // 1) Get 3×M XYZs for the requested indices
    Eigen::Matrix3Xd xyz = getKFXYZs(camera_yaw, kf_indices);
    int M = xyz.cols();
    if (M == 0) {
        return {};
    }

    // 2) Project into homogeneous image coordinates: 3×M
    Eigen::Matrix3Xd img_h = camera_intrinsics_ * xyz;

    // 3) Vectorized normalize: 
    //    Divide row 0 and row 1 by row 2, element-wise
    Eigen::ArrayXd inv_z = img_h.row(2).array().inverse();   // 1×M
    Eigen::ArrayXd px   = img_h.row(0).array() * inv_z;      // 1×M
    Eigen::ArrayXd py   = img_h.row(1).array() * inv_z;      // 1×M

    // 4) Bounds check in one shot:
    //    0 <= px < width, 0 <= py < height
    Eigen::Array<bool,1,Eigen::Dynamic> in_frame_mask = 
        (px >= 0.0) 
      && (px <  image_size_.first)
      && (py >= 0.0) 
      && (py <  image_size_.second);

    // 5) Convert mask to std::vector<bool>
    std::vector<bool> in_frame(M);
    for (int i = 0; i < M; ++i) {
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

std::vector<cv::Vec3b> PositionKalmanFilters::generateVizColor(int num_colors)
{
    // 1) Create an Nx1 HSV Mat
    cv::Mat hsv(num_colors, 1, CV_8UC3);

    // 2) Fill H channel ∈ [0,180), S channel ∈ [128,256), V channel = 255
    //    We can use randu with per‑channel bounds:
    cv::Scalar lowerb(  0, 128, 255);
    cv::Scalar upperb(180, 256, 256);
    cv::randu(hsv, lowerb, upperb);

    // 3) Convert HSV → BGR (still Nx1)
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    // 4) Convert BGR → RGB in one shot
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // 5) Now rgb is Nx1 CV_8UC3 and continuous.  
    //    We can bulk‑copy into std::vector without a manual loop:
    std::vector<cv::Vec3b> colors;
    colors.assign(
        (cv::Vec3b*)rgb.ptr<cv::Vec3b>(0),
        (cv::Vec3b*)rgb.ptr<cv::Vec3b>(0) + num_colors
    );

    return colors;
}

bool PositionKalmanFilters::isInitialized() const
{
    return initialized_;
}