#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <optional>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace py = pybind11;
using Eigen::MatrixXf;
using Eigen::Vector3f;
using std::vector;

// build with:
// g++ -O3 -Wall -shared -std=c++17 -fPIC     $(python3 -m pybind11 --includes)     -I/usr/include/eigen3     $(pkg-config --cflags opencv4)     ransac_bindings.cpp     -o ransac_bindings$(python3-config --extension-suffix)     $(pkg-config --libs opencv4)

// Fold angle to [0, pi]
float fold_angle_from_0_to_pi(float angle)
{
    angle = std::fmod(angle, 2 * M_PI);
    if (angle < 0)
        angle += 2 * M_PI;
    return (angle > M_PI) ? (angle - M_PI) : angle;
}

float perpendicular_angle_rad(float angle_rad)
{
    float perp_angle = fold_angle_from_0_to_pi(angle_rad + static_cast<float>(M_PI) / 2.0f);
    if (perp_angle > static_cast<float>(M_PI) / 2.0f)
    {
        perp_angle -= static_cast<float>(M_PI);
    }
    return perp_angle;
}

// Calculate closest distances from points to line segment defined by two points p1, p2
Eigen::VectorXf find_closest_distance_from_points_to_line_3d(
    const Eigen::MatrixXf &points,   // Nx3 matrix
    const Eigen::MatrixXf &line_ends // 2x3 matrix
)
{
    if (points.cols() != 3)
    {
        throw std::runtime_error("Points must be 3D (Nx3 matrix)");
    }
    if (line_ends.rows() != 2 || line_ends.cols() != 3)
    {
        throw std::runtime_error("Line ends must be a 2x3 matrix");
    }

    const Eigen::RowVector3f p1 = line_ends.row(0).cast<float>(); // Fixed: Cast to float
    const Eigen::RowVector3f p2 = line_ends.row(1).cast<float>(); // Fixed: Cast to float
    const Eigen::RowVector3f line_vector = p2 - p1;
    const float line_length_squared = line_vector.squaredNorm();

    // Vector from p1 to each point
    Eigen::MatrixXf p1_to_points = points.rowwise() - p1;

    // Project each point onto the line direction
    Eigen::VectorXf t = (p1_to_points * line_vector.transpose()) / line_length_squared;

    // Clamp t to [0, 1]
    t = t.array().min(1.0f).max(0.0f); // Fixed: Use float literals

    // Compute closest points on the segment
    Eigen::MatrixXf closest_points = t.replicate(1, 3).array() * line_vector.replicate(t.rows(), 1).array();
    closest_points.rowwise() += p1;

    // Compute distances
    Eigen::VectorXf distances = (points - closest_points).rowwise().norm();
    return distances;
}

std::tuple<vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>, vector<int>> ransac_line_fitting(
    Eigen::MatrixXf points,
    float avg_angle,
    int num_lines,
    int num_iterations,
    float inlier_threshold,
    float vert_angle_maximum_rad,
    float horz_angle_diff_maximum_rad)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, points.rows() - 1);

    avg_angle = fold_angle_from_0_to_pi(avg_angle);

    vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> best_lines;
    vector<int> line_inlier_counts;

    for (int line_i = 0; line_i < num_lines; ++line_i)
    {
        Eigen::VectorXi best_inliers_mask(points.rows());
        best_inliers_mask.setZero();
        int best_inlier_count = 0;
        std::pair<Eigen::Vector3f, Eigen::Vector3f> best_line;

        for (int iter = 0; iter < num_iterations; ++iter)
        {
            if (points.rows() < 2)
                break;

            // Sample two unique indices
            int idx1 = dist(rng);
            int idx2;
            do
            {
                idx2 = dist(rng);
            } while (idx2 == idx1);

            // Work with float vectors consistently
            Eigen::Vector3f p1 = points.row(idx1);
            Eigen::Vector3f p2 = points.row(idx2);

            // Compute pitch angle
            float pitch_angle = std::atan2(std::abs(p2.z() - p1.z()), (p2.head<2>() - p1.head<2>()).norm());
            pitch_angle = fold_angle_from_0_to_pi(pitch_angle);
            if (pitch_angle > vert_angle_maximum_rad && pitch_angle < M_PI - vert_angle_maximum_rad)
            {
                continue;
            }

            // Compute yaw angle
            float yaw_angle = std::atan2(p2.y() - p1.y(), p2.x() - p1.x());
            yaw_angle = fold_angle_from_0_to_pi(yaw_angle);

            float angle_diff = std::abs(yaw_angle - avg_angle);
            if (angle_diff > M_PI / 2)
            {
                angle_diff = std::abs(M_PI - angle_diff);
            }
            if (angle_diff > horz_angle_diff_maximum_rad)
            {
                continue;
            }

            // Calculate distances
            Eigen::MatrixXf line_points(2, 3);
            line_points.row(0) = p1;
            line_points.row(1) = p2;
            Eigen::VectorXf distances = find_closest_distance_from_points_to_line_3d(points, line_points);

            // Find inliers
            Eigen::ArrayXf dist_array = distances.array();

            // Create a boolean mask array where distance <= threshold
            Eigen::Array<bool, Eigen::Dynamic, 1> bool_mask = (dist_array <= inlier_threshold);

            // Convert boolean mask to integer mask (1 or 0)
            Eigen::VectorXi inlier_mask = bool_mask.cast<int>();

            // Count the number of inliers (sum of ones)
            int inlier_count = inlier_mask.sum();

            if (inlier_count > best_inlier_count)
            {
                best_inlier_count = inlier_count;
                best_line = {p1, p2};
                best_inliers_mask = inlier_mask;
            }
        }

        if (best_inlier_count == 0)
        {
            break;
        }

        best_lines.push_back(best_line);
        line_inlier_counts.push_back(best_inlier_count);

        // Remove inliers
        if (num_lines > 1)
        {
            vector<int> outlier_indices;
            for (int i = 0; i < points.rows(); ++i)
            {
                if (best_inliers_mask(i) == 0)
                {
                    outlier_indices.push_back(i);
                }
            }

            if ((int)outlier_indices.size() <= 2)
            {
                break;
            }

            Eigen::MatrixXf filtered_points(outlier_indices.size(), 3);
            for (size_t i = 0; i < outlier_indices.size(); ++i)
            {
                filtered_points.row(i) = points.row(outlier_indices[i]);
            }
            points = filtered_points;
            dist = std::uniform_int_distribution<>(0, points.rows() - 1);
        }
    }

    // Combine lines if their avg z height is within threshold
    if (best_lines.size() > 1)
    {
        vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> combined_lines;
        vector<int> combined_inlier_counts;

        for (size_t i = 0; i < best_lines.size(); ++i)
        {
            const auto &line = best_lines[i];
            float avg_height = (line.first.z() + line.second.z()) / 2.0f;

            bool merged = false;
            for (size_t j = 0; j < combined_lines.size(); ++j)
            {
                const auto &c_line = combined_lines[j];
                float combined_avg_height = (c_line.first.z() + c_line.second.z()) / 2.0f;
                if (std::abs(combined_avg_height - avg_height) <= inlier_threshold * 2)
                {
                    // Check distances to decide merging
                    float d1 = (line.first - c_line.first).norm() + (line.second - c_line.second).norm();
                    float d2 = (line.first - c_line.second).norm() + (line.second - c_line.first).norm();

                    int combined_count = combined_inlier_counts[j];
                    int current_count = line_inlier_counts[i];
                    int total_count = combined_count + current_count;

                    Eigen::Vector3f new_p1, new_p2;
                    if (d1 < d2)
                    {
                        new_p1 = (c_line.first * combined_count + line.first * current_count) / total_count;
                        new_p2 = (c_line.second * combined_count + line.second * current_count) / total_count;
                    }
                    else
                    {
                        new_p1 = (c_line.first * combined_count + line.second * current_count) / total_count;
                        new_p2 = (c_line.second * combined_count + line.first * current_count) / total_count;
                    }

                    combined_lines[j] = {new_p1, new_p2};
                    combined_inlier_counts[j] += current_count;
                    merged = true;
                    break;
                }
            }
            if (!merged)
            {
                combined_lines.push_back(line);
                combined_inlier_counts.push_back(line_inlier_counts[i]);
            }
        }
        return {combined_lines, combined_inlier_counts};
    }

    return {best_lines, line_inlier_counts};
}

struct ROIData
{
    std::vector<cv::Mat> roi_depths;
    std::vector<cv::Mat> roi_rgbs;
    cv::Mat depth_img_masked;
    cv::Mat masked_viz_img;
};

ROIData roi_to_point_clouds(
    const std::vector<std::pair<float, float>> &rois,
    const float avg_angle,
    const cv::Mat &depth_image,
    const cv::Mat *viz_img = nullptr)
{
    const cv::Size img_shape = depth_image.size();
    const cv::Point2f img_center(img_shape.width * 0.5f, img_shape.height * 0.5f);
    const float perp_angle = perpendicular_angle_rad(avg_angle);

    std::vector<cv::Mat> roi_depths;
    std::vector<cv::Mat> roi_rgbs;

    cv::Mat viz_mask;
    if (viz_img)
        viz_mask = cv::Mat::zeros(viz_img->size(), CV_8UC1);

    cv::Mat single_roi_mask(depth_image.size(), CV_8UC1);
    const float perp_angle_deg = static_cast<float>(perp_angle * 180.0 / M_PI);

    for (const auto &[start, end] : rois)
    {
        const float center_dist = 0.5f * (start + end);
        const float length = std::abs(end - start);

        const float dx = center_dist * std::cos(perp_angle);
        const float dy = center_dist * std::sin(perp_angle);
        const cv::Point2f center_coords(img_center.x + dx, img_center.y + dy);

        cv::Size2f size(length, img_shape.width * 2.0f);
        cv::RotatedRect rect(center_coords, size, perp_angle_deg);

        cv::Point2f box_pts[4];
        rect.points(box_pts);

        std::vector<cv::Point> box_int_pts(4);
        for (int i = 0; i < 4; ++i)
            box_int_pts[i] = cv::Point(cvRound(box_pts[i].x), cvRound(box_pts[i].y));

        single_roi_mask.setTo(0);
        cv::fillConvexPoly(single_roi_mask, box_int_pts, 255);

        cv::Mat roi_depth;
        cv::bitwise_and(depth_image, depth_image, roi_depth, single_roi_mask);
        roi_depths.push_back(std::move(roi_depth));

        if (viz_img)
        {
            cv::Mat roi_rgb;
            cv::bitwise_and(*viz_img, *viz_img, roi_rgb, single_roi_mask);
            roi_rgbs.push_back(std::move(roi_rgb));
            cv::fillConvexPoly(viz_mask, box_int_pts, 255);
        }
    }

    ROIData result;
    result.roi_depths = std::move(roi_depths);
    result.roi_rgbs = std::move(roi_rgbs);

    if (viz_img)
    {
        viz_img->copyTo(result.masked_viz_img, viz_mask);
        depth_image.copyTo(result.depth_img_masked, viz_mask);
    }

    return result;
}

// Return type: point cloud (Nx3) and optional RGBs (Nx3)
std::pair<Eigen::MatrixXf, std::optional<Eigen::MatrixXf>> depth_to_pointcloud(
    const cv::Mat &depth_image,         // CV_32FC1, size HxW
    const Eigen::MatrixXf &camera_rays, // (H*W x 3)
    const std::pair<float, float> &depth_clip = {0.5f, 10.0f},
    const cv::Mat *rgb_image = nullptr) // Optional CV_8UC3, size HxW
{
    const int H = depth_image.rows;
    const int W = depth_image.cols;
    const int N = H * W;

    if (depth_image.type() != CV_32FC1)
    {
        throw std::runtime_error("depth_image must be of type CV_32FC1 (32-bit float, single channel)");
    }

    if (camera_rays.rows() != N || camera_rays.cols() != 3)
    {
        throw std::runtime_error("camera_rays must have shape (N, 3)");
    }

    if (rgb_image)
    {
        if (rgb_image->type() != CV_8UC3)
        {
            throw std::runtime_error("rgb_image must be of type CV_8UC3 (8-bit 3-channel color image)");
        }
        if (rgb_image->rows != H || rgb_image->cols != W)
        {
            throw std::runtime_error("rgb_image dimensions must match expected size (H, W)");
        }
    }

    Eigen::Map<const Eigen::VectorXf> z_coords(depth_image.ptr<float>(), N);

    std::vector<int> valid_indices;
    valid_indices.reserve(N);
    for (int i = 0; i < N; ++i)
    {
        float z = z_coords(i);
        if (!std::isnan(z) && z > depth_clip.first && z < depth_clip.second)
            valid_indices.push_back(i);
    }

    const int M = static_cast<int>(valid_indices.size());
    Eigen::MatrixXf points(M, 3);
    std::optional<Eigen::MatrixXf> rgb_points = std::nullopt;

    if (rgb_image)
    {
        Eigen::MatrixXf colors(M, 3);                            // float RGB output
        const cv::Vec3b *rgb_data = rgb_image->ptr<cv::Vec3b>(); // BGR format

        for (int i = 0; i < M; ++i)
        {
            int idx = valid_indices[i];
            float z = z_coords(idx);

            points.row(i) = camera_rays.row(idx) * z;

            const cv::Vec3b &bgr = rgb_data[idx];
            colors.row(i) << bgr[2], bgr[1], bgr[0]; // RGB, not normalized
        }

        rgb_points = colors;
        return {points, rgb_points};
    }
    else
    {
        for (int i = 0; i < M; ++i)
        {
            int idx = valid_indices[i];
            float z = z_coords(idx);
            points.row(i) = camera_rays.row(idx) * z;
        }
        return {points, std::nullopt};
    }
}

// Struct definition
struct RansacResult
{
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> fitted_lines;
    std::vector<int> inlier_counts;
    std::vector<Eigen::MatrixXf> roi_point_clouds;
    std::optional<std::vector<Eigen::MatrixXf>> point_colors;
    std::optional<py::array_t<uint8_t>> masked_viz_img;
};

RansacResult ransac_on_rois(
    const std::vector<std::pair<float, float>> &rois,
    const std::vector<int> &roi_line_counts,
    float avg_angle,
    const Eigen::MatrixXf &camera_rays,
    const py::array_t<float> &depth_img,
    float min_depth_clip,
    float max_depth_clip,
    int ransac_max_iters,
    float inlier_threshold_m,
    float vert_angle_maximum_rad,
    float horz_angle_diff_maximum_rad,
    const std::optional<py::array_t<uint8_t>> &viz_img)
{
    cv::Mat depth_mat(depth_img.shape(0), depth_img.shape(1), CV_32FC1);
    std::memcpy(depth_mat.data, depth_img.data(), depth_img.size() * sizeof(float));

    if (depth_mat.type() != CV_32FC1)
    {
        throw std::runtime_error("depth_img must be of type CV_32FC1 (float grayscale image)");
    }

    if (camera_rays.rows() != depth_mat.rows * depth_mat.cols || camera_rays.cols() != 3)
    {
        throw std::runtime_error("camera_rays must have shape (H*W, 3) where H and W match depth_img");
    }

    const cv::Mat *viz_ptr = nullptr;
    cv::Mat viz_mat;

    if (viz_img.has_value())
    {
        if (viz_img->ndim() != 3 || viz_img->shape(0) != depth_mat.rows || viz_img->shape(1) != depth_mat.cols || viz_img->shape(2) != 3)
        {
            throw std::runtime_error("viz_img must have shape (H, W, 3) and match depth_img dimensions");
        }
        viz_mat = cv::Mat(viz_img->shape(0), viz_img->shape(1), CV_8UC3);
        std::memcpy(viz_mat.data, viz_img->data(), viz_img->size() * sizeof(uint8_t));

        if (viz_mat.rows != depth_mat.rows || viz_mat.cols != depth_mat.cols)
        {
            throw std::runtime_error("viz_img dimensions do not match depth_img");
        }

        if (viz_mat.type() != CV_8UC3)
        {
            throw std::runtime_error("viz_img must be of type CV_8UC3");
        }

        viz_ptr = &viz_mat;
    }
    ROIData roi_data = roi_to_point_clouds(rois, avg_angle, depth_mat, viz_ptr);

    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> fitted_lines;
    std::vector<int> line_inlier_counts;
    std::vector<Eigen::MatrixXf> roi_pcs;
    std::vector<Eigen::MatrixXf> roi_point_colors;

    if (roi_data.roi_rgbs.empty())
    {
        roi_data.roi_rgbs.resize(roi_data.roi_depths.size());
    }

    for (size_t i = 0; i < roi_data.roi_depths.size(); ++i)
    {
        const auto &roi_depth = roi_data.roi_depths[i];
        const auto &roi_rgb = roi_data.roi_rgbs[i];
        int line_count = roi_line_counts[i];

        auto [points, colors] = depth_to_pointcloud(
            roi_depth, camera_rays, {min_depth_clip, max_depth_clip}, viz_ptr ? &roi_rgb : nullptr);

        roi_pcs.push_back(points);
        if (colors.has_value())
        {
            roi_point_colors.push_back(colors.value());
        }
        else
        {
            roi_point_colors.emplace_back(); // Empty matrix
        }

        auto [lines, inlier_counts] = ransac_line_fitting(
            points, avg_angle, line_count, ransac_max_iters,
            inlier_threshold_m, vert_angle_maximum_rad, horz_angle_diff_maximum_rad);

        fitted_lines.insert(fitted_lines.end(), lines.begin(), lines.end());
        line_inlier_counts.insert(line_inlier_counts.end(), inlier_counts.begin(), inlier_counts.end());
    }

    RansacResult result;
    result.fitted_lines = std::move(fitted_lines);
    result.inlier_counts = std::move(line_inlier_counts);
    result.roi_point_clouds = std::move(roi_pcs);

    if (!roi_point_colors.empty())
    {
        result.point_colors = std::move(roi_point_colors);
    }
    else
    {
        result.point_colors = std::nullopt;
    }

    if (!roi_data.masked_viz_img.empty())
    {
        cv::Mat rgb_masked;
        cv::cvtColor(roi_data.masked_viz_img, rgb_masked, cv::COLOR_BGR2RGB);
        py::array_t<uint8_t> masked_viz_img(
            {rgb_masked.rows, rgb_masked.cols, 3},
            {rgb_masked.step[0], rgb_masked.step[1], sizeof(uint8_t)},
            rgb_masked.data);
        result.masked_viz_img = masked_viz_img;
    }
    else
    {
        result.masked_viz_img = std::nullopt;
    }

    return result;
}

std::tuple<py::array_t<float>, py::array_t<uint8_t>> test_img_handoff(const py::array_t<float> &depth_img, const py::array_t<uint8_t> &viz_img)
{
    auto rows = viz_img.shape(0);
    auto cols = viz_img.shape(1);
    auto type = CV_8UC3;

    cv::Mat depth_mat(depth_img.shape(0), depth_img.shape(1), CV_32FC1);
    std::memcpy(depth_mat.data, depth_img.data(), depth_img.size() * sizeof(float));

    cv::Mat bgr_mat = cv::Mat(rows, cols, type);
    std::memcpy(bgr_mat.data, viz_img.data(), viz_img.size() * sizeof(uint8_t));
    cv::Mat rgb_mat;
    cv::cvtColor(bgr_mat, rgb_mat, cv::COLOR_BGR2RGB);

    if (rgb_mat.type() != CV_8UC3)
    {
        throw std::runtime_error("Expected image type CV_8UC3");
    }
    if (rgb_mat.rows != rows || rgb_mat.cols != cols)
    {
        throw std::runtime_error("Image dimensions do not match expected size");
    }
    py::array_t<uint8_t> rgb_result(
        {rgb_mat.rows, rgb_mat.cols, 3},                     // Shape: HxWx3
        {rgb_mat.step[0], rgb_mat.step[1], sizeof(uint8_t)}, // Strides
        rgb_mat.data);                                       // Data pointer
    // Ensure the data is contiguous
    if (!rgb_result.flags() & py::array::c_style)
    {
        throw std::runtime_error("Result array is not C-style contiguous");
    }

    py::array_t<float> depth_result(
        {depth_mat.rows, depth_mat.cols},                     // Shape: HxW
        {depth_mat.step[0], sizeof(float)},                   // Strides
        depth_mat.ptr<float>());                              // Data pointer

    return  {depth_result, rgb_result};
}

// Pybind11 bindings
PYBIND11_MODULE(ransac_bindings, m)
{
    m.doc() = "RANSAC line fitting C++ binding";

    py::class_<RansacResult>(m, "RansacResult") // Fixed: Added struct binding
        .def_readwrite("fitted_lines", &RansacResult::fitted_lines)
        .def_readwrite("inlier_counts", &RansacResult::inlier_counts)
        .def_readwrite("roi_point_clouds", &RansacResult::roi_point_clouds)
        .def_readwrite("point_colors", &RansacResult::point_colors)
        .def_readwrite("masked_viz_img", &RansacResult::masked_viz_img);

    m.def("ransac_line_fitting", &ransac_line_fitting,
          py::arg("points"),
          py::arg("avg_angle"),
          py::arg("num_lines"),
          py::arg("num_iterations"),
          py::arg("inlier_threshold"),
          py::arg("vert_angle_maximum_rad"),
          py::arg("horz_angle_diff_maximum_rad"));

    m.def("ransac_on_rois", &ransac_on_rois,
          py::arg("rois"),
          py::arg("roi_line_counts"),
          py::arg("avg_angle"),
          py::arg("camera_rays"),
          py::arg("depth_image"),
          py::arg("min_depth_clip"),
          py::arg("max_depth_clip"),
          py::arg("ransac_max_iters"),
          py::arg("inlier_threshold_m"),
          py::arg("vert_angle_maximum_rad"),
          py::arg("horz_angle_diff_maximum_rad"),
          py::arg("viz_img") = std::nullopt);

    m.def("test_img_handoff", &test_img_handoff,
          py::arg("depth_img"),
          py::arg("rgb_img"),
          "Test function to ensure RGB image handoff works correctly");
}
