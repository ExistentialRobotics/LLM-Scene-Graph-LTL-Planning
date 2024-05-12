#include "erl_env/environment_base.hpp"

namespace erl::env {

    void
    EnvironmentBase::InflateGridMap2D(
        const cv::Mat &original_grid_map,
        cv::Mat &inflated_grid_map,
        const std::shared_ptr<common::GridMapInfo2D> &grid_map_info,
        const Eigen::Ref<const Eigen::Matrix2Xd> &shape_metric_vertices) {

        // inflate the grid map: rows are x, cols are y
        // create the dilation_kernel and anchor
        std::vector<std::vector<cv::Point>> contours(1);
        auto &contour = contours[0];  // vector of points, each point is a 2D point (col, row), i.e. (y, x)
        long n_vertices = shape_metric_vertices.cols();
        contour.reserve(n_vertices);
        int row_min = std::numeric_limits<int>::max();
        int row_max = -std::numeric_limits<int>::max();
        int col_min = std::numeric_limits<int>::max();
        int col_max = -std::numeric_limits<int>::max();
        for (int i = 0; i < n_vertices; ++i) {
            int x = grid_map_info->MeterToGridForValue(shape_metric_vertices(0, i), 0);
            int y = grid_map_info->MeterToGridForValue(shape_metric_vertices(1, i), 1);
            contour.emplace_back(y, x);
            if (x < row_min) { row_min = x; }
            if (x > row_max) { row_max = x; }
            if (y < col_min) { col_min = y; }
            if (y > col_max) { col_max = y; }
        }
        int row_0 = grid_map_info->MeterToGridForValue(0, 0);
        int col_0 = grid_map_info->MeterToGridForValue(0, 1);
        if (row_0 < row_min) { row_min = row_0; }
        if (row_0 > row_max) { row_max = row_0; }
        if (col_0 < col_min) { col_min = col_0; }
        if (col_0 > col_max) { col_max = col_0; }
        // When anchor is (-1, -1), the anchor is set to the dilation_kernel center.
        // Otherwise, anchor (col, row) should be within the dilation_kernel.
        int kernel_height = row_max - row_min + 1;
        int kernel_width = col_max - col_min + 1;
        ERL_ASSERTM(kernel_height > 0, "kernel_height %d is negative.", kernel_height);
        ERL_ASSERTM(kernel_width > 0, "kernel_width %d is negative.", kernel_width);
        cv::Mat dilation_kernel(kernel_height, kernel_width, CV_8UC1, cv::Scalar(0));
        for (auto &point: contour) {
            point.x -= col_min;
            point.y -= row_min;
        }
        cv::drawContours(dilation_kernel, contours, 0, 1, cv::FILLED, cv::LINE_8);

        // dilate the grid map
        cv::Point anchor(col_0 - col_min, row_0 - row_min);
        // dst(x, y) = max_{(x', y') of nonzero elements in kernel} src(x + x' - anchor.x, y + y' - anchor.y)
        cv::dilate(original_grid_map, inflated_grid_map, dilation_kernel, anchor, 1);
    }
}  // namespace erl::env
