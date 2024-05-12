#include "erl_common/opencv.hpp"

namespace erl::common {

    void
    ColorGrayCustom(const cv::Mat &gray, cv::Mat &custom) {
        // gray must be CV_32SC1
        ERL_ASSERTM(gray.type() == CV_32SC1, "gray must be CV_32SC1.");
        ERL_ASSERTM(gray.channels() == 1, "gray must have 1 channel.");
        std::unordered_set<int> unique_values;
        for (int row = 0; row < gray.rows; ++row) {
            for (int col = 0; col < gray.cols; ++col) { unique_values.insert(gray.at<int>(row, col)); }
        }
        std::vector unique_values_vec(unique_values.begin(), unique_values.end());
        std::sort(unique_values_vec.begin(), unique_values_vec.end());  // ascending order
        std::unordered_map<int, std::size_t> value_to_index;
        for (std::size_t i = 0; i < unique_values_vec.size(); ++i) { value_to_index[unique_values_vec[i]] = i; }
        custom = cv::Mat::zeros(gray.rows, gray.cols, CV_8UC3);
        for (int row = 0; row < gray.rows; ++row) {
            for (int col = 0; col < gray.cols; ++col) {
                int value = gray.at<int>(row, col);
                std::size_t index = value_to_index[value];
                custom.at<cv::Vec3b>(row, col) = kCustomColorMap[index];
            }
        }
    }

    void
    ColorGrayToJet(const cv::Mat &gray, cv::Mat &jet, const bool normalize) {
        cv::Mat gray_uint8;

        if (normalize) {
            cv::Mat gray_float;
            gray.convertTo(gray_float, CV_32FC1);
            cv::normalize(gray_float, gray_float, 0, 1, cv::NORM_MINMAX);
            gray_float.convertTo(gray_uint8, CV_8UC1, 255.0);
        } else {
            gray.convertTo(gray_uint8, CV_8UC1);
        }

        cv::applyColorMap(gray_uint8, jet, cv::COLORMAP_JET);
    }

    cv::Mat
    AlphaBlending(const cv::Mat &foreground, const cv::Mat &background) {
        ERL_ASSERTM(foreground.channels() == 4, "Foreground image must have 4 channels.");
        ERL_ASSERTM(background.channels() == 4, "Background image must have 4 channels.");
        ERL_ASSERTM(foreground.size() == background.size(), "Foreground and background image must have the same size.");

        cv::Mat channels[4];
        cv::split(foreground, channels);
        cv::Mat alpha_fg;
        cv::cvtColor(channels[3], alpha_fg, cv::COLOR_GRAY2RGBA);
        alpha_fg.convertTo(alpha_fg, CV_32FC4, 1.0 / 255.0);

        cv::Mat alpha_bg;
        alpha_bg = cv::Scalar::all(1.0) - alpha_fg;

        cv::Mat alpha_foreground;
        foreground.convertTo(alpha_foreground, CV_32FC4);
        cv::multiply(alpha_fg, alpha_foreground, alpha_foreground);

        cv::Mat alpha_background;
        background.convertTo(alpha_background, CV_32FC4);
        cv::multiply(alpha_bg, alpha_background, alpha_background);

        cv::Mat out_image(alpha_foreground.size(), alpha_foreground.type());
        cv::add(alpha_foreground, alpha_background, out_image);  // it seems only the first 3 channels are modified

        out_image.convertTo(out_image, CV_8UC4);
        cv::split(out_image, channels);
        channels[3].setTo(255);  // set alpha channel to 255
        cv::merge(channels, 4, out_image);

        return out_image;
    }

    cv::Mat &
    DrawTrajectoryInplace(
        cv::Mat &map,
        const Eigen::Ref<const Eigen::Matrix2Xd> &trajectory,
        const std::shared_ptr<GridMapInfo2D> &grid_map_info,
        const cv::Scalar &color,
        const int thickness,
        const bool pixel_based) {

        const long num_points = trajectory.cols();
        if (num_points == 0) { return map; }

        std::vector<cv::Point2i> points;
        points.reserve(num_points);
        if (pixel_based) {
            for (long i = 0; i < num_points; ++i) {
                points.emplace_back(
                    grid_map_info->MeterToGridForValue(trajectory(0, i), 0),
                    grid_map_info->Shape(1) - grid_map_info->MeterToGridForValue(trajectory(1, i), 1));
            }
        } else {
            for (long i = 0; i < num_points; ++i) {
                points.emplace_back(grid_map_info->MeterToGridForValue(trajectory(1, i), 1), grid_map_info->MeterToGridForValue(trajectory(0, i), 0));
            }
        }
        cv::polylines(map, points, false, color, thickness);
        return map;
    }
}  // namespace erl::common
