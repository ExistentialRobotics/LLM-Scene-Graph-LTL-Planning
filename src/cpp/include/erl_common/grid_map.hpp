#pragma once

#include "template_helper.hpp"
#include "grid_map_info.hpp"
#include "tensor.hpp"

#include <memory>
#include <mutex>
#include <shared_mutex>

namespace erl::common {

    template<typename T, int Dim, bool RowMajor = true>
    struct GridMap {  // for 2D, row-axis is x, column-axis is y

        Tensor<T, Dim, RowMajor> data;
        std::shared_ptr<GridMapInfo<Dim>> info;

        explicit GridMap(std::shared_ptr<GridMapInfo<Dim>> grid_map_info)
            : data(grid_map_info->Shape()),
              info(std::move(grid_map_info)) {}

        GridMap(std::shared_ptr<GridMapInfo<Dim>> grid_map_info, T value)
            : data(grid_map_info->Shape(), value),
              info(std::move(grid_map_info)) {}

        GridMap(std::shared_ptr<GridMapInfo<Dim>> grid_map_info, Tensor<T, Dim, RowMajor> data)
            : data(std::move(data)),
              info(std::move(grid_map_info)) {}

        GridMap(std::shared_ptr<GridMapInfo<Dim>> grid_map_info, Eigen::VectorX<T> data)
            : data(grid_map_info->Shape(), data),
              info(std::move(grid_map_info)) {}

        GridMap(std::shared_ptr<GridMapInfo<Dim>> grid_map_info, const std::function<T()> &data_init_func)
            : data(grid_map_info->Shape(), data_init_func),
              info(std::move(grid_map_info)) {}
    };

    template<typename T, int Dim, bool RowMajor>
    std::ostream &
    operator<<(std::ostream &os, const GridMap<T, Dim, RowMajor> &grid_map) {
        grid_map.data.Print(os);
        return os;
    }

    template<typename T, bool RowMajor = true>
    using GridMapX = GridMap<T, Eigen::Dynamic, RowMajor>;

    typedef GridMap<double, 2> GridMapDouble2D;
    typedef GridMap<int, 2> GridMapInt2D;
    typedef GridMap<uint8_t, 2> GridMapUnsigned2D;
    typedef GridMap<double, 3> GridMapDouble3D;
    typedef GridMap<int, 3> GridMapInt3D;
    typedef GridMap<uint8_t, 3> GridMapUnsigned3D;
    typedef GridMapX<double> GridMapDoubleXd;
    typedef GridMapX<int> GridMapIntXd;
    typedef GridMapX<uint8_t> GridMapUnsignedXd;

    template<typename T>
    class IncrementalGridMap2D {
        std::shared_ptr<GridMapInfo2D> m_grid_map_info_;
        Eigen::MatrixX<T> m_data_;
        std::function<T()> m_data_init_func_;
        mutable std::shared_mutex m_mutex_;  // mutable for const methods, m_mutex_ is for thread-safe of m_data_

        enum ExtendCode {
            kToTopLeft = 0b1001,
            kToTopCentral = 0b1000,
            kToTopRight = 0b1010,
            kToCentralLeft = 0b0001,
            kToCentralRight = 0b0010,
            kToBottomLeft = 0b0101,
            kToBottomCentral = 0b0100,
            kToBottomRight = 0b0110,
            kNoExtend = 0b0000
        };

    public:
        IncrementalGridMap2D() = delete;

        explicit IncrementalGridMap2D(std::shared_ptr<GridMapInfo2D> grid_map_info, const std::function<T()> &data_init_func = {})
            : m_grid_map_info_(std::move(grid_map_info)),
              m_data_(m_grid_map_info_->Shape(0), m_grid_map_info_->Shape(1)),
              m_data_init_func_(data_init_func) {
            ERL_ASSERTM(m_data_.cols() > 0 && m_data_.rows() > 0, "The shape of the grid map must be positive.");
        }

        [[nodiscard]] std::shared_ptr<GridMapInfo2D>
        GetGridMapInfo() const {
            return m_grid_map_info_;
        }

        [[nodiscard]] Eigen::Vector2d
        GetCanonicalMetricCoords(const Eigen::Ref<const Eigen::Vector2d> &metric_coords) const {
            return {
                m_grid_map_info_->GridToMeterForValue(m_grid_map_info_->MeterToGridForValue(metric_coords[0], 0), 0),
                m_grid_map_info_->GridToMeterForValue(m_grid_map_info_->MeterToGridForValue(metric_coords[1], 1), 1)};
        }

        [[nodiscard]] Eigen::MatrixX8U
        AsImage(const std::shared_ptr<GridMapInfo2D> &grid_map_info, const std::function<uint8_t(const T &)> &cast_func) const {
            Eigen::MatrixX8U image;
            long n_rows = m_data_.rows();
            long n_cols = m_data_.cols();

            if (grid_map_info == nullptr) {
                image.resize(m_data_.rows(), m_data_.cols());
                for (int i = 0; i < n_rows; ++i) {
                    for (int j = 0; j < n_cols; ++j) {
                        auto &data = m_data_(i, j);
                        image(i, j) = cast_func(data);
                    }
                }
            } else {
                image.setConstant(grid_map_info->Shape(0), grid_map_info->Shape(1), 0);
                for (int i = 0; i < n_rows; ++i) {
                    double x = m_grid_map_info_->GridToMeterForValue(i, 0);
                    int ii = grid_map_info->MeterToGridForValue(x, 0);
                    for (int j = 0; j < n_cols; ++j) {
                        auto &data = m_data_(i, j);
                        double y = m_grid_map_info_->GridToMeterForValue(j, 1);
                        if (!grid_map_info->InMap(Eigen::Vector2d(x, y))) { continue; }
                        int jj = grid_map_info->MeterToGridForValue(y, 1);
                        image(ii, jj) = cast_func(data);
                    }
                }
            }

            return image;
        }

        /**
         * @brief Get the data at the grid coordinates. If the point is not in the grid map, return 0 if T is smart pointer, otherwise throw std::out_of_range.
         * @param x_grid
         * @param y_grid
         * @return
         */
        T
        operator()(int x_grid, int y_grid) const {
            if (x_grid < 0 || y_grid < 0 || x_grid >= m_grid_map_info_->Shape(0) || y_grid >= m_grid_map_info_->Shape(1)) {
                if (!IsSmartPtr<T>::value) { throw std::out_of_range("The grid coordinates are out of range."); }
                return 0;
            }
            return m_data_(x_grid, y_grid);
        }

        /**
         * @brief Get the data at the grid coordinates. If the point is not in the grid map, return nullptr if T is smart pointer or throw std::out_of_range
         * otherwise. The data is unmodifiable.
         * @param grid_coords
         * @return
         */
        T
        operator[](const Eigen::Ref<const Eigen::Vector2i> &grid_coords) const {
            return operator()(grid_coords[0], grid_coords[1]);
        }

        T
        operator()(const double x, const double y) const {
            return operator()(m_grid_map_info_->MeterToGridForValue(x, 0), m_grid_map_info_->MeterToGridForValue(y, 1));
        }

        /**
         * @brief Get the data at the metric coordinates. If the point is not in the grid map, return nullptr if T is smart pointer or throw std::out_of_range
         * otherwise. The data is unmodifiable.
         * @param metric_coords
         * @return
         */
        T
        operator[](const Eigen::Ref<const Eigen::Vector2d> &metric_coords) const {
            return operator()(metric_coords[0], metric_coords[1]);
        }

        T &
        GetMutableData(int x_grid, int y_grid) {
            ERL_DEBUG_ASSERT(
                x_grid >= 0 && y_grid >= 0 && x_grid < m_grid_map_info_->Shape(0) && y_grid < m_grid_map_info_->Shape(1),
                "The grid coordinates are out of the grid map, auto extend is not working properly.");
            auto &data = m_data_(x_grid, y_grid);
            if (IsSmartPtr<T>::value && m_data_init_func_ != nullptr && data == 0) { data = m_data_init_func_(); }
            return data;
        }

        T &
        GetMutableData(const Eigen::Ref<const Eigen::Vector2i> &grid_coords) {
            return GetMutableData(grid_coords[0], grid_coords[1]);
        }

        T &
        GetMutableData(const double x, const double y) {
            ERL_DEBUG_ASSERT(!omp_in_parallel(), "The grid map is not thread safe.");
            int x_grid = m_grid_map_info_->MeterToGridForValue(x, 0);
            int y_grid = m_grid_map_info_->MeterToGridForValue(y, 1);
            while (x_grid < 0 || y_grid < 0 || x_grid >= m_grid_map_info_->Shape(0) || y_grid >= m_grid_map_info_->Shape(1)) {
                Extend(GetExtendCode(x_grid, y_grid));
                x_grid = m_grid_map_info_->MeterToGridForValue(x, 0);
                y_grid = m_grid_map_info_->MeterToGridForValue(y, 1);
            }
            return GetMutableData(x_grid, y_grid);
        }

        T &
        GetMutableData(const Eigen::Ref<const Eigen::Vector2d> &metric_coords) {
            return GetMutableData(metric_coords[0], metric_coords[1]);
        }

        T &
        GetMutableDataThreadSafe(const double x, const double y) {
            std::lock_guard lock(m_mutex_);
            return GetMutableData(x, y);
        }

        T &
        GetMutableDataThreadSafe(const Eigen::Ref<const Eigen::Vector2d> &metric_coords) {
            return GetMutableDataThreadSafe(metric_coords[0], metric_coords[1]);
        }

        Eigen::Ref<Eigen::MatrixX<T>>
        GetBlock(int x_grid, int y_grid, int height, int width) {
            ERL_DEBUG_ASSERT(
                x_grid >= 0 && y_grid >= 0 && (x_grid + height <= m_grid_map_info_->Shape(0)) && (y_grid + width <= m_grid_map_info_->Shape(1)),
                "The grid coordinates (%d, %d) are out of the grid map or the block size (%d, %d) is too large.",
                x_grid,
                y_grid,
                height,
                width);
            return m_data_.block(x_grid, y_grid, height, width);
        }

        Eigen::Ref<Eigen::MatrixX<T>>
        GetBlock(const double x_min, const double y_min, const double x_max, const double y_max, const bool safe_crop = true) {
            int x_min_grid = m_grid_map_info_->MeterToGridForValue(x_min, 0);
            int y_min_grid = m_grid_map_info_->MeterToGridForValue(y_min, 1);
            int x_max_grid = m_grid_map_info_->MeterToGridForValue(x_max, 0);
            int y_max_grid = m_grid_map_info_->MeterToGridForValue(y_max, 1);
            if (safe_crop) {
                x_min_grid = std::max(x_min_grid, 0);
                y_min_grid = std::max(y_min_grid, 0);
                x_max_grid = std::min(x_max_grid, m_grid_map_info_->Shape(0) - 1);
                y_max_grid = std::min(y_max_grid, m_grid_map_info_->Shape(1) - 1);
            }
            return GetBlock(x_min_grid, y_min_grid, x_max_grid - x_min_grid + 1, y_max_grid - y_min_grid + 1);
        }

        Eigen::Ref<Eigen::MatrixX<T>>
        GetBlock(const Eigen::Ref<const Eigen::Vector2d> &metric_min, const Eigen::Ref<const Eigen::Vector2d> &metric_max, bool safe_crop = true) {
            return GetBlock(metric_min[0], metric_min[1], metric_max[0], metric_max[1], safe_crop);
        }

        void
        CollectNonZeroData(double x_min, double y_min, double x_max, double y_max, std::vector<T> &data) {
            int x_min_grid = m_grid_map_info_->MeterToGridForValue(x_min, 0);
            int y_min_grid = m_grid_map_info_->MeterToGridForValue(y_min, 1);
            int x_max_grid = m_grid_map_info_->MeterToGridForValue(x_max, 0);
            int y_max_grid = m_grid_map_info_->MeterToGridForValue(y_max, 1);
            x_min_grid = std::max(x_min_grid, 0);
            y_min_grid = std::max(y_min_grid, 0);
            x_max_grid = std::min(x_max_grid, m_grid_map_info_->Shape(0) - 1);
            y_max_grid = std::min(y_max_grid, m_grid_map_info_->Shape(1) - 1);
            for (int i = x_min_grid; i <= x_max_grid; ++i) {
                for (int j = y_min_grid; j <= y_max_grid; ++j) {
                    auto &element = m_data_(i, j);
                    if (element != 0) { data.push_back(element); }
                }
            }
        }

        void
        CollectNonZeroData(const Eigen::Ref<const Eigen::Vector2d> &metric_min, const Eigen::Ref<const Eigen::Vector2d> &metric_max, std::vector<T> &data) {
            CollectNonZeroData(metric_min[0], metric_min[1], metric_max[0], metric_max[1], data);
        }

    private:
        /**
         * @brief Get the extend code for a grid point not in the grid map.
         * @param x
         * @param y
         * @return
         * @refitem https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm
         */
        ExtendCode
        GetExtendCode(const int x, const int y) const {

            int code = 0;

            const int &kXSize = m_grid_map_info_->Shape()[0];
            if (x < 0) {
                code |= 0b1000;  // top
            } else if (x >= kXSize) {
                code |= 0b0100;  // bottom
            }

            const int &kYSize = m_grid_map_info_->Shape()[1];
            if (y < 0) {
                code |= 0b0001;  // left
            } else if (y >= kYSize) {
                code |= 0b0010;  // right
            }

            return static_cast<ExtendCode>(code);
        }

        void
        Extend(ExtendCode code) {
            if (code == kNoExtend) { return; }

            double x_min = m_grid_map_info_->Min(0);
            double y_min = m_grid_map_info_->Min(1);
            double x_max = m_grid_map_info_->Max(0);
            double y_max = m_grid_map_info_->Max(1);
            double x_range = x_max - x_min;
            double y_range = y_max - y_min;
            double x_res = m_grid_map_info_->Resolution(0);
            double y_res = m_grid_map_info_->Resolution(1);
            std::shared_ptr<GridMapInfo2D> new_grid_map_info;
            double new_x_min = x_min, new_y_min = y_min, new_x_max = x_max, new_y_max = y_max;
            switch (code) {
                case kToCentralLeft:
                case kToTopLeft:
                    new_x_min = x_min - x_range - x_res;
                    new_y_min = y_min - y_range - y_res;
                    new_x_max = x_max;
                    new_y_max = y_max;
                    break;
                case kToTopCentral:
                case kToTopRight:
                    new_x_min = x_min - x_range - x_res;
                    new_y_min = y_min;
                    new_x_max = x_max;
                    new_y_max = y_max + y_range + y_res;
                    break;
                case kToCentralRight:
                case kToBottomRight:
                    new_x_min = x_min;
                    new_y_min = y_min;
                    new_x_max = x_max + x_range + x_res;
                    new_y_max = y_max + y_range + y_res;
                    break;
                case kToBottomCentral:
                case kToBottomLeft:
                    new_x_min = x_min;
                    new_y_min = y_min - y_range - y_res;
                    new_x_max = x_max + x_range + x_res;
                    new_y_max = y_max;
                    break;
                case kNoExtend:
                    return;
            }
            long n_rows = m_data_.rows();
            long n_cols = m_data_.cols();
            new_grid_map_info = std::make_shared<GridMapInfo2D>(
                Eigen::Vector2i(n_rows * 2 + 1, n_cols * 2 + 1),
                Eigen::Vector2d(new_x_min, new_y_min),
                Eigen::Vector2d(new_x_max, new_y_max));
            ERL_ASSERTM(std::abs(new_grid_map_info->Resolution(0) - x_res) < 1.e-10, "x resolution is not equal.");
            ERL_ASSERTM(std::abs(new_grid_map_info->Resolution(1) - y_res) < 1.e-10, "y resolution is not equal.");

            Eigen::MatrixX<T> new_data(new_grid_map_info->Shape(0), new_grid_map_info->Shape(1));
            // copy the old data matrix to the new data matrix
            Eigen::Vector2i loc = new_grid_map_info->MeterToGridForPoints(Eigen::Vector2d(x_min, y_min));
            new_data.block(loc[0], loc[1], n_rows, n_cols) = m_data_;

            // swap the results
            m_grid_map_info_ = std::move(new_grid_map_info);
            m_data_.swap(new_data);
        }
    };

}  // namespace erl::common
