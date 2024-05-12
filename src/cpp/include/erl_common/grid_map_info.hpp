#pragma once

#include "storage_order.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace erl::common {

    inline int
    MeterToGrid(const double meter, const double meter_min, const double resolution) {
        return static_cast<int>(std::floor((meter - meter_min) / resolution));
    }

    inline double
    GridToMeter(const int grid, const double meter_min, const double resolution) {
        // a grid should cover a valid range, e.g. if the grid is 0, the valid range is [meter_min, meter_min + resolution)
        // If we use rounding when converting from meter to grid, the covered range will be [meter_min - 0.5 * resolution, meter_min + 0.5 * resolution)
        // Half of the range is invalid. Therefore, we need to add 0.5 to the grid index to get the correct range.
        // If we use floor instead, the covered range will be [meter_min, meter_min + resolution).
        return (static_cast<double>(grid) + 0.5) * resolution + meter_min;
    }

    /**
     * GridMapInfo defines
     * 1. the mapping between right-handed n-dim world system and right-handed n-dim grid map (xy-indexing).
     * 2. hashing of grid coordinates with respect to the specific storing order (row-major or column-major).
     */
    template<int Dim>
    class GridMapInfo {

        Eigen::Vector<int, Dim> m_map_shape_;
        Eigen::Vector<double, Dim> m_resolution_;
        Eigen::Vector<double, Dim> m_min_;
        Eigen::Vector<double, Dim> m_max_;
        Eigen::Vector<double, Dim> m_center_;
        Eigen::Vector<int, Dim> m_center_grid_;

    public:
        GridMapInfo(
            const Eigen::Vector<double, Dim>& min,
            const Eigen::Vector<double, Dim>& max,
            const Eigen::Vector<double, Dim>& resolution,
            Eigen::Vector<int, Dim> padding)
            : m_map_shape_(Eigen::Vector<int, Dim>(
                  ((max - min).array() / resolution.array()).ceil().template cast<int>().unaryExpr([](const int& x) { return x % 2 ? x + 1 : x; }).array() + 1 +
                  2 * padding.array())),
              m_resolution_((max - min).array() / (m_map_shape_.array() - 2 * padding.array()).template cast<double>().array()),
              m_min_(min.array() - m_resolution_.array() * padding.template cast<double>().array()),
              m_max_(max.array() + m_resolution_.array() * padding.template cast<double>().array()),
              m_center_((m_min_ + m_max_) / 2.0),
              m_center_grid_(m_map_shape_.array() / 2) {}

        GridMapInfo(const Eigen::Vector<int, Dim>& map_shape, const Eigen::Vector<double, Dim>& min, const Eigen::Vector<double, Dim>& max)
            : m_map_shape_(Eigen::Vector<int, Dim>(map_shape.unaryExpr([](int x) { return x % 2 ? x : x + 1; }))),
              m_resolution_((max - min).array() / m_map_shape_.template cast<double>().array()),
              m_min_(min),
              m_max_(max),
              m_center_((m_min_ + m_max_) / 2),
              m_center_grid_(m_map_shape_.array() / 2) {
            if (Dim == Eigen::Dynamic) {
                ERL_DEBUG_ASSERT(m_map_shape_.size() > 0, "0-dim map is not allowed!");
                ERL_DEBUG_ASSERT(Size() > 0, "0-element map is not allowed!");
            }
        }

        GridMapInfo(const Eigen::Vector<double, Dim>& origin, const Eigen::Vector<double, Dim>& resolution, const Eigen::Vector<int, Dim>& map_shape)
            : m_map_shape_(Eigen::Vector<int, Dim>(map_shape.unaryExpr([](int x) { return x % 2 ? x : x + 1; }))),
              m_resolution_(resolution),
              m_min_(origin),
              m_max_(origin.array() + resolution.array() * map_shape.template cast<double>().array()),
              m_center_((m_min_ + m_max_) / 2),
              m_center_grid_(m_map_shape_.array() / 2) {}

        explicit GridMapInfo(const GridMapInfo<Eigen::Dynamic>& info)
            : m_map_shape_(info.Shape()),
              m_resolution_(info.Resolution()),
              m_min_(info.Min()),
              m_max_(info.Max()),
              m_center_(info.Center()),
              m_center_grid_(info.CenterGrid()) {}

        [[nodiscard]] GridMapInfo<Eigen::Dynamic>
        Extend(int size, double min, double max, int dim) const {

            int n_dims = Dims();
            ERL_DEBUG_ASSERT(dim >= 0 && dim <= n_dims, "dim = %d is out of range [%d, %d].", dim, 0, n_dims);

            Eigen::VectorXi new_shape(n_dims + 1);
            Eigen::VectorXd new_min(n_dims + 1);
            Eigen::VectorXd new_max(n_dims + 1);

            for (int i = 0; i <= n_dims; ++i) {
                if (i < dim) {
                    new_shape[i] = m_map_shape_[i];
                    new_min[i] = m_min_[i];
                    new_max[i] = m_max_[i];
                } else if (i == dim) {
                    if (size % 2 == 0) { size++; }  // make sure size is odd
                    new_shape[i] = size;
                    new_min[i] = min;
                    new_max[i] = max;
                } else {
                    new_shape[i] = m_map_shape_[i - 1];
                    new_min[i] = m_min_[i - 1];
                    new_max[i] = m_max_[i - 1];
                }
            }

            return {new_shape, new_min, new_max};
        }

        [[nodiscard]] GridMapInfo<Eigen::Dynamic>
        Extend(const double min, const double max, const double resolution, const int padding, const int dim) const {

            int n_dims = Dims();
            ERL_DEBUG_ASSERT(dim >= 0 && dim <= n_dims, "dim = %d is out of range [%d, %d]", dim, 0, n_dims);

            Eigen::VectorXi new_map_shape(n_dims + 1);
            Eigen::VectorXd new_min(n_dims + 1);
            Eigen::VectorXd new_resolution(n_dims + 1);

            for (int i = 0; i <= n_dims; ++i) {
                if (i < dim) {
                    new_map_shape[i] = m_map_shape_[i];
                    new_min[i] = m_min_[i];
                    new_resolution[i] = m_resolution_[i];
                } else if (i == dim) {
                    new_map_shape[i] = static_cast<int>(std::ceil((max - min) / resolution));
                    if (new_map_shape[i] % 2) { ++new_map_shape[i]; }
                    new_resolution[i] = (max - min) / new_map_shape[i];
                    new_map_shape[i] += 1 + 2 * padding;
                    new_min[i] = min - new_resolution[i] * padding;
                } else {
                    new_map_shape[i] = m_map_shape_[i - 1];
                    new_min[i] = m_min_[i - 1];
                    new_resolution[i] = m_resolution_[i - 1];
                }
            }

            return {new_map_shape, new_min, new_resolution};
        }

        [[nodiscard]] GridMapInfo<Eigen::Dynamic>
        Squeeze(const int dim) const {
            const int n_dims = Dims();
            ERL_DEBUG_ASSERT(dim >= 0 && dim < n_dims, "dim = %d is out of range [%d, %d)", dim, 0, n_dims);

            Eigen::VectorXi new_map_shape(n_dims - 1);
            Eigen::VectorXd new_min(n_dims - 1);
            Eigen::VectorXd new_resolution(n_dims - 1);

            for (int i = 0; i < n_dims; ++i) {
                if (i < dim) {
                    new_map_shape[i] = m_map_shape_[i];
                    new_min[i] = m_min_[i];
                    new_resolution[i] = m_resolution_[i];
                }
            }

            return {new_map_shape, new_min, new_resolution};
        }

        [[nodiscard]] long
        Dims() const {
            return m_map_shape_.size();
        }

        [[nodiscard]] const Eigen::Vector<int, Dim>&
        Shape() const {
            return m_map_shape_;
        }

        [[nodiscard]] int
        Shape(int dim) const {
            return m_map_shape_[dim];
        }

        [[nodiscard]] int
        Size() const {
            if (Dims()) { return m_map_shape_.prod(); }
            return 0;
        }

        [[nodiscard]] int
        Rows() const {
            return m_map_shape_[0];
        }

        [[nodiscard]] int
        Cols() const {
            return m_map_shape_[1];
        }

        [[nodiscard]] int
        Width() const {
            return m_map_shape_[0];
        }

        [[nodiscard]] int
        Height() const {
            return m_map_shape_[1];
        }

        [[nodiscard]] int
        Length() const {
            if (Dims() >= 3) { return m_map_shape_[2]; }
            return 0;
        }

        [[nodiscard]] const Eigen::Vector<double, Dim>&
        Min() const {
            return m_min_;
        }

        [[nodiscard]] double
        Min(int dim) const {
            return m_min_[dim];
        }

        [[nodiscard]] const Eigen::Vector<double, Dim>&
        Max() const {
            return m_max_;
        }

        [[nodiscard]] double
        Max(int dim) const {
            return m_max_[dim];
        }

        [[nodiscard]] const Eigen::Vector<double, Dim>&
        Resolution() const {
            return m_resolution_;
        }

        [[nodiscard]] double
        Resolution(int dim) const {
            return m_resolution_[dim];
        }

        [[nodiscard]] const Eigen::Vector<double, Dim>&
        Center() const {
            return m_center_;
        }

        [[nodiscard]] const Eigen::Vector<int, Dim>&
        CenterGrid() const {
            return m_center_grid_;
        }

        [[nodiscard]] Eigen::VectorXd
        GetDimLinSpace(int dim) const {
            return Eigen::VectorX<double>::LinSpaced(Shape(dim), Min(dim), Max(dim) - Resolution(dim));
        }

        [[nodiscard]] double
        GridToMeterForValue(int grid_value, int dim) const {
            return GridToMeter(grid_value, m_min_[dim], m_resolution_[dim]);
        }

        [[nodiscard]] Eigen::VectorXd
        GridToMeterForValues(const Eigen::Ref<const Eigen::VectorXi>& grid_values, int dim) const {
            const double& min = m_min_[dim];
            const double& res = m_resolution_[dim];
            return grid_values.unaryExpr([&](int v) -> double { return GridToMeter(v, min, res); });
        }

        [[nodiscard]] int
        MeterToGridForValue(const double meter_value, const int dim) const {
            return MeterToGrid(meter_value, m_min_[dim], m_resolution_[dim]);
        }

        [[nodiscard]] Eigen::VectorXi
        MeterToGridForValues(const Eigen::Ref<const Eigen::VectorXd>& meter_values, const int dim) const {
            const double& min = m_min_[dim];
            const double& res = m_resolution_[dim];
            return meter_values.unaryExpr([&](const double v) -> int { return MeterToGrid(v, min, res); });
        }

        [[nodiscard]] Eigen::Matrix<double, Dim, Eigen::Dynamic>
        GridToMeterForPoints(const Eigen::Ref<const Eigen::Matrix<int, Dim, Eigen::Dynamic>>& grid_points) const {
            long n_rows = grid_points.rows();
            long n_cols = grid_points.cols();
            Eigen::Matrix<double, Dim, Eigen::Dynamic> meter_points(n_rows, n_cols);
            for (long i = 0; i < n_rows; ++i) {
                for (long j = 0; j < n_cols; ++j) { meter_points(i, j) = GridToMeter(grid_points(i, j), m_min_[i], m_resolution_[i]); }
            }
            return meter_points;
        }

        [[nodiscard]] Eigen::Matrix<int, Dim, Eigen::Dynamic>
        MeterToGridForPoints(const Eigen::Ref<const Eigen::Matrix<double, Dim, Eigen::Dynamic>>& meter_points) const {
            long n_rows = meter_points.rows();
            long n_cols = meter_points.cols();
            Eigen::Matrix<int, Dim, Eigen::Dynamic> grid_points(n_rows, n_cols);
            for (long i = 0; i < n_rows; ++i) {
                for (long j = 0; j < n_cols; ++j) { grid_points(i, j) = MeterToGrid(meter_points(i, j), m_min_[i], m_resolution_[i]); }
            }
            return grid_points;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == -1, Eigen::Matrix2Xi>
        GridToPixelForPoints(const Eigen::Ref<const Eigen::Matrix2Xi>& grid_points) const {
            if (D == Eigen::Dynamic) { ERL_DEBUG_ASSERT(Dims() == 2, "Not available when Dims() != 2"); }

            Eigen::Matrix2Xi pixel_points(2, grid_points.cols());
            // m_map_shape_[0] <--> width
            // m_map_shape_[1] <--> height
            // [x, y] -> [height - y, x] (ij indexing) -> [x, height - y] (xy indexing, used by OpenCV)
            long n_cols = grid_points.cols();
            for (long j = 0; j < n_cols; ++j) {
                pixel_points(0, j) = grid_points(0, j);
                pixel_points(1, j) = m_map_shape_[1] - grid_points(1, j);
            }
            return pixel_points;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2Xi>
        PixelToGridForPoints(const Eigen::Ref<const Eigen::Matrix2Xi>& pixel_points) const {
            return GridToPixelForPoints(pixel_points);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2Xi>
        MeterToPixelForPoints(const Eigen::Ref<const Eigen::Matrix2Xd>& meter_points) const {
            return GridToPixelForPoints(MeterToGridForPoints(meter_points));
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2Xd>
        PixelToMeterForPoints(const Eigen::Ref<const Eigen::Matrix2Xi>& pixel_points) const {
            return GridToMeterForPoints(PixelToGridForPoints(pixel_points));
        }

        [[nodiscard]] Eigen::MatrixXd
        GridToMeterForVectors(const Eigen::Ref<const Eigen::MatrixXi>& grid_vectors) const {
            return grid_vectors.cast<double>().array().colwise() * m_resolution_.array();
        }

        [[nodiscard]] Eigen::MatrixXi
        MeterToGridForVectors(const Eigen::Ref<const Eigen::MatrixXd>& meter_vectors) const {
            return (meter_vectors.array().colwise() / m_resolution_.array()).floor().template cast<int>();
        }

        template<int D = Dim>
        [[nodiscard]] static std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2Xi>
        GridToPixelForVectors(const Eigen::Ref<const Eigen::Matrix2Xi>& grid_vectors) {
            if (D == Eigen::Dynamic) { ERL_DEBUG_ASSERT(Dims() == 2, "Not available when Dims() != 2"); }

            Eigen::Matrix2Xi pixel_vectors(2, grid_vectors.cols());
            long n_cols = grid_vectors.cols();
            for (long j = 0; j < n_cols; ++j) {
                pixel_vectors(0, j) = grid_vectors(0, j);
                pixel_vectors(1, j) = -grid_vectors(1, j);
            }

            return pixel_vectors;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2Xi>
        PixelToGridForVectors(const Eigen::Ref<const Eigen::Matrix2Xi>& pixel_vectors) const {
            return GridToPixelForVectors(pixel_vectors);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2Xi>
        MeterToPixelForVectors(const Eigen::Ref<const Eigen::Matrix2Xd>& meter_vectors) const {
            return GridToPixelForVectors(MeterToGridForVectors(meter_vectors));
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix2Xd>
        PixelToMeterForVectors(const Eigen::Ref<const Eigen::Matrix2Xi>& pixel_vectors) const {
            return GridToMeterForVectors(PixelToGridForVectors(pixel_vectors));
        }

        [[nodiscard]] bool
        InMap(const Eigen::Ref<const Eigen::Vector<double, Dim>>& meter_point) const {
            ERL_DEBUG_ASSERT(
                meter_point.size() == m_map_shape_.size(),
                "meter_point is %td-dim but the map is %td-dim.",
                meter_point.size(),
                m_map_shape_.size());
            for (int i = 0; i < meter_point.size(); ++i) {
                if (meter_point[i] < m_min_[i] || meter_point[i] > m_max_[i]) { return false; }
            }
            return true;
        }

        [[nodiscard]] bool
        InGrids(const Eigen::Ref<const Eigen::Vector<int, Dim>>& grid_point) const {
            // clang-format off
            if (Dim == 2) {
                return grid_point[0] >= 0 && grid_point[0] < m_map_shape_[0] &&
                       grid_point[1] >= 0 && grid_point[1] < m_map_shape_[1];
            }
            if (Dim == 3) {
                return grid_point[0] >= 0 && grid_point[0] < m_map_shape_[0] &&
                       grid_point[1] >= 0 && grid_point[1] < m_map_shape_[1] &&
                       grid_point[2] >= 0 && grid_point[2] < m_map_shape_[2];
            }
            // clang-format on

            for (int i = 0; i < Dim; ++i) {
                if (grid_point[i] < 0 || grid_point[i] >= m_map_shape_[i]) { return false; }
            }
            return true;
        }

        [[nodiscard]] int
        GridToIndex(const Eigen::Ref<const Eigen::Vector<int, Dim>>& grid, bool c_stride) const {
            ERL_DEBUG_ASSERT(InGrids(grid), "%s is out of map.\n", EigenToNumPyFmtString(grid.transpose()).c_str());
            return CoordsToIndex<Dim>(m_map_shape_, grid, c_stride);
        }

        [[nodiscard]] Eigen::Vector<int, Dim>
        IndexToGrid(int index, bool c_stride) const {
            return IndexToCoords<Dim>(m_map_shape_, index, c_stride);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, int>
        PixelToIndex(const Eigen::Ref<const Eigen::Vector<int, D>>& pixel, bool c_stride) const {
            return GridToIndex(PixelToGridForPoints(pixel), c_stride);
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Vector<int, D>>
        IndexToPixel(const int index, bool c_stride) const {
            return GridToPixelForPoints(IndexToGrid(index, c_stride));
        }

        [[nodiscard]] Eigen::Matrix<int, Dim, Eigen::Dynamic>
        GenerateGridCoordinates(bool c_stride) const {
            int size = Size();
            if (!size) { return {}; }

            Eigen::VectorXi strides;
            if (c_stride) {  // row-major, e.g.
                // coords: [x, y], shape: [r, c]
                // strides: [c, 1], when c = 3
                // x: 1, 1, 1, 2, 2, 2, ...
                // y: 1, 2, 3, 1, 2, 3, ...
                strides = ComputeCStrides<int>(m_map_shape_, 1);
            } else {
                strides = ComputeFStrides<int>(m_map_shape_, 1);
            }

            long n_dims = Dims();
            Eigen::Matrix<int, Dim, Eigen::Dynamic> grid_coords(n_dims, size);
            for (long i = 0; i < n_dims; ++i) {
                int& stride = strides[i];
                int dim_size = Shape(i);
                int n_copies = size / dim_size;
                Eigen::MatrixXi coords = Eigen::VectorXi::LinSpaced(dim_size, 0, dim_size - 1).transpose().replicate(stride, n_copies / stride);
                grid_coords.row(i) << coords.reshaped(size, 1).transpose();
            }

            return grid_coords;
        }

        [[nodiscard]] Eigen::Matrix<double, Dim, Eigen::Dynamic>
        GenerateMeterCoordinates(bool c_stride) const {
            int size = Size();
            if (!size) { return {}; }

            Eigen::VectorXi strides;
            if (c_stride) {  // row-major, e.g.
                // coords: [x, y], shape: [r, c]
                // strides: [c, 1], when c = 3
                // x: 1, 1, 1, 2, 2, 2, ...
                // y: 1, 2, 3, 1, 2, 3, ...
                strides = ComputeCStrides<int>(m_map_shape_, 1);
            } else {
                strides = ComputeFStrides<int>(m_map_shape_, 1);
            }

            long n_dims = Dims();
            Eigen::Matrix<double, Dim, Eigen::Dynamic> meter_coords(n_dims, size);
            for (long i = 0; i < n_dims; ++i) {
                int& stride = strides[i];
                int dim_size = Shape(i);
                int n_copies = size / dim_size;
                double half_res = Resolution(i) * 0.5;
                double min = Min(i) + half_res;
                double max = Max(i) - half_res;
                Eigen::MatrixXd coords = Eigen::VectorXd::LinSpaced(dim_size, min, max).transpose().replicate(stride, n_copies / stride);
                meter_coords.row(i) << coords.reshaped(size, 1).transpose();
            }

            return meter_coords;
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix<double, D, Eigen::Dynamic>>
        GetMetricCoordinatesOfFilledMetricPolygon(const Eigen::Ref<const Eigen::Matrix<double, 2, Eigen::Dynamic>>& polygon_metric_vertices) const {
            return PixelToMeterForPoints(GetPixelCoordinatesOfFilledMetricPolygon<D>(polygon_metric_vertices));
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix<int, D, Eigen::Dynamic>>
        GetGridCoordinatesOfFilledMetricPolygon(const Eigen::Ref<const Eigen::Matrix2Xd>& polygon_metric_vertices) const {
            return PixelToGridForPoints(GetPixelCoordinatesOfFilledMetricPolygon<D>(polygon_metric_vertices));
        }

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2 || D == Eigen::Dynamic, Eigen::Matrix<int, D, Eigen::Dynamic>>
        GetPixelCoordinatesOfFilledMetricPolygon(const Eigen::Ref<const Eigen::Matrix2Xd>& polygon_metric_vertices) const {
            if (D == Eigen::Dynamic) {
                ERL_DEBUG_ASSERT(
                    polygon_metric_vertices.rows() == Dims(),
                    "polygon_metric_vertices is %d-dim but the map is %ld-dim.",
                    polygon_metric_vertices.rows(),
                    Dims());
            }
            ERL_DEBUG_ASSERT(polygon_metric_vertices.cols() >= 3, "polygon_metric_vertices must have at least 3 vertices.");

            int u_min = std::numeric_limits<int>::max();
            int u_max = -std::numeric_limits<int>::max();
            int v_min = std::numeric_limits<int>::max();
            int v_max = -std::numeric_limits<int>::max();
            long num_vertices = polygon_metric_vertices.cols();
            Eigen::Matrix2Xi polygon_pixel_points(2, num_vertices);
            for (long i = 0; i < num_vertices; ++i) {
                polygon_pixel_points.col(i) = MeterToPixelForPoints<D>(polygon_metric_vertices.col(i));
                const int& u = polygon_pixel_points(0, i);
                const int& v = polygon_pixel_points(1, i);
                if (u < u_min) { u_min = u; }
                if (u > u_max) { u_max = u; }
                if (v < v_min) { v_min = v; }
                if (v > v_max) { v_max = v; }
            }

            cv::Mat canvas(v_max - v_min + 1, u_max - u_min + 1, CV_8UC1, cv::Scalar(0));
            std::vector<std::vector<cv::Point>> polygon_points(1);
            auto& points = polygon_points[0];
            for (long i = 0; i < num_vertices; ++i) {  // make sure all points are in the canvas
                points.emplace_back(polygon_pixel_points(0, i) - u_min, polygon_pixel_points(1, i) - v_min);
            }
            cv::drawContours(canvas, polygon_points, 0, cv::Scalar(255), cv::FILLED, cv::LINE_8);

            polygon_pixel_points.conservativeResize(Eigen::NoChange, canvas.rows * canvas.cols);
            num_vertices = 0;
            for (int i = 0; i < canvas.rows; ++i) {
                for (int j = 0; j < canvas.cols; ++j) {  // pick filled pixels
                    if (canvas.at<uint8_t>(i, j) == 255) { polygon_pixel_points.col(num_vertices++) = Eigen::Vector2i(j + u_min, i + v_min); }
                }
            }
            polygon_pixel_points.conservativeResize(Eigen::NoChange, num_vertices);

            return polygon_pixel_points;
        }

        [[nodiscard]] Eigen::MatrixXi
        RayCasting(const Eigen::Ref<const Eigen::VectorXd>& start, const Eigen::Ref<const Eigen::VectorXd>& end) const {
            if (!InMap(start)) {
                ERL_WARN("start point (%f, %f, %f) is out of map.", start[0], start[1], start[2]);
                return {};
            }
            if (!InMap(end)) {
                ERL_WARN("end point (%f, %f, %f) is out of map.", end[0], end[1], end[2]);
                return {};
            }
            Eigen::VectorXi cur_grid = MeterToGridForPoints(start);
            Eigen::VectorXi end_grid = MeterToGridForPoints(end);
            if (cur_grid == end_grid) { return Eigen::MatrixXi(cur_grid); }

            // initialize
            int dim = Dims();
            Eigen::VectorXd direction = end - start;
            double length = direction.norm();
            direction /= length;

            // compute step direction
            Eigen::VectorXi step(dim);
            for (int i = 0; i < dim; ++i) {
                if (direction[i] > 0) {
                    step[i] = 1;
                } else if (direction[i] < 0) {
                    step[i] = -1;
                } else {
                    step[i] = 0;
                }
            }
            if (step.isZero()) {
                ERL_WARN("Ray casting in direction (0, 0, 0) is impossible!");
                return {};
            }

            // compute t_max and t_delta
            Eigen::VectorXd t_max(dim);
            Eigen::VectorXd t_delta(dim);
            for (int i = 0; i < dim; ++i) {
                if (step[i] == 0) {
                    t_max[i] = std::numeric_limits<double>::infinity();
                    t_delta[i] = std::numeric_limits<double>::infinity();
                } else {
                    double voxel_border = GridToMeterForValue(cur_grid[i], i) + static_cast<double>(step[i]) * 0.5 * m_resolution_[i];
                    t_max[i] = (voxel_border - start[i]) / direction[i];
                    t_delta[i] = m_resolution_[i] / std::abs(direction[i]);
                }
            }

            // incremental phase
            Eigen::MatrixXi points(dim, (3 + dim) * (end_grid - cur_grid).cwiseAbs().maxCoeff());
            int cnt = 0;
            points.col(cnt++) = cur_grid;
            while (true) {
                int min_dim = 0;
                for (int i = 1; i < dim; ++i) {
                    if (t_max[i] < t_max[min_dim]) { min_dim = i; }
                }

                t_max[min_dim] += t_delta[min_dim];
                cur_grid[min_dim] += step[min_dim];
                ERL_DEBUG_ASSERT(
                    cur_grid[min_dim] >= 0 && cur_grid[min_dim] < m_map_shape_[min_dim],
                    "cur_grid[%d] = %d is out of range [0, %d)",
                    min_dim,
                    cur_grid[min_dim],
                    m_map_shape_[min_dim]);

                if (cur_grid == end_grid) {
                    points.col(cnt++) = cur_grid;
                    break;
                }

                // this seems to be unlikely to happen
                double dist_from_origin = t_max.minCoeff();
                if (dist_from_origin > length) { break; }  // this happens due to numerical error
                points.col(cnt++) = cur_grid;
                ERL_ASSERTM(cnt < points.cols() - 1, "Pre-allocated points are not enough.");
            }
            points.conservativeResize(Eigen::NoChange, cnt);
            return points;
        }
    };

    typedef GridMapInfo<2> GridMapInfo2D;
    typedef GridMapInfo<3> GridMapInfo3D;
    typedef GridMapInfo<Eigen::Dynamic> GridMapInfoXd;

}  // namespace erl::common
