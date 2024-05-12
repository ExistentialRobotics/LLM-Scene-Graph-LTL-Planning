#pragma once

#include "erl_common/eigen.hpp"
#include "erl_common/assert.hpp"

#include <nanoflann.hpp>
#include <memory>

namespace erl::geometry {

    template<typename T, int Dim, typename Metric = nanoflann::metric_L2_Simple, typename IndexType = long>
    class KdTreeEigenAdaptor {

        using EigenMatrix = Eigen::Matrix<T, Dim, Eigen::Dynamic>;
        using Self = KdTreeEigenAdaptor;
        using NumType = typename EigenMatrix::Scalar;
        using MetricType = typename Metric::template traits<NumType, Self>::distance_t;
        using TreeType = nanoflann::KDTreeSingleIndexAdaptor<MetricType, Self, Dim, IndexType>;

        std::shared_ptr<TreeType> m_tree_ = nullptr;
        EigenMatrix m_data_matrix_;
        const int mk_MLeafMaxSize_;

    public:
        explicit KdTreeEigenAdaptor(int leaf_max_size = 10)
            : mk_MLeafMaxSize_(leaf_max_size) {}

        explicit KdTreeEigenAdaptor(EigenMatrix mat, bool build = true, int leaf_max_size = 10)
            : m_data_matrix_(std::move(mat)),
              mk_MLeafMaxSize_(leaf_max_size) {

            if (build) { Build(); }
        }

        explicit KdTreeEigenAdaptor(const T *data, long num_points, bool build = true, int leaf_max_size = 10)
            : m_data_matrix_(Eigen::Map<const EigenMatrix>(data, Dim, num_points)),
              mk_MLeafMaxSize_(leaf_max_size) {

            if (build) { Build(); }
        }

        [[nodiscard]] const EigenMatrix &
        GetDataMatrix() const {
            return m_data_matrix_;
        }

        [[nodiscard]] Eigen::Vector<T, Dim>
        GetPoint(IndexType idx) const {
            return m_data_matrix_.col(idx);
        }

        void
        SetDataMatrix(EigenMatrix mat, bool build = true) {
            m_data_matrix_ = std::move(mat);
            m_tree_ = nullptr;  // invalidate the tree
            if (build) { Build(); }
        }

        void
        SetDataMatrix(const T *data, long num_points, bool build = true) {
            m_data_matrix_ = Eigen::Map<const EigenMatrix>(data, Dim, num_points);
            m_tree_ = nullptr;  // invalidate the tree
            if (build) { Build(); }
        }

        void
        Clear() {
            if (Dim == Eigen::Dynamic) {
                m_data_matrix_.resize(0, 0);
            } else {
                m_data_matrix_.resize(Dim, 0);
            }
            m_tree_ = nullptr;
        }

        [[nodiscard]] bool
        Ready() const {
            return m_tree_ != nullptr;
        }

        // Rebuild the KD tree from scratch
        void
        Build() {
            ERL_ASSERTM(m_data_matrix_.cols() > 0, "no data. cannot build tree.");
            m_tree_ = std::make_shared<TreeType>(Dim, *this, nanoflann::KDTreeSingleIndexAdaptorParams(mk_MLeafMaxSize_));
            m_tree_->buildIndex();
        }

        void
        Knn(size_t k, const Eigen::Vector<T, Dim> &point, IndexType &indices_out, NumType &metric_out) {
            nanoflann::KNNResultSet<NumType, IndexType> result_set(k);
            result_set.init(&indices_out, &metric_out);  // default metric is squared euclidean distance
            m_tree_->findNeighbors(result_set, point.data(), nanoflann::SearchParameters());
        }

        // Returns the number of points: used by TreeType
        [[nodiscard]] size_t
        kdtree_get_point_count() const {
            return m_data_matrix_.cols();
        }

        // Returns the dim-th component of the idx-th point in the class, used by TreeType
        [[nodiscard]] NumType
        kdtree_get_pt(const size_t idx, int dim) const {
            return m_data_matrix_(dim, idx);
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        template<class BBOX>
        bool  // ReSharper disable once CppMemberFunctionMayBeStatic
        kdtree_get_bbox(BBOX &) const {
            return false;
        }
    };

    typedef KdTreeEigenAdaptor<double, 3> KdTree3d;
    typedef KdTreeEigenAdaptor<double, 2> KdTree2d;
}  // namespace erl::geometry
