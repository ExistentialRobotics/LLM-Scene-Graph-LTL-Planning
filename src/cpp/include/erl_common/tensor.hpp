#pragma once

#include "erl_common/storage_order.hpp"

#include <numeric>
#include <unordered_set>

namespace erl::common {

    /**
     * Eigen provides only Tensor support of fixed-rank (number of dimensions) or fixed-size.
     * Tensor supports dynamic tensor shape like NumPy NDArray.
     *
     * @tparam T
     */
    template<typename T, int Rank, bool RowMajor = true>
    class Tensor {

    protected:
        Eigen::VectorX<T> m_data_;
        Eigen::Vector<int, Rank> m_shape_;

    public:
        Tensor() = default;

        explicit Tensor(Eigen::VectorXi shape)
            : m_shape_(std::move(shape)) {
            CheckShape();
            int total_size = Size();
            if (total_size > 0) { m_data_.resize(total_size); }
        }

        Tensor(Eigen::VectorXi shape, const T &fill_value)
            : m_shape_(std::move(shape)) {
            CheckShape();
            int total_size = Size();
            if (total_size > 0) { m_data_.setConstant(total_size, fill_value); }
        }

        Tensor(Eigen::VectorXi shape, Eigen::VectorX<T> data)
            : m_shape_(std::move(shape)) {
            CheckShape();
            int total_size = Size();
            ERL_ASSERTM(total_size == data.size(), "shape and data are not matched.");
            if (total_size > 0) { m_data_ = data; }
        }

        Tensor(Eigen::VectorXi shape, const std::function<T(void)> &data_init_func)
            : m_shape_(std::move(shape)) {
            CheckShape();
            int total_size = Size();
            if (total_size > 0) {
                m_data_.resize(total_size);
                for (int i = 0; i < total_size; ++i) { m_data_[i] = data_init_func(); }
            }
        }

        Eigen::VectorX<T> &
        Data() {
            return m_data_;
        }

        const Eigen::VectorX<T> &
        Data() const {
            return m_data_;
        }

        const T *
        GetDataPtr() const {
            return m_data_.data();
        }

        T *
        GetMutableDataPtr() {
            return m_data_.data();
        }

        [[nodiscard]] int
        Dims() const {
            return m_shape_.size();
        }

        [[nodiscard]] Eigen::VectorXi
        Shape() const {
            return m_shape_;
        }

        [[nodiscard]] int
        Size() const {
            if (Dims()) { return m_shape_.prod(); }
            return 0;
        }

        [[nodiscard]] static bool
        IsRowMajor() {
            return RowMajor;
        }

        void
        Fill(const T &value) {
            int total_size = Size();
            if (total_size > 0) { m_data_.setConstant(total_size, value); }
        }

        T &
        operator[](const Eigen::Ref<const Eigen::Vector<int, Rank>> &coords) {
            int index = CoordsToIndex<Rank>(m_shape_, coords, RowMajor);
            return m_data_[index];
        }

        const T &
        operator[](const Eigen::Ref<const Eigen::Vector<int, Rank>> &coords) const {
            int index = CoordsToIndex<Rank>(m_shape_, coords, RowMajor);
            return m_data_[index];
        }

        T &
        operator[](int index) {
            return m_data_[index];
        }

        const T &
        operator[](int index) const {
            return m_data_[index];
        }

        Tensor<T, Eigen::Dynamic>
        GetSlice(const std::vector<int> &dims_to_remove, const std::vector<int> &dim_indices_at_removed) const {
            ERL_ASSERTM(dims_to_remove.size() == dim_indices_at_removed.size(), "dims_to_remove and dim_indices_at_removed should be of the same size");
            ERL_ASSERTM(!dims_to_remove.empty(), "dims_to_remove should not be empty");
            ERL_ASSERTM(!dim_indices_at_removed.empty(), "dim_indices_at_removed should not be empty");
            ERL_ASSERTM(
                std::unordered_set<int>(dims_to_remove.begin(), dims_to_remove.end()).size() == dims_to_remove.size(),
                "there are duplicate dims in dims_to_remove");

            const int ndim = Dims();
            // Remove unwanted dimensions
            std::vector<int> dims_to_keep(ndim);
            std::iota(dims_to_keep.begin(), dims_to_keep.end(), 0);

            // descending order ensures that indices remain valid when removing
            std::vector<int> sorted_indices(dims_to_remove.size());
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
            std::sort(sorted_indices.begin(), sorted_indices.end(), std::greater());
            Eigen::VectorXi coords = Eigen::VectorXi::Zero(ndim);
            for (auto &i: sorted_indices) {
                ERL_ASSERTM(0 <= dims_to_remove[i] && dims_to_remove[i] < ndim, "%d-dim is out of range for %d-dim shape", dims_to_remove[i], ndim);
                dims_to_keep.erase(dims_to_keep.begin() + dims_to_remove[i]);
                coords[dims_to_remove[i]] = dim_indices_at_removed[i];
            }

            // generate new tensor
            Eigen::VectorXi slice_shape = m_shape_(dims_to_keep);
            Tensor<T, Eigen::Dynamic> slice(slice_shape);
            for (int i = 0; i < slice.Size(); ++i) {
                auto slice_coords = IndexToCoords<Eigen::Dynamic>(slice_shape, i, RowMajor);
                for (int j = 0; j < static_cast<int>(dims_to_keep.size()); ++j) { coords[dims_to_keep[j]] = slice_coords[j]; }
                slice[i] = (*this)[coords];
            }

            return slice;
        }

        void
        Print(std::ostream &os) const {
            os << "Tensor, shape: " << m_shape_.transpose() << ", data: array of " << typeid(T).name() << std::endl;
        }

    private:
        void
        CheckShape() {
            for (int i = 0; i < m_shape_.size(); ++i) { ERL_ASSERTM(m_shape_[i] >= 0, "negative size %d at %d-dim", m_shape_[i], i); }
        }
    };

    template<typename T, int Rank, bool RowMajor = true>
    std::ostream &
    operator<<(std::ostream &os, const Tensor<T, Rank, RowMajor> &tensor) {
        tensor.Print(os);
        return os;
    }

    template<typename T, bool RowMajor = true>
    using TensorX = Tensor<T, Eigen::Dynamic, RowMajor>;

    typedef Tensor<double, 2> TensorDouble2D;
    typedef Tensor<double, 3> TensorDouble3D;
    typedef TensorX<double> TensorDoubleXd;
    typedef Tensor<int, 2> TensorInt2D;
    typedef Tensor<int, 3> TensorInt3D;
    typedef TensorX<int> TensorIntXd;
    typedef Tensor<uint8_t, 2> TensorUnsigned2D;
    typedef Tensor<uint8_t, 3> TensorUnsigned3D;
    typedef TensorX<uint8_t> TensorUnsignedXd;

}  // namespace erl::common
