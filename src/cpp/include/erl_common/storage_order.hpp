#pragma once

#include "assert.hpp"
#include "eigen.hpp"

#include <vector>

namespace erl::common {

    template<typename T>
    std::vector<T>
    ComputeCStrides(const std::vector<T> &shape, T item_size) {
        auto ndim = T(shape.size());
        std::vector<T> strides(ndim, item_size);
        for (T i = ndim - 1; i > 0; --i) { strides[i - 1] = strides[i] * shape[i]; }
        return strides;
    }

    template<typename T>
    Eigen::VectorX<T>
    ComputeCStrides(const Eigen::Ref<const Eigen::VectorX<T>> &shape, T item_size) {
        auto ndim = T(shape.size());
        Eigen::VectorX<T> strides = Eigen::VectorX<T>::Constant(ndim, item_size);
        for (T i = ndim - 1; i > 0; --i) { strides[i - 1] = strides[i] * shape[i]; }
        return strides;
    }

    template<typename T>
    std::vector<T>
    ComputeFStrides(const std::vector<T> &shape, T item_size) {
        auto ndim = T(shape.size());
        std::vector<T> strides(ndim, item_size);
        for (T i = 1; i < ndim; ++i) { strides[i] = strides[i - 1] * shape[i - 1]; }
        return strides;
    }

    template<typename T>
    Eigen::VectorX<T>
    ComputeFStrides(const Eigen::Ref<const Eigen::VectorX<T>> &shape, T item_size) {
        auto ndim = T(shape.size());
        Eigen::VectorX<T> strides = Eigen::VectorX<T>::Constant(ndim, item_size);
        for (T i = 1; i < ndim; ++i) { strides[i] = strides[i - 1] * shape[i - 1]; }
        return strides;
    }

    template<int Dim>
    [[nodiscard]] int
    CoordsToIndex(const Eigen::Ref<const Eigen::Vector<int, Dim>> &shape, const Eigen::Ref<const Eigen::Vector<int, Dim>> &coords, const bool c_stride) {
        const auto ndim = static_cast<int>(shape.size());

        if (Dim == 2) {
            if (c_stride) { return coords[1] + coords[0] * shape[1]; }
            return coords[0] + coords[1] * shape[0];
        }

        if (Dim == 3) {
            if (c_stride) { return coords[2] + shape[2] * (coords[1] + shape[1] * coords[0]); }
            return coords[0] + shape[0] * (coords[1] + shape[1] * coords[2]);
        }

        if (Dim == 4) {
            if (c_stride) { return coords[3] + shape[3] * (coords[2] + shape[2] * (coords[1] + shape[1] * coords[0])); }
            return coords[0] + shape[0] * (coords[1] + shape[1] * (coords[2] + shape[2] * coords[3]));
        }

        if (c_stride) {
            int index = coords[0];
            for (int i = 1; i < ndim; ++i) { index = index * shape[i] + coords[i]; }

            // int index = coords[ndim - 1];
            // int prod = 1;
            // for (int i = ndim - 1; i > 0; --i) {
            //     prod *= shape[i];
            //     index += coords[i - 1] * prod;
            // }

            return index;
        }

        int index = coords[ndim - 1];
        for (int i = ndim - 1; i > 0; --i) { index = index * shape[i - 1] + coords[i - 1]; }

        // int index = coords[0];
        // int prod = 1;
        // for (int i = 1; i < ndim - 1; ++i) {
        //     prod *= shape[i - 1];
        //     index += coords[i] * prod;
        // }

        return index;
    }

    template<int Dim>
    [[nodiscard]] int
    CoordsToIndex(const Eigen::Ref<const Eigen::Vector<int, Dim>> &strides, const Eigen::Ref<const Eigen::Vector<int, Dim>> &coords) {
        const auto ndim = static_cast<int>(strides.size());
        for (int i = 0; i < ndim; ++i) { ERL_DEBUG_ASSERT(coords[i] >= 0, "%d-dim of coords is not positive: %d", i, coords[i]); }
        return strides.dot(coords);
    }

    template<int Dim>
    [[nodiscard]] Eigen::Vector<int, Dim>
    IndexToCoords(const Eigen::Ref<const Eigen::Vector<int, Dim>> &shape, int index, const bool c_stride) {
        // for (int i = 0; i < shape.size(); ++i) { ERL_DEBUG_ASSERT(shape[i] >= 0,
        // "negative size %d at %d-dim", shape[i], i); } int total_size =
        // shape.prod(); ERL_DEBUG_ASSERT(index >= -total_size && index < total_size,
        // "%s", AsString("index ", index, "is out of range of shape ",
        // shape.transpose()).c_str()); if (index < 0) { index += total_size; }

        auto ndim = static_cast<int>(shape.size());
        Eigen::Vector<int, Dim> coords;
        coords.setZero(ndim);

        if (Dim == 2) {
            if (c_stride) {
                coords[0] = index / shape[1];
                coords[1] = index - coords[0] * shape[1];
            } else {
                coords[1] = index / shape[0];
                coords[0] = index - coords[1] * shape[0];
            }
            return coords;
        }

        if (Dim == 3) {
            if (c_stride) {  // coords[2] + shape[2] * (coords[1] + shape[1] * coords[0])
                int prod = shape[1] * shape[2];
                coords[0] = index / prod;
                index -= coords[0] * prod;
                coords[1] = index / shape[2];
                coords[2] = index - coords[1] * shape[2];
            } else {  // coords[0] + shape[0] * (coords[1] + shape[1] * coords[2])
                int prod = shape[0] * shape[1];
                coords[2] = index / prod;
                index -= coords[2] * prod;
                coords[1] = index / shape[0];
                coords[0] = index - coords[1] * shape[0];
            }
            return coords;
        }

        if (Dim == 4) {
            if (c_stride) {  // coords[3] + shape[3] * (coords[2] + shape[2] * (coords[1]
                             // + shape[1] * coords[0]))
                int prod_23 = shape[2] * shape[3];
                int prod_123 = shape[1] * prod_23;
                coords[0] = index / prod_123;
                index -= coords[0] * prod_123;
                coords[1] = index / prod_23;
                index -= coords[1] * prod_23;
                coords[2] = index / shape[3];
                coords[3] = index - coords[2] * shape[3];
            } else {  // coords[0] + shape[0] * (coords[1] + shape[1] * (coords[2] +
                      // shape[2] * coords[3]))
                int prod_12 = shape[1] * shape[2];
                int prod_012 = shape[0] * prod_12;
                coords[3] = index / prod_012;
                index -= coords[3] * prod_012;
                coords[2] = index / prod_12;
                index -= coords[2] * prod_12;
                coords[1] = index / shape[2];
                coords[0] = index - coords[1] * shape[2];
            }
            return coords;
        }

        if (c_stride) {
            for (int i = ndim - 1; i >= 0; --i) {
                coords[i] = index % shape[i];
                index -= coords[i];
                index /= shape[i];
            }
            return coords;
        }

        for (int i = 0; i < ndim; ++i) {
            coords[i] = index % shape[i];
            index -= coords[i];
            index /= shape[i];
        }
        return coords;
    }
}  // namespace erl::common
