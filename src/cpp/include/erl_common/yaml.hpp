#pragma once

#include "assert.hpp"
#include "eigen.hpp"
#include "opencv.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <optional>
#include <memory>
#include <filesystem>

// https://yaml.org/spec/1.2.2/
// https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started

namespace erl::common {

    struct YamlableBase {
        virtual ~YamlableBase() = default;

        virtual void
        FromYamlNode(const YAML::Node& node) = 0;

        void
        FromYamlString(const std::string& yaml_string) {
            auto node = YAML::Load(yaml_string);
            FromYamlNode(node);
        }

        [[nodiscard]] virtual YAML::Node
        AsYamlNode() const = 0;

        [[nodiscard]] virtual std::string
        AsYamlString() const = 0;

        void
        FromYamlFile(const std::string& yaml_file) {
            auto node = YAML::LoadFile(yaml_file);
            FromYamlNode(node);
        }

        void
        AsYamlFile(const std::string& yaml_file) const {
            auto node = AsYamlNode();
            std::ofstream ofs(yaml_file);
            if (!ofs.is_open()) { throw std::runtime_error("Failed to open file: " + yaml_file); }
            YAML::Emitter emitter(ofs);
            emitter << node;
        }
    };

    template<typename T>
    struct Yamlable : YamlableBase {

        void
        FromYamlNode(const YAML::Node& node) override {
            YAML::convert<T>::decode(node, *static_cast<T*>(this));
        }

        [[nodiscard]] YAML::Node
        AsYamlNode() const override {
            return YAML::convert<T>::encode(*static_cast<const T*>(this));
        }

        [[nodiscard]] std::string
        AsYamlString() const override {
            YAML::Emitter emitter;
            // this triggers YAML::Emitter& operator<<(YAML::Emitter& out, const T& rhs)
            emitter << *static_cast<const T*>(this);  // allow YAML style manipulation, e.g., emitter.SetIndent(4);
            return emitter.c_str();
        }
    };

    /**
     * @brief Override the Yamlable interface when the class is derived from a Yamlable class.
     * @tparam Base The base class that is derived from Yamlable<Base>.
     * @tparam T The derived class that is derived from Base.
     */
    template<typename Base, typename T>
    struct OverrideYamlable : Base {
        static_assert(std::is_base_of_v<Yamlable<Base>, Base>, "Base must be derived from Yamlable<Base>.");

        void
        FromYamlNode(const YAML::Node& node) override {
            YAML::convert<T>::decode(node, *static_cast<T*>(this));
        }

        [[nodiscard]] YAML::Node
        AsYamlNode() const override {
            return YAML::convert<T>::encode(*static_cast<const T*>(this));
        }

        [[nodiscard]] std::string
        AsYamlString() const override {
            YAML::Emitter emitter;
            // this triggers YAML::Emitter& operator<<(YAML::Emitter& out, const T& rhs)
            emitter << *static_cast<const T*>(this);  // allow YAML style manipulation, e.g., emitter.SetIndent(4);
            return emitter.c_str();
        }
    };
}  // namespace erl::common

namespace YAML {

    template<typename T, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic, int Order = Eigen::ColMajor>
    struct ConvertEigenMatrix {
        static Node
        encode(const Eigen::Matrix<T, Rows, Cols, Order>& rhs) {
            Node node(NodeType::Sequence);
            int rows = Rows == Eigen::Dynamic ? rhs.rows() : Rows;
            int cols = Cols == Eigen::Dynamic ? rhs.cols() : Cols;

            if (Order == Eigen::RowMajor) {
                for (int i = 0; i < rows; ++i) {
                    Node row_node(NodeType::Sequence);
                    for (int j = 0; j < cols; ++j) { row_node.push_back(rhs(i, j)); }
                    node.push_back(row_node);
                }
            } else {
                for (int j = 0; j < cols; ++j) {
                    Node col_node(NodeType::Sequence);
                    for (int i = 0; i < rows; ++i) { col_node.push_back(rhs(i, j)); }
                    node.push_back(col_node);
                }
            }

            return node;
        }

        static bool
        decode(const Node& node, Eigen::Matrix<T, Rows, Cols, Order>& rhs) {
            if (node.IsNull() && (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic)) { return true; }
            if (!node.IsSequence()) { return false; }
            if (!node[0].IsSequence()) { return false; }

            if (Order == Eigen::RowMajor) {
                int rows = Rows == Eigen::Dynamic ? node.size() : Rows;
                int cols = Cols == Eigen::Dynamic ? node[0].size() : Cols;
                rhs.resize(rows, cols);
                ERL_DEBUG_ASSERT(rows == int(node.size()), "expecting rows: %d, get node.size(): %lu", rows, node.size());
                for (int i = 0; i < rows; ++i) {
                    ERL_DEBUG_ASSERT(cols == int(node[i].size()), "expecting cols: %d, get node[0].size(): %lu", cols, node[i].size());
                    auto& kRowNode = node[i];
                    for (int j = 0; j < cols; ++j) { rhs(i, j) = kRowNode[j].as<T>(); }
                }
            } else {
                int cols = Cols == Eigen::Dynamic ? node.size() : Cols;
                int rows = Rows == Eigen::Dynamic ? node[0].size() : Rows;
                rhs.resize(rows, cols);
                ERL_DEBUG_ASSERT(cols == int(node.size()), "expecting cols: %d, get node.size(): %lu", cols, node.size());
                for (int j = 0; j < cols; ++j) {
                    ERL_DEBUG_ASSERT(rows == int(node[j].size()), "expecting rows: %d, get node[0].size(): %lu", rows, node[j].size());
                    auto& kColNode = node[j];
                    for (int i = 0; i < rows; ++i) { rhs(i, j) = kColNode[i].as<T>(); }
                }
            }

            return true;
        }
    };

    template<typename T, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic, int Order = Eigen::ColMajor>
    Emitter&
    operator<<(Emitter& out, const Eigen::Matrix<T, Rows, Cols, Order>& rhs) {
        out << BeginSeq;
        int rows = Rows == Eigen::Dynamic ? rhs.rows() : Rows;
        int cols = Cols == Eigen::Dynamic ? rhs.cols() : Cols;
        if (Order == Eigen::RowMajor) {
            for (int i = 0; i < rows; ++i) {
                out << BeginSeq;
                for (int j = 0; j < cols; ++j) { out << rhs(i, j); }
                out << EndSeq;
            }
        } else {
            for (int j = 0; j < cols; ++j) {
                out << BeginSeq;
                for (int i = 0; i < rows; ++i) { out << rhs(i, j); }
                out << EndSeq;
            }
        }
        out << EndSeq;
        return out;
    }

    template<>
    struct convert<Eigen::Matrix2i> : ConvertEigenMatrix<int, 2, 2> {};

    template<>
    struct convert<Eigen::Matrix2f> : ConvertEigenMatrix<float, 2, 2> {};

    template<>
    struct convert<Eigen::Matrix2d> : ConvertEigenMatrix<double, 2, 2> {};

    template<>
    struct convert<Eigen::Matrix2Xi> : ConvertEigenMatrix<int, 2> {};

    template<>
    struct convert<Eigen::Matrix2Xf> : ConvertEigenMatrix<float, 2> {};

    template<>
    struct convert<Eigen::Matrix2Xd> : ConvertEigenMatrix<double, 2> {};

    template<>
    struct convert<Eigen::MatrixX2i> : ConvertEigenMatrix<int, Eigen::Dynamic, 2> {};

    template<>
    struct convert<Eigen::MatrixX2f> : ConvertEigenMatrix<float, Eigen::Dynamic, 2> {};

    template<>
    struct convert<Eigen::MatrixX2d> : ConvertEigenMatrix<double, Eigen::Dynamic, 2> {};

    template<>
    struct convert<Eigen::Matrix3i> : ConvertEigenMatrix<int, 3, 3> {};

    template<>
    struct convert<Eigen::Matrix3f> : ConvertEigenMatrix<float, 3, 3> {};

    template<>
    struct convert<Eigen::Matrix3d> : ConvertEigenMatrix<double, 3, 3> {};

    template<>
    struct convert<Eigen::Matrix3Xi> : ConvertEigenMatrix<int, 3> {};

    template<>
    struct convert<Eigen::Matrix3Xf> : ConvertEigenMatrix<float, 3> {};

    template<>
    struct convert<Eigen::Matrix3Xd> : ConvertEigenMatrix<double, 3> {};

    template<>
    struct convert<Eigen::MatrixX3i> : ConvertEigenMatrix<int, Eigen::Dynamic, 3> {};

    template<>
    struct convert<Eigen::MatrixX3f> : ConvertEigenMatrix<float, Eigen::Dynamic, 3> {};

    template<>
    struct convert<Eigen::MatrixX3d> : ConvertEigenMatrix<double, Eigen::Dynamic, 3> {};

    template<>
    struct convert<Eigen::Matrix4i> : ConvertEigenMatrix<int, 4, 4> {};

    template<>
    struct convert<Eigen::Matrix4f> : ConvertEigenMatrix<float, 4, 4> {};

    template<>
    struct convert<Eigen::Matrix4d> : ConvertEigenMatrix<double, 4, 4> {};

    template<>
    struct convert<Eigen::Matrix4Xi> : ConvertEigenMatrix<int, 4> {};

    template<>
    struct convert<Eigen::Matrix4Xf> : ConvertEigenMatrix<float, 4> {};

    template<>
    struct convert<Eigen::Matrix4Xd> : ConvertEigenMatrix<double, 4> {};

    template<>
    struct convert<Eigen::MatrixX4i> : ConvertEigenMatrix<int, Eigen::Dynamic, 4> {};

    template<>
    struct convert<Eigen::MatrixX4f> : ConvertEigenMatrix<float, Eigen::Dynamic, 4> {};

    template<>
    struct convert<Eigen::MatrixX4d> : ConvertEigenMatrix<double, Eigen::Dynamic, 4> {};

    template<typename T, int Size = Eigen::Dynamic>
    struct ConvertEigenVector {
        static Node
        encode(const Eigen::Vector<T, Size>& rhs) {
            Node node(NodeType::Sequence);
            if (Size == Eigen::Dynamic) {
                for (int i = 0; i < rhs.size(); ++i) { node.push_back(rhs[i]); }
            } else {
                for (int i = 0; i < Size; ++i) { node.push_back(rhs[i]); }
            }
            return node;
        }

        static bool
        decode(const Node& node, Eigen::Vector<T, Size>& rhs) {
            if (!node.IsSequence()) { return false; }
            if (Size == Eigen::Dynamic) {
                rhs.resize(node.size());
                for (int i = 0; i < rhs.size(); ++i) { rhs[i] = node[i].as<T>(); }
            } else {
                for (int i = 0; i < Size; ++i) { rhs[i] = node[i].as<T>(); }
            }
            return true;
        }
    };

    // This is a faster way to dump an Eigen vector as YAML format to a stream.
    // An alternative is to use YAML::convert<Eigen::Vector<T, Size>>::encode(Eigen::Vector<T, GetSize>) -> YAML::Node and YAML::Dump(YAML::Node).
    template<typename T, int Size>
    Emitter&
    operator<<(Emitter& out, const Eigen::Vector<T, Size>& rhs) {
        out << BeginSeq << Flow;
        if (Size == Eigen::Dynamic) {
            for (int i = 0; i < rhs.size(); ++i) { out << rhs[i]; }
        } else {
            for (int i = 0; i < Size; ++i) { out << rhs[i]; }
        }
        out << EndSeq;
        return out;
    }

    template<>
    struct convert<Eigen::VectorXd> : ConvertEigenVector<double> {};

    template<>
    struct convert<Eigen::Vector2d> : ConvertEigenVector<double, 2> {};

    template<>
    struct convert<Eigen::Vector3d> : ConvertEigenVector<double, 3> {};

    template<>
    struct convert<Eigen::Vector4d> : ConvertEigenVector<double, 4> {};

    template<>
    struct convert<Eigen::VectorXf> : ConvertEigenVector<float> {};

    template<>
    struct convert<Eigen::Vector2f> : ConvertEigenVector<float, 2> {};

    template<>
    struct convert<Eigen::Vector3f> : ConvertEigenVector<float, 3> {};

    template<>
    struct convert<Eigen::Vector4f> : ConvertEigenVector<float, 4> {};

    template<>
    struct convert<Eigen::VectorXi> : ConvertEigenVector<int> {};

    template<>
    struct convert<Eigen::Vector2i> : ConvertEigenVector<int, 2> {};

    template<>
    struct convert<Eigen::Vector3i> : ConvertEigenVector<int, 3> {};

    template<>
    struct convert<Eigen::Vector4i> : ConvertEigenVector<int, 4> {};

    template<typename T>
    struct convert<std::optional<T>> {
        static Node
        encode(const std::optional<T>& rhs) {
            if (rhs) { return convert<T>::encode(*rhs); }
            return Node(NodeType::Null);
        }

        static bool
        decode(const Node& node, std::optional<T>& rhs) {
            if (node.Type() != NodeType::Null) {
                T value;
                if (convert<T>::decode(node, value)) {
                    rhs = value;
                    return true;
                }
                return false;
            }
            rhs = std::nullopt;
            return true;
        }
    };

    template<typename T>
    Emitter&
    operator<<(Emitter& out, const std::optional<T>& rhs) {
        if (rhs) {
            out << *rhs;
        } else {
            out << Null;
        }
        return out;
    }

    template<typename T>
    struct convert<std::shared_ptr<T>> {
        static Node
        encode(const std::shared_ptr<T>& rhs) {
            if (rhs == nullptr) { return Node(NodeType::Null); }
            return convert<T>::encode(*rhs);
        }

        static bool
        decode(const Node& node, std::shared_ptr<T>& rhs) {
            ERL_DEBUG_ASSERT(rhs == nullptr, "rhs should not be nullptr.");
            auto value = std::make_shared<T>();
            if (convert<T>::decode(node, *value)) {
                rhs = value;
                return true;
            }
            return false;
        }
    };

    template<typename T>
    Emitter&
    operator<<(Emitter& out, const std::shared_ptr<T>& rhs) {
        if (rhs == nullptr) {
            out << Null;
        } else {
            out << *rhs;
        }
        return out;
    }

    template<typename T>
    struct convert<std::unique_ptr<T>> {
        static Node
        encode(const std::unique_ptr<T>& rhs) {
            if (rhs == nullptr) { return Node(NodeType::Null); }
            return convert<T>::encode(*rhs);
        }

        static bool
        decode(const Node& node, std::unique_ptr<T>& rhs) {
            ERL_DEBUG_ASSERT(rhs == nullptr, "rhs should not be nullptr.");
            auto value = std::make_unique<T>();
            if (convert<T>::decode(node, *value)) {
                rhs = std::move(value);
                return true;
            }
            return false;
        }
    };

    template<typename T>
    Emitter&
    operator<<(Emitter& out, const std::unique_ptr<T>& rhs) {
        if (rhs == nullptr) {
            out << Null;
        } else {
            out << *rhs;
        }
        return out;
    }

    template<typename KeyType, typename ValueType>
    struct convert<std::unordered_map<KeyType, ValueType>> {
        static Node
        encode(const std::unordered_map<KeyType, ValueType>& rhs) {
            Node node(NodeType::Map);
            for (const auto& [key, value]: rhs) { node[convert<KeyType>::encode(key)] = convert<ValueType>::encode(value); }
            return node;
        }

        static bool
        decode(const Node& node, std::unordered_map<KeyType, ValueType>& rhs) {
            if (!node.IsMap()) { return false; }
            for (auto it = node.begin(); it != node.end(); ++it) {
                KeyType key;
                ValueType value;
                if (convert<KeyType>::decode(it->first, key) && convert<ValueType>::decode(it->second, value)) {
                    rhs[key] = value;
                } else {
                    return false;
                }
            }
            return true;
        }
    };

    template<typename KeyType, typename ValueType>
    Emitter&
    operator<<(Emitter& out, const std::unordered_map<KeyType, ValueType>& rhs) {
        out << BeginMap;
        for (const auto& [key, value]: rhs) { out << Key << key << Value << value; }
        out << EndMap;
        return out;
    }

    template<typename T, std::size_t N>
    Emitter&
    operator<<(Emitter& out, const std::array<T, N>& rhs) {
        out << BeginSeq;
        for (const auto& value: rhs) { out << value; }
        out << EndSeq;
        return out;
    }

    template<>
    struct convert<cv::Scalar> {
        static Node
        encode(const cv::Scalar& rhs) {
            Node node(NodeType::Sequence);
            node.push_back(rhs[0]);
            node.push_back(rhs[1]);
            node.push_back(rhs[2]);
            node.push_back(rhs[3]);
            return node;
        }

        static bool
        decode(const Node& node, cv::Scalar& rhs) {
            if (!node.IsSequence()) { return false; }
            rhs[0] = node[0].as<double>();
            rhs[1] = node[1].as<double>();
            rhs[2] = node[2].as<double>();
            rhs[3] = node[3].as<double>();
            return true;
        }
    };

    inline Emitter&
    operator<<(Emitter& out, const cv::Scalar& rhs) {
        out << BeginSeq << rhs[0] << rhs[1] << rhs[2] << rhs[3] << EndSeq;
        return out;
    }
}  // namespace YAML

inline std::ostream&
operator<<(std::ostream& out, const erl::common::YamlableBase& yaml) {
    out << yaml.AsYamlString();
    return out;
}
