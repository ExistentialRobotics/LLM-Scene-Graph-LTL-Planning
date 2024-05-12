#pragma once

#include "erl_common/yaml.hpp"

namespace erl::env {

    struct AtomicProposition final : common::Yamlable<AtomicProposition> {
        enum class Type {
            kNA = 0,
            kEnterRoom = 1,
            kReachObject = 2,
        };

        Type type = Type::kNA;
        int uuid = -1;
        double reach_distance = -1.0;

        AtomicProposition() = default;

        AtomicProposition(Type type, int uuid, double reach_distance)
            : type(type),
              uuid(uuid),
              reach_distance(reach_distance) {}
    };
}  // namespace erl::env

namespace YAML {

    template<>
    struct convert<erl::env::AtomicProposition::Type> {
        static Node
        encode(const erl::env::AtomicProposition::Type &rhs) {
            Node node;
            switch (rhs) {
                case erl::env::AtomicProposition::Type::kNA:
                    node = "kNA";
                    break;
                case erl::env::AtomicProposition::Type::kEnterRoom:
                    node = "kEnterRoom";
                    break;
                case erl::env::AtomicProposition::Type::kReachObject:
                    node = "kReachObject";
                    break;
            }
            return node;
        }

        static bool
        decode(const Node &node, erl::env::AtomicProposition::Type &rhs) {
            if (!node.IsScalar()) { return false; }
            auto value = node.as<std::string>();
            if (value == "kNA") {
                rhs = erl::env::AtomicProposition::Type::kNA;
                return true;
            }
            if (value == "kEnterRoom") {
                rhs = erl::env::AtomicProposition::Type::kEnterRoom;
                return true;
            }
            if (value == "kReachObject") {
                rhs = erl::env::AtomicProposition::Type::kReachObject;
                return true;
            }
            return false;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::env::AtomicProposition::Type &rhs) {
        switch (rhs) {
            case erl::env::AtomicProposition::Type::kNA:
                out << "kNA";
                break;
            case erl::env::AtomicProposition::Type::kEnterRoom:
                out << "kEnterRoom";
                break;
            case erl::env::AtomicProposition::Type::kReachObject:
                out << "kReachObject";
                break;
            default:
                throw std::runtime_error("Unknown AtomicProposition::Type");
        }
        return out;
    }

    template<>
    struct convert<erl::env::AtomicProposition> {
        static Node
        encode(const erl::env::AtomicProposition &rhs) {
            Node node;
            node["type"] = rhs.type;
            node["uuid"] = rhs.uuid;
            if (rhs.type == erl::env::AtomicProposition::Type::kReachObject) { node["reach_distance"] = rhs.reach_distance; }
            return node;
        }

        static bool
        decode(const Node &node, erl::env::AtomicProposition &rhs) {
            if (!node.IsMap()) { return false; }
            rhs.type = node["type"].as<erl::env::AtomicProposition::Type>();
            rhs.uuid = node["uuid"].as<int>();
            if (rhs.type == erl::env::AtomicProposition::Type::kReachObject) { rhs.reach_distance = node["reach_distance"].as<double>(); }
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::env::AtomicProposition &rhs) {
        out << BeginMap;
        out << Key << "type" << Value << rhs.type;
        out << Key << "uuid" << Value << rhs.uuid;
        if (rhs.type == erl::env::AtomicProposition::Type::kReachObject) { out << Key << "reach_distance" << Value << rhs.reach_distance; }
        out << EndMap;
        return out;
    }
}  // namespace YAML
