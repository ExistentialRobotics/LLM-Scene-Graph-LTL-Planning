#pragma once

#include "erl_common/yaml.hpp"
#include "erl_env/environment_ltl_scene_graph.hpp"
#include "erl_env/atomic_proposition.hpp"
#include "heuristic.hpp"

#include <memory>

namespace erl::search_planning {

    class LLMSceneGraphHeuristic : public HeuristicBase {

    public:
        struct Setting : common::Yamlable<Setting> {
            struct LLMWaypoint {
                env::AtomicProposition::Type type;
                int uuid1;
                int uuid2;
            };

            std::map<int, std::map<uint32_t, std::vector<LLMWaypoint>>> llm_paths;  // <room_id, <fsa_state, path>>
        };

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::shared_ptr<env::EnvironmentLTLSceneGraph> m_env_ = nullptr;
        std::unordered_map<int, std::vector<double>> m_heuristic_cache_ = {};

    public:
        LLMSceneGraphHeuristic(std::shared_ptr<Setting> setting, std::shared_ptr<env::EnvironmentLTLSceneGraph> env);

        [[nodiscard]] double
        operator()(const env::EnvironmentState &env_state) const override;
    };

}  // namespace erl::search_planning

namespace YAML {
    template<>
    struct convert<erl::search_planning::LLMSceneGraphHeuristic::Setting::LLMWaypoint> {
        static Node
        encode(const erl::search_planning::LLMSceneGraphHeuristic::Setting::LLMWaypoint &rhs) {
            Node node;
            switch (rhs.type) {
                case erl::env::AtomicProposition::Type::kEnterRoom: {
                    node = erl::common::AsString("move(", rhs.uuid1, ", ", rhs.uuid2, ")");
                    break;
                }
                case erl::env::AtomicProposition::Type::kReachObject: {
                    node = erl::common::AsString("reach(", rhs.uuid1, ", ", rhs.uuid2, ")");
                    break;
                }
                default:
                    throw std::runtime_error("Unknown LLMWaypoint type.");
            }
            return node;
        }

        static bool
        decode(const Node &node, erl::search_planning::LLMSceneGraphHeuristic::Setting::LLMWaypoint &rhs) {
            if (!node.IsScalar()) { return false; }
            auto str = node.as<std::string>();
            if (str.find("move(") != std::string::npos) {
                rhs.type = erl::env::AtomicProposition::Type::kEnterRoom;
                std::stringstream ss(str);
                std::string tmp;
                std::getline(ss, tmp, '(');
                std::getline(ss, tmp, ',');
                rhs.uuid1 = std::stoi(tmp);
                std::getline(ss, tmp, ')');
                rhs.uuid2 = std::stoi(tmp);
            } else if (str.find("reach(") != std::string::npos) {
                rhs.type = erl::env::AtomicProposition::Type::kReachObject;
                std::stringstream ss(str);
                std::string tmp;
                std::getline(ss, tmp, '(');
                std::getline(ss, tmp, ',');
                rhs.uuid1 = std::stoi(tmp);
                std::getline(ss, tmp, ')');
                rhs.uuid2 = std::stoi(tmp);
            } else {
                throw std::runtime_error("Unknown LLMWaypoint string: " + str);
            }
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::search_planning::LLMSceneGraphHeuristic::Setting::LLMWaypoint &rhs) {
        switch (rhs.type) {
            case erl::env::AtomicProposition::Type::kEnterRoom: {
                out << erl::common::AsString("move(", rhs.uuid1, ", ", rhs.uuid2, ")");
                break;
            }
            case erl::env::AtomicProposition::Type::kReachObject: {
                out << erl::common::AsString("reach(", rhs.uuid1, ", ", rhs.uuid2, ")");
                break;
            }
            default:
                throw std::runtime_error("Unknown LLMWaypoint type.");
        }
        return out;
    }

    template<>
    struct convert<erl::search_planning::LLMSceneGraphHeuristic::Setting> {
        static Node
        encode(const erl::search_planning::LLMSceneGraphHeuristic::Setting &rhs) {
            Node node;
            node = rhs.llm_paths;
            return node;
        }

        static bool
        decode(const Node &node, erl::search_planning::LLMSceneGraphHeuristic::Setting &rhs) {
            using namespace erl::search_planning;
            using LLMPath = std::vector<LLMSceneGraphHeuristic::Setting::LLMWaypoint>;
            if (!node.IsMap()) { return false; }
            rhs.llm_paths = node.as<std::map<int, std::map<uint32_t, LLMPath>>>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::search_planning::LLMSceneGraphHeuristic::Setting &rhs) {
        out << rhs.llm_paths;
        return out;
    }
}  // namespace YAML
