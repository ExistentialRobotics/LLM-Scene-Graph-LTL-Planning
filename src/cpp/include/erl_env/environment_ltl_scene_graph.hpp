#pragma once

#include "erl_common/yaml.hpp"
#include "erl_common/grid_map_info.hpp"
#include "environment_scene_graph.hpp"
#include "scene_graph.hpp"
#include "finite_state_automaton.hpp"
#include "atomic_proposition.hpp"

#include <absl/container/flat_hash_map.h>

namespace erl::env {

    class EnvironmentLTLSceneGraph final : public EnvironmentSceneGraph {
    public:
        struct Setting final : common::OverrideYamlable<EnvironmentSceneGraph::Setting, Setting> {
            std::unordered_map<std::string, std::shared_ptr<AtomicProposition>> atomic_propositions;
            std::shared_ptr<FiniteStateAutomaton::Setting> fsa;  // finite state automaton

            void
            LoadAtomicPropositions(const std::string &yaml_file) {
                const YAML::Node node = YAML::LoadFile(yaml_file);
                YAML::convert<std::unordered_map<std::string, std::shared_ptr<AtomicProposition>>>::decode(node, atomic_propositions);
            }
        };

    protected:
        std::shared_ptr<FiniteStateAutomaton> m_fsa_ = nullptr;
        std::unordered_map<int, Eigen::MatrixX<uint32_t>> m_label_maps_ = {};
        absl::flat_hash_map<int, Eigen::MatrixX<std::vector<int>>> m_up_stairs_path_q_maps_ = {};
        absl::flat_hash_map<int, Eigen::MatrixX<std::vector<int>>> m_down_stairs_path_q_maps_ = {};
        absl::flat_hash_map<int, Eigen::MatrixX<std::vector<int>>> m_object_path_q_maps_ = {};
        absl::flat_hash_map<int, absl::flat_hash_map<int, Eigen::MatrixX<std::vector<int>>>> m_room_path_q_maps_ = {};

        friend class search_planning::LLMSceneGraphHeuristic;

    public:
        EnvironmentLTLSceneGraph(std::shared_ptr<scene_graph::Building> building, const std::shared_ptr<Setting> &setting)
            : EnvironmentSceneGraph(std::move(building), setting),
              m_fsa_(std::make_shared<FiniteStateAutomaton>(setting->fsa)) {

            GenerateLabelMaps();

            // initialize path_q maps
            for (auto &[floor_id, cost_map]: m_up_stairs_cost_maps_) { m_up_stairs_path_q_maps_[floor_id].resize(cost_map.rows(), cost_map.cols()); }
            for (auto &[floor_id, cost_map]: m_down_stairs_cost_maps_) { m_down_stairs_path_q_maps_[floor_id].resize(cost_map.rows(), cost_map.cols()); }
            for (auto &[obj_id, cost_map]: m_object_cost_maps_) { m_object_path_q_maps_[obj_id].resize(cost_map.cost_map.rows(), cost_map.cost_map.cols()); }
            for (auto &[room1_id, cost_maps]: m_room_cost_maps_) {
                for (auto &[room2_id, cost_map]: cost_maps) {
                    m_room_path_q_maps_[room1_id][room2_id].resize(cost_map.cost_map.rows(), cost_map.cost_map.cols());
                }
            }
        }

        [[nodiscard]] std::shared_ptr<FiniteStateAutomaton>
        GetFiniteStateAutomaton() const {
            return m_fsa_;
        }

        [[nodiscard]] std::unordered_map<int, Eigen::MatrixX<uint32_t>>
        GetLabelMaps() const {
            return m_label_maps_;
        }

        [[nodiscard]] std::vector<std::shared_ptr<EnvironmentState>>
        ForwardAction(const std::shared_ptr<const EnvironmentState> &env_state, const std::vector<int> &action_coords) const override;

        [[nodiscard]] std::vector<Successor>
        GetSuccessorsAtLevel(const std::shared_ptr<EnvironmentState> &env_state, std::size_t resolution_level) const override;

        [[nodiscard]] bool
        InStateSpace(const std::shared_ptr<EnvironmentState> &env_state) const override {
            return m_grid_map_info_->InGrids(env_state->grid.head<3>());
        }

        [[nodiscard]] bool
        InStateSpaceAtLevel(const std::shared_ptr<EnvironmentState> &env_state, std::size_t resolution_level) const override {
            if (resolution_level == 0) { return m_grid_map_info_->InGrids(env_state->grid.head<3>()); }
            const auto level = static_cast<scene_graph::Node::Type>(resolution_level - 1);
            if (!m_grid_map_info_->InGrids(env_state->grid.head<3>())) { return false; }
            switch (level) {
                case scene_graph::Node::Type::kObject:
                    return !m_object_reached_maps_.at(env_state->grid[2])(env_state->grid[0], env_state->grid[1]).empty();
                case scene_graph::Node::Type::kRoom:
                    return m_room_maps_[env_state->grid[2]].at<int>(env_state->grid[0], env_state->grid[1]) > 0;
                case scene_graph::Node::Type::kOcc:
                case scene_graph::Node::Type::kFloor:
                case scene_graph::Node::Type::kBuilding:
                    return true;
                default:
                    throw std::runtime_error("Unknown level.");
            }
        }

        [[nodiscard]] uint32_t
        StateHashing(const std::shared_ptr<env::EnvironmentState> &env_state) const override {
            uint32_t hashing = m_grid_map_info_->GridToIndex(env_state->grid.head<3>(), true);
            const auto *setting = reinterpret_cast<Setting *>(m_setting_.get());
            hashing = hashing * setting->fsa->num_states + env_state->grid[3];
            return hashing;
        }

        [[nodiscard]] Eigen::VectorXi
        MetricToGrid(const Eigen::Ref<const Eigen::VectorXd> &metric_state) const override {
            return Eigen::Vector4i(
                m_grid_map_info_->MeterToGridForValue(metric_state[0], 0),
                m_grid_map_info_->MeterToGridForValue(metric_state[1], 1),
                m_grid_map_info_->MeterToGridForValue(metric_state[2], 2),
                static_cast<int>(metric_state[3]));
        }

        [[nodiscard]] Eigen::VectorXd
        GridToMetric(const Eigen::Ref<const Eigen::VectorXi> &grid_state) const override {
            return Eigen::Vector4d(
                m_grid_map_info_->GridToMeterForValue(grid_state[0], 0),
                m_grid_map_info_->GridToMeterForValue(grid_state[1], 1),
                m_grid_map_info_->GridToMeterForValue(grid_state[2], 2),
                grid_state[3]);
        }

    protected:
        void
        GenerateLabelMaps();

        bool
        EvaluateAtomicProposition(const int x, const int y, const int floor_num, const std::shared_ptr<AtomicProposition> &proposition) {
            switch (proposition->type) {
                case AtomicProposition::Type::kNA:
                    return false;
                case AtomicProposition::Type::kEnterRoom:
                    return EvaluateEnterRoom(x, y, floor_num, proposition->uuid);
                case AtomicProposition::Type::kReachObject: {
                    double reach_distance = proposition->reach_distance > 0 ? proposition->reach_distance : m_setting_->object_reach_distance;
                    return EvaluateReachObject(x, y, floor_num, proposition->uuid, reach_distance);
                }
                default:
                    throw std::runtime_error("Unknown atomic proposition type.");
            }
        }

        bool
        EvaluateEnterRoom(const int x, const int y, const int floor_num, const int uuid) {
            const auto room = m_scene_graph_->GetNode<scene_graph::Room>(uuid);
            return m_room_maps_[floor_num].at<int>(x, y) == room->id;
        }

        bool
        EvaluateReachObject(int x, int y, int floor_num, int uuid, double reach_distance);

    private:
        std::vector<std::shared_ptr<EnvironmentState>>
        ConvertPath(const std::vector<std::array<int, 2>> &path, int floor_num, int cur_q) const;

        std::vector<std::shared_ptr<EnvironmentState>>
        GetPathToFloor(int xg, int yg, int floor_num, int cur_q, int next_floor_num) const;
    };
}  // namespace erl::env

namespace YAML {
    template<>
    struct convert<erl::env::EnvironmentLTLSceneGraph::Setting> {
        static Node
        encode(const erl::env::EnvironmentLTLSceneGraph::Setting &rhs) {
            Node node = convert<erl::env::EnvironmentSceneGraph::Setting>::encode(rhs);
            node["atomic_propositions"] = rhs.atomic_propositions;
            node["fsa"] = rhs.fsa;
            return node;
        }

        static bool
        decode(const Node &node, erl::env::EnvironmentLTLSceneGraph::Setting &rhs) {
            if (!convert<erl::env::EnvironmentSceneGraph::Setting>::decode(node, rhs)) { return false; }
            rhs.atomic_propositions = node["atomic_propositions"].as<std::unordered_map<std::string, std::shared_ptr<erl::env::AtomicProposition>>>();
            rhs.fsa = node["fsa"].as<std::shared_ptr<erl::env::FiniteStateAutomaton::Setting>>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::env::EnvironmentLTLSceneGraph::Setting &rhs) {
        out << static_cast<const erl::env::EnvironmentSceneGraph::Setting &>(rhs);
        out << BeginMap;
        out << Key << "atomic_propositions" << Value << rhs.atomic_propositions;
        out << Key << "fsa" << Value << rhs.fsa;
        out << EndMap;
        return out;
    }
}  // namespace YAML
