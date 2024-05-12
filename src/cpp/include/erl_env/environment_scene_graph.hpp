#pragma once

#include "erl_common/yaml.hpp"
#include "erl_common/grid_map_info.hpp"
#include "environment_multi_resolution.hpp"
#include "scene_graph.hpp"

#include <absl/container/flat_hash_map.h>

namespace erl::search_planning {
    class LLMSceneGraphHeuristic;  // forward declaration
}

namespace erl::env {

    /**
     * Scene graph environment. The scene graph is a representation of
     */
    class EnvironmentSceneGraph : public EnvironmentMultiResolution {

    public:
        struct Setting : common::Yamlable<Setting> {
            std::string data_dir = {};                                            // folder to store scene graph data, actions, cost maps and so on
            long num_threads = 64;                                                // number of threads to use
            bool allow_diagonal = true;                                           // whether allow diagonal movement
            double object_reach_distance = 0.6;                                   // distance (meter) to reach an object
            Eigen::Matrix2Xd shape = {};                                          // shape of the robot, assume the shape center is at the origin
            scene_graph::Node::Type max_level = scene_graph::Node::Type::kFloor;  // maximum resolution level (inclusive)
        };

        struct AtomicAction {
            double cost = 0.0;
            Eigen::Vector3i state_diff = Eigen::Vector3i::Zero();

            AtomicAction(double cost_in, Eigen::Vector3i state_diff_in)
                : cost(cost_in),
                  state_diff(std::move(state_diff_in)) {}
        };

    protected:
        using PathMatrix = Eigen::MatrixX<std::vector<std::array<int, 2>>>;

        struct LocalCostMap {
            int grid_min_x = 0;             // global x coordinate of the local cost map origin
            int grid_min_y = 0;             // global y coordinate of the local cost map origin
            int grid_max_x = 0;             // width of the local cost map
            int grid_max_y = 0;             // height of the local cost map
            Eigen::MatrixXd cost_map = {};  // local cost map
            PathMatrix path_map = {};       // path map
        };

        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::shared_ptr<scene_graph::Building> m_scene_graph_ = nullptr;
        std::shared_ptr<common::GridMapInfo3D> m_grid_map_info_ = nullptr;                              // (x, y, floor_num), for hashing
        std::vector<cv::Mat> m_room_maps_ = {};                                                         // room maps for each floor
        std::vector<cv::Mat> m_cat_maps_ = {};                                                          // category maps for each floor
        std::vector<cv::Mat> m_ground_masks_ = {};                                                      // ground masks for each floor, 0: is ground
        std::vector<cv::Mat> m_obstacle_maps_ = {};                                                     // obstacle space maps, 0: free, >=1: obstacle
        absl::flat_hash_map<int, Eigen::MatrixXd> m_up_stairs_cost_maps_ = {};                          // cost maps to go upstairs for each floor
        absl::flat_hash_map<int, PathMatrix> m_up_stairs_path_maps_ = {};                               // path maps to go upstairs for each floor
        absl::flat_hash_map<int, Eigen::MatrixXd> m_down_stairs_cost_maps_ = {};                        // cost maps to go downstairs for each floor
        absl::flat_hash_map<int, PathMatrix> m_down_stairs_path_maps_ = {};                             // path maps to go downstairs for each floor
        absl::flat_hash_map<int, absl::flat_hash_map<int, LocalCostMap>> m_room_cost_maps_ = {};        // cost maps to go to each room, key: room id
        absl::flat_hash_map<int, Eigen::MatrixX<std::unordered_set<int>>> m_object_reached_maps_ = {};  // object reached maps for each floor
        absl::flat_hash_map<int, LocalCostMap> m_object_cost_maps_ = {};                                // cost maps to reach each object, key: object id
        std::vector<AtomicAction> m_atomic_actions_ = {};                                               // atomic actions
        int m_floor_up_action_id_ = 0;                                                                  // atomic action id to go upstairs
        int m_floor_down_action_id_ = 0;                                                                // atomic action id to go downstairs

        friend class search_planning::LLMSceneGraphHeuristic;

    public:
        explicit EnvironmentSceneGraph(std::shared_ptr<scene_graph::Building> scene_graph, std::shared_ptr<Setting> setting = nullptr)
            : m_setting_(std::move(setting)),
              m_scene_graph_(std::move(scene_graph)) {
            ERL_ASSERTM(m_scene_graph_ != nullptr, "scene_graph should not be nullptr.");
            if (m_setting_ == nullptr) { m_setting_ = std::make_shared<Setting>(); }
            GenerateAtomicActions();
            LoadMaps();
        }

        template<typename T>
        [[nodiscard]] std::shared_ptr<T>
        GetSetting() const {
            return std::dynamic_pointer_cast<T>(m_setting_);
        }

        [[nodiscard]] std::shared_ptr<common::GridMapInfo3D>
        GetGridMapInfo() const {
            return m_grid_map_info_;
        }

        [[nodiscard]] std::size_t
        GetNumResolutionLevels() const override {
            // 0: anchor
            // 1: kNA
            // 2: kObject
            // 3: kRoom
            // 4: kFloor
            return static_cast<int>(m_setting_->max_level) + 2;
        }

        [[nodiscard]] std::size_t
        GetStateSpaceSize() const override {
            return m_grid_map_info_->Size();
        }

        [[nodiscard]] std::size_t
        GetActionSpaceSize() const override {
            /**
             * (level, goal_id)
             * level  | num of goal_id
             * floor  | 2
             * room   | m_scene_graph_->room_ids.size()
             * object | m_scene_graph_->object_ids.size()
             * atomic | m_atomic_actions_.size()
             */
            return 2 + m_scene_graph_->room_ids.size() + m_scene_graph_->object_ids.size() + m_atomic_actions_.size();
        }

        /**
         * @brief Get the trajectory starting from the given state and following the given action.
         * @param env_state
         * @param action_coords (level, goal_id)
         * @return
         */
        [[nodiscard]] std::vector<std::shared_ptr<EnvironmentState>>
        ForwardAction(const std::shared_ptr<const EnvironmentState> &env_state, const std::vector<int> &action_coords) const override;

        [[nodiscard]] std::vector<Successor>
        GetSuccessors(const std::shared_ptr<EnvironmentState> &env_state) const override {  // NOLINT(*-no-recursion)
            std::vector<Successor> successors;
            successors.reserve(m_scene_graph_->object_ids.size() + m_scene_graph_->room_ids.size());
            for (auto level: {
                     scene_graph::Node::Type::kOcc,
                     scene_graph::Node::Type::kObject,
                     scene_graph::Node::Type::kRoom,
                     scene_graph::Node::Type::kFloor,
                 }) {
                if (level > m_setting_->max_level) { break; }
                std::vector<Successor> level_successors = GetSuccessorsAtLevel(env_state, static_cast<std::size_t>(level) + 1);
                successors.insert(successors.end(), level_successors.begin(), level_successors.end());
            }
            return successors;
        }

        [[nodiscard]] std::vector<Successor>
        GetSuccessorsAtLevel(const std::shared_ptr<EnvironmentState> &env_state, std::size_t resolution_level) const override;

        [[nodiscard]] bool
        InStateSpace(const std::shared_ptr<EnvironmentState> &env_state) const override {
            return m_grid_map_info_->InGrids(env_state->grid);
        }

        [[nodiscard]] bool
        InStateSpaceAtLevel(const std::shared_ptr<EnvironmentState> &env_state, std::size_t resolution_level) const override {
            if (resolution_level == 0) { return m_grid_map_info_->InGrids(env_state->grid); }
            auto level = static_cast<scene_graph::Node::Type>(resolution_level - 1);
            if (!m_grid_map_info_->InGrids(env_state->grid)) { return false; }
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
            return m_grid_map_info_->GridToIndex(env_state->grid, true);
        }

        [[nodiscard]] Eigen::VectorXi
        MetricToGrid(const Eigen::Ref<const Eigen::VectorXd> &metric_state) const override {
            Eigen::VectorXi grid;
            grid.resize(3);
            grid[0] = m_grid_map_info_->MeterToGridForValue(metric_state[0], 0);
            grid[1] = m_grid_map_info_->MeterToGridForValue(metric_state[1], 1);
            grid[2] = m_grid_map_info_->MeterToGridForValue(metric_state[2], 2);
            return grid;
        }

        [[nodiscard]] Eigen::VectorXd
        GridToMetric(const Eigen::Ref<const Eigen::VectorXi> &grid_state) const override {
            Eigen::VectorXd metric;
            metric.resize(3);
            metric[0] = m_grid_map_info_->GridToMeterForValue(grid_state[0], 0);
            metric[1] = m_grid_map_info_->GridToMeterForValue(grid_state[1], 1);
            metric[2] = m_grid_map_info_->GridToMeterForValue(grid_state[2], 2);
            return metric;
        }

        [[nodiscard]] cv::Mat
        ShowPaths(const std::map<int, Eigen::MatrixXd> &, bool) const override {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

    protected:
        void
        LoadMaps();

        void
        GenerateAtomicActions();

        void
        GenerateFloorCostMaps();

        void
        GenerateRoomCostMaps();

        void
        GenerateObjectCostMaps();

        void
        ReverseAStar(const Eigen::Ref<Eigen::Matrix2Xi> &goals, const cv::Mat &obstacle_map, Eigen::MatrixXd &cost_map, PathMatrix &path_map) const {
            Eigen::MatrixX<std::vector<uint32_t>> action_map;  // empty
            Eigen::MatrixXi goal_index_map;                    // empty
            ReverseAStar(goals, obstacle_map, cost_map, path_map, action_map, goal_index_map);
        }

        /**
         * @brief reverse A* search to compute the cost of a composite action
         * @param goals
         * @param obstacle_map
         * @param cost_map
         * @param path_map
         * @param action_map
         * @param goal_index_map
         * @return
         */
        void
        ReverseAStar(
            const Eigen::Ref<Eigen::Matrix2Xi> &goals,
            const cv::Mat &obstacle_map,
            Eigen::MatrixXd &cost_map,
            PathMatrix &path_map,
            Eigen::MatrixX<std::vector<uint32_t>> &action_map,
            Eigen::MatrixXi &goal_index_map) const;

    private:
        [[nodiscard]] std::vector<std::shared_ptr<EnvironmentState>>
        ConvertPath(const std::vector<std::array<int, 2>> &path, const int floor_num) const {
            std::vector<std::shared_ptr<EnvironmentState>> next_env_states;
            next_env_states.reserve(path.size() + 1);
            for (const auto &point: path) {
                auto next_env_state = std::make_shared<EnvironmentState>();
                next_env_state->grid.resize(3);
                next_env_state->grid[0] = point[0];
                next_env_state->grid[1] = point[1];
                next_env_state->grid[2] = floor_num;
                next_env_state->metric = GridToMetric(next_env_state->grid);
                next_env_states.push_back(next_env_state);
            }
            return next_env_states;
        }

        [[nodiscard]] std::vector<std::shared_ptr<EnvironmentState>>
        GetPathToFloor(const int xg, const int yg, const int floor_num, const int next_floor_num) const {
            ERL_DEBUG_ASSERT(std::abs(floor_num - next_floor_num) == 1, "floor_num and next_floor_num should differ by 1.");
            std::vector<std::shared_ptr<EnvironmentState>> next_env_states;
            if (floor_num < next_floor_num) {  // go upstairs
                const auto &path = m_up_stairs_path_maps_.at(floor_num)(xg, yg);
                next_env_states = ConvertPath(path, floor_num);
            } else {  // go downstairs
                const auto &path = m_down_stairs_path_maps_.at(floor_num)(xg, yg);
                next_env_states = ConvertPath(path, floor_num);
            }
            const auto next_env_state = std::make_shared<EnvironmentState>();
            const auto &floor = m_scene_graph_->floors.at(next_floor_num);
            next_env_state->grid.resize(3);
            if (floor_num < next_floor_num) {  // go upstairs
                next_env_state->grid[0] = floor->down_stairs_portal.value()[0];
                next_env_state->grid[1] = floor->down_stairs_portal.value()[1];
            } else {  // go downstairs
                next_env_state->grid[0] = floor->up_stairs_portal.value()[0];
                next_env_state->grid[1] = floor->up_stairs_portal.value()[1];
            }
            next_env_state->grid[2] = floor->id;
            next_env_state->metric = GridToMetric(next_env_state->grid);
            next_env_states.push_back(next_env_state);
            return next_env_states;
        }
    };

}  // namespace erl::env

namespace YAML {
    template<>
    struct convert<erl::env::EnvironmentSceneGraph::Setting> {
        static Node
        encode(const erl::env::EnvironmentSceneGraph::Setting &rhs) {
            Node node;
            node["data_dir"] = rhs.data_dir;
            node["num_threads"] = rhs.num_threads;
            node["allow_diagonal"] = rhs.allow_diagonal;
            node["object_reach_distance"] = rhs.object_reach_distance;
            node["shape"] = rhs.shape;
            return node;
        }

        static bool
        decode(const Node &node, erl::env::EnvironmentSceneGraph::Setting &rhs) {
            if (!node.IsMap()) { return false; }
            rhs.data_dir = node["data_dir"].as<std::string>();
            rhs.num_threads = node["num_threads"].as<long>();
            rhs.allow_diagonal = node["allow_diagonal"].as<bool>();
            rhs.object_reach_distance = node["object_reach_distance"].as<double>();
            rhs.shape = node["shape"].as<Eigen::Matrix2Xd>();
            return true;
        }
    };

    inline Emitter &
    operator<<(Emitter &out, const erl::env::EnvironmentSceneGraph::Setting &rhs) {
        out << BeginMap;
        out << Key << "data_dir" << Value << rhs.data_dir;
        out << Key << "num_threads" << Value << rhs.num_threads;
        out << Key << "allow_diagonal" << Value << rhs.allow_diagonal;
        out << Key << "object_reach_distance" << Value << rhs.object_reach_distance;
        out << Key << "shape" << Value << rhs.shape;
        out << EndMap;
        return out;
    }
}  // namespace YAML
