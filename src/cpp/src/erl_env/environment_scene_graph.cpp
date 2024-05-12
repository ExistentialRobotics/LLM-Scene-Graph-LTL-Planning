#include <boost/heap/d_ary_heap.hpp>
#include "erl_env/environment_scene_graph.hpp"

namespace erl::env {

    std::vector<std::shared_ptr<EnvironmentState>>
    EnvironmentSceneGraph::ForwardAction(const std::shared_ptr<const EnvironmentState> &env_state, const std::vector<int> &action_coords) const {
        const auto level = static_cast<scene_graph::Node::Type>(action_coords[0]);
        const int &cur_x = env_state->grid[0];
        const int &cur_y = env_state->grid[1];
        const int &cur_z = env_state->grid[2];
        ERL_DEBUG("forward action, x: %d, y: %d, floor_num: %d, cur_room_id: %d", cur_x, cur_y, cur_z, m_room_maps_[cur_z].at<int>(cur_x, cur_y));
        switch (level) {
            case scene_graph::Node::Type::kOcc: {  // atomic action
                const int &atomic_action_id = action_coords[1];
                ERL_DEBUG("kNA action id: %d", atomic_action_id);
                ERL_DEBUG_ASSERT(
                    atomic_action_id >= 0 && static_cast<std::size_t>(atomic_action_id) < m_atomic_actions_.size(),
                    "atomic_action_id is out of range.");
                if (static_cast<std::size_t>(atomic_action_id) < m_atomic_actions_.size() - 2) {
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid = env_state->grid + m_atomic_actions_[atomic_action_id].state_diff;
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    return {next_env_state};
                }
                if (static_cast<std::size_t>(atomic_action_id) == m_atomic_actions_.size() - 2) {  // floor up
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(3);
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    int &nz = next_env_state->grid[2];
                    nz = cur_z + 1;  // go upstairs
                    const auto &floor = m_scene_graph_->floors.at(nz);
                    nx = floor->down_stairs_portal.value()[0];
                    ny = floor->down_stairs_portal.value()[1];
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    return {next_env_state};
                }
                if (static_cast<std::size_t>(atomic_action_id) == m_atomic_actions_.size() - 1) {  // floor down
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(3);
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    int &nz = next_env_state->grid[2];
                    nz = cur_z - 1;  // go downstairs
                    const auto &floor = m_scene_graph_->floors.at(nz);
                    nx = floor->up_stairs_portal.value()[0];
                    ny = floor->up_stairs_portal.value()[1];
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    return {next_env_state};
                }
                throw std::runtime_error("Invalid atomic action.");
            }
            case scene_graph::Node::Type::kObject: {  // reach object
                const int &goal_object_id = action_coords[1];
                ERL_DEBUG("kObject action id: %d", goal_object_id);
                const auto &[grid_min_x, grid_min_y, grid_max_x, grid_max_y, cost_map, path_map] = m_object_cost_maps_.at(goal_object_id);
#ifndef NDEBUG
                auto &object = m_scene_graph_->id_to_object[goal_object_id];
                const int &room_id = object->parent_id;
                auto &room = m_scene_graph_->id_to_room[room_id];
                ERL_ASSERTM(room->parent_id == cur_z, "On %d floor but action is to reach object on %d floor.", cur_z, room->parent_id);
                const int &at_room_id = m_room_maps_[cur_z].at<int>(cur_x, cur_y);
                ERL_ASSERTM(at_room_id == room_id, "In room %d but action is to reach object in room %d.", at_room_id, room_id);
                ERL_ASSERTM(
                    (cur_x >= grid_min_x && cur_x <= grid_max_x) && (cur_y >= grid_min_y && cur_y <= grid_max_y),
                    "Not in the local cost map of object (id: %d) to reach.",
                    goal_object_id);
#endif
                auto &path = path_map(cur_x - grid_min_x, cur_y - grid_min_y);
                return ConvertPath(path, cur_z);
            }
            case scene_graph::Node::Type::kRoom: {  // reach room
                const int &goal_room_id = action_coords[1];
                const int &at_room_id = m_room_maps_[cur_z].at<int>(cur_x, cur_y);
                ERL_DEBUG("kRoom action id: %d, at_room_id: %d", goal_room_id, at_room_id);
                const auto &[grid_min_x, grid_min_y, grid_max_x, grid_max_y, cost_map, path_map] = m_room_cost_maps_.at(at_room_id).at(goal_room_id);
#ifndef NDEBUG
                auto &room = m_scene_graph_->id_to_room[goal_room_id];
                ERL_ASSERTM(room->parent_id == cur_z, "On %d floor but action is to reach room on %d floor.", cur_z, room->parent_id);
                ERL_ASSERTM(
                    (cur_x >= grid_min_x && cur_x < grid_max_x) && (cur_y >= grid_min_y && cur_y < grid_max_y),
                    "Not in the local cost map of room (id: %d) to take the action.",
                    goal_room_id);
#endif
                auto &path = path_map(cur_x - grid_min_x, cur_y - grid_min_y);
                return ConvertPath(path, cur_z);
            }
            case scene_graph::Node::Type::kFloor: {  // floor up or down
                const int &goal_floor_num = action_coords[1];
                ERL_DEBUG("kFloor action id: %d", goal_floor_num);
                return GetPathToFloor(cur_x, cur_y, cur_z, goal_floor_num);
            }
            case scene_graph::Node::Type::kBuilding:
                throw std::runtime_error("No action for building.");
            default:
                throw std::runtime_error("Invalid action: unknown level.");
        }
    }

    std::vector<Successor>
    EnvironmentSceneGraph::GetSuccessorsAtLevel(const std::shared_ptr<EnvironmentState> &env_state, std::size_t resolution_level) const {
        if (!InStateSpace(env_state)) { return {}; }
        if (resolution_level == 0) { return GetSuccessors(env_state); }
        auto level = static_cast<scene_graph::Node::Type>(resolution_level - 1);
        int &cur_x = env_state->grid[0];
        int &cur_y = env_state->grid[1];
        int &cur_z = env_state->grid[2];
        std::vector<Successor> successors;
        switch (level) {
            case scene_graph::Node::Type::kOcc: {
                for (int atomic_action_id = 0; atomic_action_id < m_floor_up_action_id_; ++atomic_action_id) {  // grid movement
                    auto &atomic_action = m_atomic_actions_[atomic_action_id];
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(3);
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    nx = cur_x + atomic_action.state_diff[0];
                    if (nx < 0 || nx >= m_grid_map_info_->Shape(0)) { continue; }  // out of map boundary
                    ny = cur_y + atomic_action.state_diff[1];
                    if (ny < 0 || ny >= m_grid_map_info_->Shape(1)) { continue; }       // out of map boundary
                    if (m_obstacle_maps_[cur_z].at<uint8_t>(nx, ny) > 0) { continue; }  // obstacle
                    next_env_state->grid[2] = cur_z;
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    // the room maps may have some small regions marked N/A due to the original mesh processing, we should skip it.
                    if (m_room_maps_[cur_z].at<int>(nx, ny) <= 0) { continue; }  // room id missing, skip it.
                    successors.emplace_back(next_env_state, atomic_action.cost, std::vector{static_cast<int>(scene_graph::Node::Type::kOcc), atomic_action_id});
                }
                // going up/down stairs only happens when the robot arrives at the stairs portal
                auto &floor = m_scene_graph_->floors.at(cur_z);
                if (int floor_num_up = cur_z + 1; floor_num_up < m_scene_graph_->num_floors && floor->up_stairs_portal.has_value() &&
                                                  cur_x == floor->up_stairs_portal.value()[0] && cur_y == floor->up_stairs_portal.value()[1]) {  // go upstairs
                    auto &floor_up = m_scene_graph_->floors.at(floor_num_up);
                    // ERL_ASSERTM(floor->up_stairs_portal.has_value(), "floor->up_stairs_portal should have value.");
                    ERL_ASSERTM(floor_up->down_stairs_portal.has_value(), "floor_up->down_stairs_portal should have value.");
                    if (const double &cost = floor->up_stairs_cost; !std::isinf(cost)) {  // no path to go upstairs (should not happen, but just in case)
                        auto next_env_state = std::make_shared<EnvironmentState>();
                        next_env_state->grid.resize(3);
                        next_env_state->grid[0] = floor_up->down_stairs_portal.value()[0];
                        next_env_state->grid[1] = floor_up->down_stairs_portal.value()[1];
                        next_env_state->grid[2] = floor_num_up;
                        next_env_state->metric = GridToMetric(next_env_state->grid);
                        successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kOcc), m_floor_up_action_id_});
                    }
                }
                if (int floor_num_down = cur_z - 1; floor_num_down >= 0 && floor->down_stairs_portal.has_value() &&
                                                    cur_x == floor->down_stairs_portal.value()[0] &&
                                                    cur_y == floor->down_stairs_portal.value()[1]) {  // go downstairs
                    const auto &floor_down = m_scene_graph_->floors.at(floor_num_down);
                    // ERL_ASSERTM(floor->down_stairs_portal.has_value(), "floor->down_stairs_portal should have value.");
                    ERL_ASSERTM(floor_down->up_stairs_portal.has_value(), "floor_down->up_stairs_portal should have value.");
                    const double &cost = floor->down_stairs_cost;  // m_down_stairs_cost_maps_.at(cur_z)(cur_x, cur_y);
                    if (std::isinf(cost)) { return successors; }   // no path to go downstairs (should not happen, but just in case)
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(3);
                    next_env_state->grid[0] = floor_down->up_stairs_portal.value()[0];
                    next_env_state->grid[1] = floor_down->up_stairs_portal.value()[1];
                    next_env_state->grid[2] = floor_num_down;
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kOcc), m_floor_down_action_id_});
                }
                return successors;
            }
            case scene_graph::Node::Type::kObject: {
                int at_room_id = m_room_maps_.at(cur_z).at<int>(cur_x, cur_y);
                auto &room = m_scene_graph_->id_to_room.at(at_room_id);
                auto &reached_object_ids = m_object_reached_maps_.at(cur_z)(cur_x, cur_y);
                if (reached_object_ids.empty()) { return successors; }  // no object reached at this grid
                successors.reserve(room->objects.size() - reached_object_ids.size());
                for (const auto &[object_id, object]: room->objects) {
                    if (reached_object_ids.count(object_id) > 0) { continue; }  // already reached
                    const auto &[grid_min_x, grid_min_y, grid_max_x, grid_max_y, cost_map, path_map] = m_object_cost_maps_.at(object_id);
                    ERL_DEBUG_ASSERT(grid_min_x <= cur_x && cur_x <= grid_max_x, "x is out of range.");
                    ERL_DEBUG_ASSERT(grid_min_y <= cur_y && cur_y <= grid_max_y, "y is out of range.");
                    int r = cur_x - grid_min_x;
                    int c = cur_y - grid_min_y;
                    auto &path = path_map(r, c);          // path to reach the object
                    if (path.empty()) { continue; }       // cannot reach the object
                    const double &cost = cost_map(r, c);  // cost to reach the object
                    if (std::isinf(cost)) { continue; }   // cannot reach the object
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(3);
                    next_env_state->grid[0] = path.back()[0];
                    next_env_state->grid[1] = path.back()[1];
                    next_env_state->grid[2] = cur_z;
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kObject), object_id});
                }
                return successors;
            }
            case scene_graph::Node::Type::kRoom: {
                int at_room_id = m_room_maps_.at(cur_z).at<int>(cur_x, cur_y);
                auto connected_room_cost_maps_itr = m_room_cost_maps_.find(at_room_id);
                if (connected_room_cost_maps_itr == m_room_cost_maps_.end()) { return successors; }  // no action for this room
                auto &connected_room_cost_maps = connected_room_cost_maps_itr->second;
                auto &room = m_scene_graph_->id_to_room.at(at_room_id);
                ERL_DEBUG_ASSERT(room->parent_id == cur_z, "On %d floor but action is to reach room on %d floor.", cur_z, room->parent_id);
                ERL_DEBUG_ASSERT(room->id == at_room_id, "In room %d but action is to reach room %d.", at_room_id, room->id);
                successors.reserve(room->connected_room_ids.size());
                for (int &connected_room_id: room->connected_room_ids) {
                    ERL_DEBUG_ASSERT(m_room_maps_.at(cur_z).at<int>(cur_x, cur_y) != connected_room_id, "The current state is in the connected room.");
                    ERL_DEBUG_ASSERT(connected_room_id != room->id, "Room %d is connected to itself.", room->id);  // self-connected (loop)
                    auto local_cost_map_itr = connected_room_cost_maps.find(connected_room_id);
                    if (local_cost_map_itr == connected_room_cost_maps.end()) { continue; }  // no action to go to this room
                    const auto &[grid_min_x, grid_min_y, grid_max_x, grid_max_y, cost_map, path_map] = local_cost_map_itr->second;
                    int r = cur_x - grid_min_x;
                    int c = cur_y - grid_min_y;
                    auto &path = path_map(r, c);
                    const double &cost = cost_map(r, c);
                    if (path.empty()) { continue; }  // cannot reach the room
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(3);
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    nx = path.back()[0];
                    ny = path.back()[1];
                    ERL_DEBUG_ASSERT(m_room_maps_.at(cur_z).at<int>(nx, ny) == connected_room_id, "The next state is not in the connected room.");
                    next_env_state->grid[2] = cur_z;
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kRoom), connected_room_id});
                }
                return successors;
            }
            case scene_graph::Node::Type::kFloor: {
                int floor_num_up = cur_z + 1;
                int floor_num_down = cur_z - 1;
                successors.reserve(2);
                if (floor_num_up < m_scene_graph_->num_floors) {  // go upstairs
                    auto &floor = m_scene_graph_->floors.at(floor_num_up);
                    if (const double &cost = m_up_stairs_cost_maps_.at(cur_z)(cur_x, cur_y); !std::isinf(cost)) {
                        auto next_env_state = std::make_shared<EnvironmentState>();
                        next_env_state->grid.resize(3);
                        next_env_state->grid[0] = floor->down_stairs_portal.value()[0];
                        next_env_state->grid[1] = floor->down_stairs_portal.value()[1];
                        next_env_state->grid[2] = floor_num_up;
                        next_env_state->metric = GridToMetric(next_env_state->grid);
                        successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kFloor), floor_num_up});
                    }
                }
                if (floor_num_down >= 0) {  // go downstairs
                    auto &floor = m_scene_graph_->floors.at(floor_num_down);
                    const double &cost = m_down_stairs_cost_maps_.at(cur_z)(cur_x, cur_y);
                    if (std::isinf(cost)) { return successors; }
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(3);
                    next_env_state->grid[0] = floor->up_stairs_portal.value()[0];
                    next_env_state->grid[1] = floor->up_stairs_portal.value()[1];
                    next_env_state->grid[2] = floor_num_down;
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kFloor), floor_num_down});
                }
                return successors;
            }
            case scene_graph::Node::Type::kBuilding:
                throw std::runtime_error("No action for building.");
            default:
                throw std::runtime_error("Invalid action: unknown level.");
        }
    }

    void
    EnvironmentSceneGraph::LoadMaps() {
        const auto &floor = m_scene_graph_->floors[0];
        const auto floor_grid_map_info = std::make_shared<common::GridMapInfo2D>(floor->grid_map_origin, floor->grid_map_resolution, floor->grid_map_size);

        // load room maps, category maps and obstacle maps, initialize object reach maps
        m_room_maps_.reserve(m_scene_graph_->num_floors);
        m_cat_maps_.reserve(m_scene_graph_->num_floors);
        for (int i = 0; i < m_scene_graph_->num_floors; ++i) {
            m_room_maps_.push_back(m_scene_graph_->LoadRoomMap(m_setting_->data_dir, i));  // room map, CV_32SC1
            m_cat_maps_.push_back(m_scene_graph_->LoadCatMap(m_setting_->data_dir, i));    // category map, CV_32SC1
            cv::Mat &cat_map = m_cat_maps_.back();
            m_ground_masks_.push_back(cat_map != static_cast<int>(scene_graph::Object::SOC::kGround));  // CV_8UC1
            m_obstacle_maps_.push_back(m_ground_masks_.back().clone());                                 // CV_8UC1, 0: free, >=1: obstacle
            if (m_setting_->shape.cols() > 0) {                                                         // inflate the obstacle map if the robot shape is given
                cv::Mat mask_stairs_up = cat_map != static_cast<int>(scene_graph::Object::SOC::kStairsUp);
                cv::Mat mask_stairs_down = cat_map != static_cast<int>(scene_graph::Object::SOC::kStairsDown);
                cv::Mat mask_stairs = mask_stairs_up & mask_stairs_down;
                cv::Mat &obstacle_map = m_obstacle_maps_.back();
                obstacle_map &= mask_stairs;                                                           // when inflate the map, do not inflate the stairs
                InflateGridMap2D(obstacle_map, obstacle_map, floor_grid_map_info, m_setting_->shape);  // inflate the obstacle map
                obstacle_map |= ~mask_stairs;                                                          // remove the stairs
            }
        }

        GenerateFloorCostMaps();                                                                      // needed by kNA and kFloor actions
        if (m_setting_->max_level >= scene_graph::Node::Type::kRoom) { GenerateRoomCostMaps(); }      // needed by kRoom actions
        if (m_setting_->max_level >= scene_graph::Node::Type::kObject) { GenerateObjectCostMaps(); }  // needed by kObject actions
    }

    void
    EnvironmentSceneGraph::GenerateAtomicActions() {
        const auto &floor = m_scene_graph_->floors[0];
        double floor_height = std::numeric_limits<double>::infinity();
        if (m_scene_graph_->num_floors == 1) {
            floor_height = 2.0;  // default height for one-floor building, which is not used in the heuristic
        } else {
            for (int i = 1; i < m_scene_graph_->num_floors; ++i) {
                // get minimum height between two floors to make sure the heuristics is admissible
                if (const double height = m_scene_graph_->floors[i]->ground_z - m_scene_graph_->floors[i - 1]->ground_z; height < floor_height) {
                    floor_height = height;
                }
            }
        }
        Eigen::Vector3d grid_map_origin(floor->grid_map_origin.x(), floor->grid_map_origin.y(), floor->ground_z - 0.5 * floor_height);
        Eigen::Vector3d grid_map_resolution(floor->grid_map_resolution.x(), floor->grid_map_resolution.y(), floor_height);
        Eigen::Vector3i grid_map_size(floor->grid_map_size.x(), floor->grid_map_size.y(), m_scene_graph_->num_floors);
        m_grid_map_info_ = std::make_shared<common::GridMapInfo3D>(grid_map_origin, grid_map_resolution, grid_map_size);

        const double x_res = m_grid_map_info_->Resolution(0);
        const double y_res = m_grid_map_info_->Resolution(1);
        if (m_setting_->allow_diagonal) {
            m_atomic_actions_.reserve(10);
            m_floor_up_action_id_ = 8;
            m_floor_down_action_id_ = 9;
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    if (i == j && i == 0) { continue; }
                    const double dx = i * x_res;
                    const double dy = j * y_res;
                    m_atomic_actions_.emplace_back(std::sqrt(dx * dx + dy * dy), Eigen::Vector3i(i, j, 0));
                }
            }
        } else {
            m_atomic_actions_.reserve(6);
            m_floor_up_action_id_ = 4;
            m_floor_down_action_id_ = 5;
            for (int i: {-1, 1}) {
                m_atomic_actions_.emplace_back(std::abs(x_res), Eigen::Vector3i(i, 0, 0));
                m_atomic_actions_.emplace_back(std::abs(y_res), Eigen::Vector3i(0, i, 0));
            }
        }
        m_atomic_actions_.emplace_back(-1., Eigen::Vector3i(0, 0, 1));   // floor up
        m_atomic_actions_.emplace_back(-1., Eigen::Vector3i(0, 0, -1));  // floor down
    }

    void
    EnvironmentSceneGraph::GenerateFloorCostMaps() {
        if (m_scene_graph_->num_floors <= 1) { return; }  // no need to generate up/down stairs cost maps
        int grid_map_rows = m_grid_map_info_->Shape(0);
        int grid_map_cols = m_grid_map_info_->Shape(1);
        for (int floor_num = 1; floor_num < m_scene_graph_->num_floors; ++floor_num) {  // initialize cost maps
            int n1 = floor_num - 1;
            int n2 = floor_num;
            ERL_ASSERTM(m_scene_graph_->floors[n1]->up_stairs_portal.has_value(), "Floor %d does not have up stairs portal.", n1);
            ERL_ASSERTM(m_scene_graph_->floors[n2]->down_stairs_portal.has_value(), "Floor %d does not have down stairs portal.", n2);
            m_up_stairs_cost_maps_[floor_num - 1].resize(grid_map_rows, grid_map_cols);
            m_up_stairs_path_maps_[floor_num - 1].resize(grid_map_rows, grid_map_cols);
            m_down_stairs_cost_maps_[floor_num].resize(grid_map_rows, grid_map_cols);
            m_down_stairs_path_maps_[floor_num].resize(grid_map_rows, grid_map_cols);
        }

        auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for default(none)   \
    shared(m_scene_graph_,               \
               m_obstacle_maps_,         \
               m_up_stairs_cost_maps_,   \
               m_up_stairs_path_maps_,   \
               m_down_stairs_cost_maps_, \
               m_down_stairs_path_maps_, \
               Eigen::Dynamic)
        for (int floor_num = 1; floor_num < m_scene_graph_->num_floors; ++floor_num) {  // initialize cost maps
            int n1 = floor_num - 1;
            auto &floor1 = m_scene_graph_->floors[n1];
            Eigen::Vector2i goal(floor1->up_stairs_portal.value()[0], floor1->up_stairs_portal.value()[1]);
            cv::Mat &obstacle_map1 = m_obstacle_maps_[n1];
            obstacle_map1.at<uint8_t>(goal[0], goal[1]) = 0;  // set the goal to be free
            Eigen::MatrixXd &cost_map1 = m_up_stairs_cost_maps_[n1];
            ReverseAStar(goal, obstacle_map1, cost_map1, m_up_stairs_path_maps_[n1]);
            for (int r = 0; r < obstacle_map1.rows; ++r) {
                for (int c = 0; c < obstacle_map1.cols; ++c) {
                    double &cost = cost_map1(r, c);
                    if (std::isinf(cost)) { continue; }
                    cost += floor1->up_stairs_cost;
                }
            }

            int n2 = floor_num;
            auto &floor2 = m_scene_graph_->floors[n2];
            goal[0] = floor2->down_stairs_portal.value()[0];
            goal[1] = floor2->down_stairs_portal.value()[1];
            cv::Mat &obstacle_map2 = m_obstacle_maps_[n2];
            obstacle_map2.at<uint8_t>(goal[0], goal[1]) = 0;  // set the goal to be free
            Eigen::MatrixXd &cost_map2 = m_down_stairs_cost_maps_[n2];
            ReverseAStar(goal, obstacle_map2, cost_map2, m_down_stairs_path_maps_[n2]);
            for (int r = 0; r < obstacle_map2.rows; ++r) {
                for (int c = 0; c < obstacle_map2.cols; ++c) {
                    double &cost = cost_map2(r, c);
                    if (std::isinf(cost)) { continue; }
                    cost += floor2->down_stairs_cost;
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        ERL_INFO("GenerateFloorCostMaps: %f ms", std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    void
    EnvironmentSceneGraph::GenerateRoomCostMaps() {
        m_room_cost_maps_.reserve(m_scene_graph_->room_ids.size());
        for (int room_id: m_scene_graph_->room_ids) {  // initialize room cost maps
            const auto &room = m_scene_graph_->id_to_room[room_id];
            if (room->name == "staircase") { continue; }
            m_room_cost_maps_[room_id].reserve(room->connected_room_ids.size());
            for (int connected_room_id: room->connected_room_ids) {
                if (m_scene_graph_->id_to_room[connected_room_id]->name == "staircase") { continue; }
                m_room_cost_maps_[room_id][connected_room_id] = {};
            }
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for default(none) shared(m_scene_graph_, m_obstacle_maps_, m_room_cost_maps_)
        for (int room_id: m_scene_graph_->room_ids) {
            const auto &room = m_scene_graph_->id_to_room.at(room_id);
            auto connected_room_cost_maps_itr = m_room_cost_maps_.find(room_id);
            if (connected_room_cost_maps_itr == m_room_cost_maps_.end()) { continue; }  // not initialized
            auto &connected_room_cost_maps = connected_room_cost_maps_itr->second;
            cv::Mat obstacle_map = m_obstacle_maps_[room->parent_id];
            for (int connected_room_id: room->connected_room_ids) {
                auto local_cost_map_itr = connected_room_cost_maps.find(connected_room_id);
                if (local_cost_map_itr == connected_room_cost_maps.end()) { continue; }  // not initialized
                auto &[grid_min_x, grid_min_y, grid_max_x, grid_max_y, cost_map, path_map] = local_cost_map_itr->second;
                grid_min_x = room->grid_map_min.x();
                grid_min_y = room->grid_map_min.y();
                grid_max_x = room->grid_map_max.x();
                grid_max_y = room->grid_map_max.y();
                auto &door_grids = room->door_grids.at(connected_room_id);
                long max_num_goals = door_grids.cols();
                Eigen::Matrix2Xi goals(2, max_num_goals);
                long num_goals = 0;
                for (long i = 0; i < max_num_goals; ++i) {
                    const int &r = door_grids(0, i);
                    const int &c = door_grids(1, i);
                    if (obstacle_map.at<uint8_t>(r, c) > 0) { continue; }  // obstacle
                    goals(0, num_goals) = r;
                    goals(1, num_goals) = c;
                    ++num_goals;
                    // obstacle_map.at<int>(r, c) = 0;  // set the goal to be free
                    if (r < grid_min_x) { grid_min_x = r; }
                    if (r > grid_max_x) { grid_max_x = r; }
                    if (c < grid_min_y) { grid_min_y = c; }
                    if (c > grid_max_y) { grid_max_y = c; }
                }
                goals.conservativeResize(2, num_goals);
                for (long i = 0; i < num_goals; ++i) {
                    goals(0, i) -= grid_min_x;
                    goals(1, i) -= grid_min_y;
                }

                cv::Rect2i room_roi(cv::Point2i(grid_min_y, grid_min_x), cv::Point2i(grid_max_y + 1, grid_max_x + 1));
                cv::Mat obstacle_map_roi = obstacle_map(room_roi);
                cost_map.setConstant(obstacle_map_roi.rows, obstacle_map_roi.cols, std::numeric_limits<double>::infinity());
                path_map.resize(obstacle_map_roi.rows, obstacle_map_roi.cols);
                ReverseAStar(goals, obstacle_map_roi, cost_map, path_map);

                for (int r = 0; r < obstacle_map_roi.rows; ++r) {
                    for (int c = 0; c < obstacle_map_roi.cols; ++c) {
                        if (std::isinf(cost_map(r, c))) { continue; }
                        auto &path = path_map(r, c);
                        for (auto &p: path) {
                            p[0] += grid_min_x;
                            p[1] += grid_min_y;
                        }
                    }
                }
            }
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        ERL_INFO("GenerateRoomCostMaps: %f ms", std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    void
    EnvironmentSceneGraph::GenerateObjectCostMaps() {
        m_object_cost_maps_.reserve(m_scene_graph_->object_ids.size());
        for (int floor_num = 0; floor_num < m_scene_graph_->num_floors; ++floor_num) {
            m_object_reached_maps_[floor_num].resize(m_grid_map_info_->Shape(0), m_grid_map_info_->Shape(1));
        }
        for (int object_id: m_scene_graph_->object_ids) { m_object_cost_maps_[object_id] = {}; }  // initialize object cost maps
        auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for default(none) shared(m_scene_graph_, m_room_maps_, m_obstacle_maps_, m_cat_maps_, m_object_cost_maps_)
        for (int object_id: m_scene_graph_->object_ids) {
            int row_padding = static_cast<int>(m_setting_->object_reach_distance / m_grid_map_info_->Resolution(0)) + 1;
            int col_padding = static_cast<int>(m_setting_->object_reach_distance / m_grid_map_info_->Resolution(1)) + 1;
            auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * row_padding + 1, 2 * col_padding + 1));
            auto &object = m_scene_graph_->id_to_object[object_id];
            auto &room = m_scene_graph_->id_to_room[object->parent_id];
            int floor_id = room->parent_id;
            cv::Mat obstacle_map = m_obstacle_maps_[floor_id];
            auto &[grid_min_x, grid_min_y, grid_max_x, grid_max_y, cost_map, path_map] = m_object_cost_maps_[object_id];
            grid_min_x = room->grid_map_min.x();
            grid_min_y = room->grid_map_min.y();
            grid_max_x = room->grid_map_max.x();
            grid_max_y = room->grid_map_max.y();
            int x, y;
            x = object->grid_map_min[0] - row_padding;
            if (x < grid_min_x) { grid_min_x = x; }
            y = object->grid_map_min[1] - col_padding;
            if (y < grid_min_y) { grid_min_y = y; }
            x = object->grid_map_max[0] + row_padding;
            if (x > grid_max_x) { grid_max_x = x; }
            y = object->grid_map_max[1] + col_padding;
            if (y > grid_max_y) { grid_max_y = y; }
            if (grid_min_x < 0) { grid_min_x = 0; }
            if (grid_min_y < 0) { grid_min_y = 0; }
            x = obstacle_map.rows - 1;
            if (grid_max_x > x) { grid_max_x = x; }
            y = obstacle_map.cols - 1;
            if (grid_max_y > y) { grid_max_y = y; }

            // obstacle map
            cv::Rect2i room_roi(cv::Point2i(grid_min_y, grid_min_x), cv::Point2i(grid_max_y + 1, grid_max_x + 1));
            cv::Mat obstacle_map_roi = obstacle_map(room_roi);

            // get grids that reach the object
            cv::Mat object_seg_mask = m_cat_maps_[floor_id](room_roi) == object_id;  // CV_8UC1, 0: object not reached, 1: object reached
            cv::dilate(object_seg_mask, object_seg_mask, kernel);                    // dilate the object segmentation mask
            int max_num_goals = object_seg_mask.rows * object_seg_mask.cols;
            Eigen::Matrix2Xi goals(2, max_num_goals);
            cv::Mat room_map_roi = m_room_maps_[floor_id](room_roi);
            Eigen::MatrixX<std::unordered_set<int>> &object_reached_map = m_object_reached_maps_[floor_id];
            int num_goals = 0;
            for (int row = 0; row < object_seg_mask.rows; ++row) {
                for (int col = 0; col < object_seg_mask.cols; ++col) {
                    if (room_map_roi.at<int>(row, col) != room->id) {
                        object_seg_mask.at<uint8_t>(row, col) = 0;  // not in the same room
                        continue;
                    }
                    if (obstacle_map_roi.at<uint8_t>(row, col) > 0) { continue; }  // obstacle
                    uint8_t v0 = object_seg_mask.at<uint8_t>(row, col);
                    if (v0 == 0) { continue; }  // object not reached

                    {
#pragma omp critical
                        object_reached_map(row + grid_min_x, col + grid_min_y).insert(object_id);
                    }

                    bool on_boundary = false;
                    if (row < object_seg_mask.rows - 1) {
                        if (object_seg_mask.at<uint8_t>(row + 1, col) != v0) { on_boundary = true; }
                    }
                    if (row > 0) {
                        if (object_seg_mask.at<uint8_t>(row - 1, col) != v0) { on_boundary = true; }
                    }
                    if (col < object_seg_mask.cols - 1) {
                        if (object_seg_mask.at<uint8_t>(row, col + 1) != v0) { on_boundary = true; }
                    }
                    if (col > 0) {
                        if (object_seg_mask.at<uint8_t>(row, col - 1) != v0) { on_boundary = true; }
                    }
                    if (!on_boundary) { continue; }

                    goals(0, num_goals) = row;
                    goals(1, num_goals++) = col;
                }
            }
            goals.conservativeResize(2, num_goals);

            // compute cost maps
            cost_map.setConstant(obstacle_map_roi.rows, obstacle_map_roi.cols, std::numeric_limits<double>::infinity());
            path_map.resize(obstacle_map_roi.rows, obstacle_map_roi.cols);
            // local_cost_map.arg_min_map.setConstant(obstacle_map_roi.rows, obstacle_map_roi.cols, -1);
            ReverseAStar(goals, obstacle_map_roi, cost_map, path_map);

            for (int r = 0; r < obstacle_map_roi.rows; ++r) {
                for (int c = 0; c < obstacle_map_roi.cols; ++c) {
                    if (std::isinf(cost_map(r, c))) { continue; }
                    auto &path = path_map(r, c);
                    for (auto &p: path) {
                        p[0] += grid_min_x;
                        p[1] += grid_min_y;
                    }
                }
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        ERL_INFO("GenerateObjectCostMaps: %f ms", std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    void
    EnvironmentSceneGraph::ReverseAStar(
        const Eigen::Ref<Eigen::Matrix2Xi> &goals,
        const cv::Mat &obstacle_map,
        Eigen::MatrixXd &cost_map,
        PathMatrix &path_map,
        Eigen::MatrixX<std::vector<uint32_t>> &action_map,
        Eigen::MatrixXi &goal_index_map) const {

        ERL_ASSERTM(cost_map.rows() == obstacle_map.rows, "cost_map.rows() != obstacle_map.rows");
        ERL_ASSERTM(cost_map.cols() == obstacle_map.cols, "cost_map.cols() != obstacle_map.cols");
        cost_map.setConstant(std::numeric_limits<double>::infinity());
        long num_goals = goals.cols();
        long goal_index = 0;
        Eigen::Matrix2Xi reachable_goals(2, num_goals);
        for (long i = 0; i < num_goals; ++i) {
            const int &r = goals(0, i);
            const int &c = goals(1, i);
            if (r < 0 || r >= obstacle_map.rows) { continue; }     // out of bound
            if (c < 0 || c >= obstacle_map.cols) { continue; }     // out of bound
            if (obstacle_map.at<uint8_t>(r, c) > 0) { continue; }  // obstacle
            reachable_goals(0, goal_index) = r;
            reachable_goals(1, goal_index++) = c;
        }
        if (goal_index == 0) { return; }  // no reachable goal
        reachable_goals.conservativeResize(2, goal_index);
        num_goals = goal_index;

        struct Node;

        struct QueueItem {
            double f_value = std::numeric_limits<double>::infinity();
            std::shared_ptr<Node> node = nullptr;
            QueueItem() = default;

            QueueItem(const double f, std::shared_ptr<Node> n)
                : f_value(f),
                  node(std::move(n)) {}
        };

        struct Node {
            int x_grid = 0;
            int y_grid = 0;
            std::shared_ptr<Node> parent = nullptr;
            uint32_t action_id = -1;  // action id from parent to this node, also used as goal index if this node is a goal
            double g_value = std::numeric_limits<double>::infinity();
            double h_value = std::numeric_limits<double>::infinity();
        };

        struct Greater {
            bool
            operator()(const std::shared_ptr<QueueItem> &s1, const std::shared_ptr<QueueItem> &s2) const {
                if (std::abs(s1->f_value - s2->f_value) < 1.e-6) {
                    // f value is too close, compare g value
                    return s1->node->g_value > s2->node->g_value;
                }
                return s1->f_value > s2->f_value;
            }
        };

        // clang-format off
        using PriorityQueue = boost::heap::d_ary_heap<
            std::shared_ptr<QueueItem>,
            boost::heap::mutable_<true>,
            boost::heap::arity<8>,
            boost::heap::compare<Greater>>;
        // clang-format on

        Eigen::MatrixXb closed = Eigen::MatrixXb::Constant(obstacle_map.rows, obstacle_map.cols, false);
        Eigen::MatrixXb opened = Eigen::MatrixXb::Constant(obstacle_map.rows, obstacle_map.cols, false);
        Eigen::MatrixX<std::shared_ptr<Node>> nodes(obstacle_map.rows, obstacle_map.cols);
        Eigen::MatrixX<PriorityQueue::handle_type> heap_keys(obstacle_map.rows, obstacle_map.cols);
        PriorityQueue queue;

        auto heuristic_func = [&reachable_goals, &num_goals](const int x, const int y) -> double {
            double h = std::numeric_limits<double>::infinity();
            for (long i = 0; i < num_goals; ++i) {
                const double dx = x - reachable_goals(0, i);
                const double dy = y - reachable_goals(1, i);
                if (const double d = dx * dx + dy * dy; d < h) { h = d; }
            }
            return std::sqrt(h);
        };

        // initialize start nodes
        for (long i = 0; i < num_goals; ++i) {
            auto node = std::make_shared<Node>();
            node->x_grid = reachable_goals(0, i);
            node->y_grid = reachable_goals(1, i);
            node->action_id = i;
            node->g_value = 0;
            node->h_value = 0;
            nodes(node->x_grid, node->y_grid) = node;
            heap_keys(node->x_grid, node->y_grid) = queue.push(std::make_shared<QueueItem>(node->g_value + node->h_value, node));
            cost_map(node->x_grid, node->y_grid) = 0;
        }

        // start search
        // we do search within the obstacle map of a specific floor, so we don't use the floor up/down atomic actions,
        // which are used when generating composite actions
        while (!queue.empty()) {
            auto node = queue.top()->node;
            queue.pop();
            opened(node->x_grid, node->y_grid) = false;
            closed(node->x_grid, node->y_grid) = true;
            uint32_t n = m_setting_->allow_diagonal ? 8 : 4;
            for (uint32_t i = 0; i < n; ++i) {
                auto &action = m_atomic_actions_[i];
                int x_grid = node->x_grid + action.state_diff[0];
                if (x_grid < 0 || x_grid >= obstacle_map.rows) { continue; }  // out of boundary
                int y_grid = node->y_grid + action.state_diff[1];
                if (y_grid < 0 || y_grid >= obstacle_map.cols) { continue; }     // out of boundary
                if (obstacle_map.at<uint8_t>(x_grid, y_grid) > 0) { continue; }  // obstacle
                if (closed(x_grid, y_grid)) { continue; }
                double tentative_g = node->g_value + action.cost;
                double &g_value = cost_map(x_grid, y_grid);
                if (tentative_g >= g_value) { continue; }
                g_value = tentative_g;
                if (opened(x_grid, y_grid)) {
                    auto &heap_key = heap_keys(x_grid, y_grid);
                    (*heap_key)->node->parent = node;
                    (*heap_key)->node->action_id = i;
                    (*heap_key)->f_value = tentative_g + heuristic_func(x_grid, y_grid);
                    queue.increase(heap_key);
                } else {
                    auto child = std::make_shared<Node>();
                    child->x_grid = x_grid;
                    child->y_grid = y_grid;
                    child->parent = node;
                    child->action_id = i;
                    child->g_value = tentative_g;
                    child->h_value = heuristic_func(x_grid, y_grid);
                    nodes(x_grid, y_grid) = child;
                    heap_keys(x_grid, y_grid) = queue.push(std::make_shared<QueueItem>(child->g_value + child->h_value, child));
                    opened(x_grid, y_grid) = true;
                }
            }
        }

        bool get_path = path_map.size() > 0;
        bool get_action = action_map.size() > 0;
        bool get_goal_index = goal_index_map.size() > 0;
        if (!get_path && !get_action && !get_goal_index) { return; }

        // compute path map / action map / goal index map
        int n_reserve = std::max(obstacle_map.rows, obstacle_map.cols);
        for (int r = 0; r < obstacle_map.rows; ++r) {
            for (int c = 0; c < obstacle_map.cols; ++c) {
                auto &node = nodes(r, c);
                if (node == nullptr) { continue; }
                std::vector<std::array<int, 2>> path;
                if (get_path) {
                    path.clear();
                    path.reserve(n_reserve);
                }
                std::vector<uint32_t> actions;
                if (get_action) {
                    actions.clear();
                    actions.reserve(n_reserve);
                }
                while (node->parent != nullptr) {
                    if (get_path) { path.emplace_back(std::array<int, 2>{node->x_grid, node->y_grid}); }
                    if (get_action) { actions.emplace_back(node->action_id); }
                    node = node->parent;
                }
                if (get_path) {
                    path.emplace_back(std::array<int, 2>{node->x_grid, node->y_grid});
                    path.resize(path.size());
                    path_map(r, c) = std::move(path);
                }
                if (get_action) {
                    actions.resize(actions.size());
                    action_map(r, c) = std::move(actions);
                }
                if (get_goal_index) { goal_index_map(r, c) = static_cast<int>(node->action_id); }
            }
        }
    }
}  // namespace erl::env
