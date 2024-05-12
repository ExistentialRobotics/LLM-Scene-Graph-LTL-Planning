#include "erl_env/environment_ltl_scene_graph.hpp"

namespace Eigen::internal {

    template<>
    struct cast_impl<std::bitset<32>, uint32_t> {
        EIGEN_DEVICE_FUNC
        static uint32_t
        run(const std::bitset<32> &x) {
            return static_cast<uint32_t>(x.to_ulong());
        }
    };

    template<>
    struct cast_impl<uint32_t, std::bitset<32>> {
        EIGEN_DEVICE_FUNC
        static std::bitset<32>
        run(const uint32_t &x) {
            return {x};
        }
    };
}  // namespace Eigen::internal

namespace erl::env {

    std::vector<std::shared_ptr<EnvironmentState>>
    EnvironmentLTLSceneGraph::ForwardAction(const std::shared_ptr<const EnvironmentState> &env_state, const std::vector<int> &action_coords) const {
        const auto level = static_cast<scene_graph::Node::Type>(action_coords[0]);
        const int &cur_x = env_state->grid[0];
        const int &cur_y = env_state->grid[1];
        const int &cur_z = env_state->grid[2];
        const int &cur_q = env_state->grid[3];
        switch (level) {
            case scene_graph::Node::Type::kOcc: {  // atomic action
                const int &atomic_action_id = action_coords[1];
                ERL_DEBUG("kNA action id: %d", atomic_action_id);
                ERL_DEBUG_ASSERT(atomic_action_id >= 0 && static_cast<std::size_t>(atomic_action_id) < m_atomic_actions_.size(), "atomic_action_id is out of range.");
                if (static_cast<std::size_t>(atomic_action_id) < m_atomic_actions_.size() - 2) {  // grid movement
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(4);  // x, y, z, q
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    int &nz = next_env_state->grid[2];
                    int &nq = next_env_state->grid[3];
                    nx = cur_x + m_atomic_actions_[atomic_action_id].state_diff[0];                       // next x
                    ny = cur_y + m_atomic_actions_[atomic_action_id].state_diff[1];                       // next y
                    nz = cur_z;                                                                           // next z
                    nq = static_cast<int>(m_fsa_->GetNextState(cur_q, m_label_maps_.at(cur_z)(nx, ny)));  // next LTL state
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    return {next_env_state};
                }
                if (static_cast<std::size_t>(atomic_action_id) == m_atomic_actions_.size() - 2) {  // floor up
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(4);
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    int &nz = next_env_state->grid[2];
                    int &nq = next_env_state->grid[3];
                    nz = cur_z + 1;  // go upstairs
                    const auto &floor = m_scene_graph_->floors.at(nz);
                    nx = floor->down_stairs_portal.value()[0];
                    ny = floor->down_stairs_portal.value()[1];
                    nq = static_cast<int>(m_fsa_->GetNextState(cur_q, m_label_maps_.at(nz)(nx, ny)));
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    return {next_env_state};
                }
                if (static_cast<std::size_t>(atomic_action_id) == m_atomic_actions_.size() - 1) {  // floor down
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(4);
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    int &nz = next_env_state->grid[2];
                    int &nq = next_env_state->grid[3];
                    nz = cur_z - 1;  // go downstairs
                    const auto &floor = m_scene_graph_->floors.at(nz);
                    nx = floor->up_stairs_portal.value()[0];
                    ny = floor->up_stairs_portal.value()[1];
                    nq = static_cast<int>(m_fsa_->GetNextState(cur_q, m_label_maps_.at(nz)(nx, ny)));
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
                    (cur_x >= grid_min_x && cur_x < grid_max_x) && (cur_y >= grid_min_y && cur_y < grid_max_y),
                    "Not in the local cost map of object (id: %d) to reach.",
                    goal_object_id);
#endif
                auto &path = path_map(cur_x - grid_min_x, cur_y - grid_min_y);
                return ConvertPath(path, cur_z, cur_q);
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
                    (cur_x >= grid_min_x && cur_x <= grid_max_x) && (cur_y >= grid_min_y && cur_y <= grid_max_y),
                    "Not in the local cost map of room (id: %d) to take the action.",
                    goal_room_id);
#endif
                auto &path = path_map(cur_x - grid_min_x, cur_y - grid_min_y);
                return ConvertPath(path, cur_z, cur_q);
            }
            case scene_graph::Node::Type::kFloor: {  // floor up or down
                const int &goal_floor_num = action_coords[1];
                ERL_DEBUG("kFloor action id: %d", goal_floor_num);
                return GetPathToFloor(cur_x, cur_y, cur_z, cur_q, goal_floor_num);
            }
            case scene_graph::Node::Type::kBuilding:
                throw std::runtime_error("No action for building.");
            default:
                throw std::runtime_error("Invalid action: unknown level.");
        }
    }

    std::vector<Successor>
    EnvironmentLTLSceneGraph::GetSuccessorsAtLevel(const std::shared_ptr<EnvironmentState> &env_state, std::size_t resolution_level) const {
        if (!InStateSpace(env_state)) { return {}; }
        if (resolution_level == 0) { return EnvironmentSceneGraph::GetSuccessors(env_state); }
        auto level = static_cast<scene_graph::Node::Type>(resolution_level - 1);
        std::vector<Successor> successors;
        int &cur_x = env_state->grid[0];
        int &cur_y = env_state->grid[1];
        int &cur_z = env_state->grid[2];
        int &cur_q = env_state->grid[3];
        auto fsa = reinterpret_cast<Setting *>(m_setting_.get())->fsa;
        switch (level) {
            case scene_graph::Node::Type::kOcc: {
                int num_actions = static_cast<int>(m_atomic_actions_.size()) - 2;
                for (int atomic_action_id = 0; atomic_action_id < num_actions; ++atomic_action_id) {  // grid movement
                    auto &atomic_action = m_atomic_actions_[atomic_action_id];
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(4);  // x, y, z, q
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    int &nz = next_env_state->grid[2];
                    int &nq = next_env_state->grid[3];
                    nx = cur_x + atomic_action.state_diff[0];                                             // next x
                    if (nx < 0 || nx >= m_grid_map_info_->Shape(0)) { continue; }                         // out of map boundary
                    ny = cur_y + atomic_action.state_diff[1];                                             // next y
                    if (ny < 0 || ny >= m_grid_map_info_->Shape(1)) { continue; }                         // out of map boundary
                    if (m_obstacle_maps_[cur_z].at<uint8_t>(nx, ny) > 0) { continue; }                    // obstacle
                    nq = static_cast<int>(m_fsa_->GetNextState(cur_q, m_label_maps_.at(cur_z)(nx, ny)));  // next LTL state
                    if (m_fsa_->IsSinkState(nq)) { continue; }                                            // sink state
                    nz = cur_z;                                                                           // write nz only if nq is not a sink state
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    if (m_room_maps_[cur_z].at<int>(nx, ny) <= 0) { continue; }  // room id missing, skip it.
                    successors.emplace_back(next_env_state, atomic_action.cost, std::vector{static_cast<int>(scene_graph::Node::Type::kOcc), atomic_action_id});
                }
                // going up/down stairs only happens when the robot arrives at the stairs portal
                auto &floor = m_scene_graph_->floors.at(cur_z);
                if (int floor_num_up = cur_z + 1; floor_num_up < m_scene_graph_->num_floors && floor->up_stairs_portal.has_value() &&
                                                  cur_x == floor->up_stairs_portal.value()[0] && cur_y == floor->up_stairs_portal.value()[1]) {  // go upstairs
                    auto &floor_up = m_scene_graph_->floors.at(floor_num_up);
                    ERL_ASSERTM(floor_up->down_stairs_portal.has_value(), "floor_up->down_stairs_portal should have value.");
                    if (const double &cost = floor->up_stairs_cost; !std::isinf(cost)) {
                        auto next_env_state = std::make_shared<EnvironmentState>();
                        next_env_state->grid.resize(4);
                        int &nq = next_env_state->grid[3];
                        auto &dst_q = const_cast<std::vector<int> &>(m_up_stairs_path_q_maps_.at(cur_z)(cur_x, cur_y));
                        if (dst_q.empty()) { dst_q.resize(fsa->num_states, -1); }     // initialize
                        nq = dst_q[cur_q];                                            // read from the cache
                        auto &path = m_up_stairs_path_maps_.at(cur_z)(cur_x, cur_y);  // path to the upstairs portal
                        if (nq < 0) {                                                 // not computed yet
                            nq = cur_q;                                               // initialize
                            bool encounter_sink_state = false;
                            for (auto &point: path) {  // compute the next LTL state
                                nq = static_cast<int>(m_fsa_->GetNextState(nq, m_label_maps_.at(cur_z)(point[0], point[1])));
                                if (m_fsa_->IsSinkState(nq)) {
                                    encounter_sink_state = true;
                                    break;
                                }
                            }
                            if (!encounter_sink_state) {  // one more step to go upstairs
                                nq = static_cast<int>(m_fsa_->GetNextState(
                                    nq,
                                    m_label_maps_.at(floor_num_up)(floor_up->down_stairs_portal.value()[0], floor_up->down_stairs_portal.value()[1])));
                            }
                            dst_q[cur_q] = nq;  // write to the cache
                        }
                        if (!m_fsa_->IsSinkState(nq)) {
                            next_env_state->grid[0] = floor_up->down_stairs_portal.value()[0];
                            next_env_state->grid[1] = floor_up->down_stairs_portal.value()[1];
                            next_env_state->grid[2] = floor_num_up;
                            // nq is already computed or read from the cache
                            next_env_state->metric = GridToMetric(next_env_state->grid);
                            successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kOcc), m_floor_up_action_id_});
                        }
                    }
                }
                if (int floor_num_down = cur_z - 1; floor_num_down >= 0 && floor->down_stairs_portal.has_value() &&
                                                    cur_x == floor->down_stairs_portal.value()[0] &&
                                                    cur_y == floor->down_stairs_portal.value()[1]) {  // go downstairs
                    auto &floor_down = m_scene_graph_->floors.at(floor_num_down);
                    ERL_ASSERTM(floor_down->up_stairs_portal.has_value(), "floor_down->up_stairs_portal should have value.");
                    const double &cost = floor->down_stairs_cost;  // m_down_stairs_cost_maps_.at(cur_z)(cur_x, cur_y);
                    if (std::isinf(cost)) { return successors; }   // no path to go downstairs (should not happen, but just in case)
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(4);
                    int &nq = next_env_state->grid[3];
                    auto &dst_q = const_cast<std::vector<int> &>(m_down_stairs_path_q_maps_.at(cur_z)(cur_x, cur_y));
                    if (dst_q.empty()) { dst_q.resize(fsa->num_states, -1); }       // initialize
                    nq = dst_q[cur_q];                                              // read from the cache
                    auto &path = m_down_stairs_path_maps_.at(cur_z)(cur_x, cur_y);  // path to the downstairs portal
                    if (nq < 0) {                                                   // not computed yet
                        nq = cur_q;
                        bool encounter_sink_state = false;
                        for (auto &point: path) {
                            nq = static_cast<int>(m_fsa_->GetNextState(nq, m_label_maps_.at(cur_z)(point[0], point[1])));
                            if (m_fsa_->IsSinkState(nq)) {
                                encounter_sink_state = true;
                                break;
                            }
                        }
                        if (!encounter_sink_state) {  // one more step to go downstairs
                            nq = static_cast<int>(m_fsa_->GetNextState(
                                nq,
                                m_label_maps_.at(floor_num_down)(floor_down->up_stairs_portal.value()[0], floor_down->up_stairs_portal.value()[1])));
                        }
                        dst_q[cur_q] = nq;  // write to the cache
                    }
                    if (!m_fsa_->IsSinkState(nq)) {
                        next_env_state->grid[0] = floor_down->up_stairs_portal.value()[0];
                        next_env_state->grid[1] = floor_down->up_stairs_portal.value()[1];
                        next_env_state->grid[2] = floor_num_down;
                        // nq is already computed or read from the cache
                        next_env_state->metric = GridToMetric(next_env_state->grid);
                        successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kOcc), m_floor_down_action_id_});
                    }
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
                    if (path.empty()) { continue; }       // no path to reach the object
                    const double &cost = cost_map(r, c);  // cost to reach the object
                    if (std::isinf(cost)) { continue; }   // no path to reach the object (should not happen, but just in case)
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(4);
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    int &nz = next_env_state->grid[2];
                    int &nq = next_env_state->grid[3];
                    auto &dst_q = const_cast<std::vector<int> &>(m_object_path_q_maps_.at(object_id)(r, c));
                    if (dst_q.empty()) { dst_q.resize(fsa->num_states, -1); }  // initialize
                    nq = dst_q[cur_q];                                         // read from the cache
                    if (nq < 0) {                                              // not computed yet
                        nq = cur_q;                                            // initialize
                        for (auto &point: path) {                              // compute the next LTL state
                            nq = static_cast<int>(m_fsa_->GetNextState(nq, m_label_maps_.at(cur_z)(point[0], point[1])));
                            if (m_fsa_->IsSinkState(nq)) { break; }  // sink state
                        }
                        dst_q[cur_q] = nq;  // write to the cache
                    }
                    if (m_fsa_->IsSinkState(nq)) { continue; }  // sink state
                    nx = path.back()[0];
                    ny = path.back()[1];
                    nz = cur_z;
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
                    if (std::isinf(cost)) { continue; }  // no path to reach the room (should not happen, but just in case)
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(4);
                    int &nx = next_env_state->grid[0];
                    int &ny = next_env_state->grid[1];
                    int &nz = next_env_state->grid[2];
                    int &nq = next_env_state->grid[3];
                    auto &dst_q = const_cast<std::vector<int> &>(m_room_path_q_maps_.at(at_room_id).at(connected_room_id)(r, c));
                    if (dst_q.empty()) { dst_q.resize(fsa->num_states, -1); }  // initialize
                    nq = dst_q[cur_q];                                         // read from the cache
                    if (nq < 0) {                                              // not computed yet
                        nq = cur_q;                                            // initialize
                        for (auto &point: path) {                              // compute the next LTL state
                            nq = static_cast<int>(m_fsa_->GetNextState(nq, m_label_maps_.at(cur_z)(point[0], point[1])));
                            if (m_fsa_->IsSinkState(nq)) { break; }  // sink state
                        }
                        dst_q[cur_q] = nq;  // write to the cache
                    }
                    if (m_fsa_->IsSinkState(nq)) { continue; }  // sink state
                    nx = path.back()[0];
                    ny = path.back()[1];
                    nz = cur_z;
                    next_env_state->metric = GridToMetric(next_env_state->grid);
                    successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kRoom), connected_room_id});
                    ERL_DEBUG_ASSERT(m_room_maps_.at(cur_z).at<int>(nx, ny) == connected_room_id, "The next state is not in the connected room.");
                }
                return successors;
            }
            case scene_graph::Node::Type::kFloor: {
                int floor_num_up = cur_z + 1;
                int floor_num_down = cur_z - 1;
                successors.reserve(2);
                if (floor_num_up < m_scene_graph_->num_floors) {  // go upstairs
                    auto &floor_up = m_scene_graph_->floors.at(floor_num_up);
                    if (const double &cost = m_up_stairs_cost_maps_.at(cur_z)(cur_x, cur_y);
                        !std::isinf(cost)) {  // no path to go upstairs (should not happen, but just in case
                        auto next_env_state = std::make_shared<EnvironmentState>();
                        next_env_state->grid.resize(4);
                        int &nq = next_env_state->grid[3];
                        auto &dst_q = const_cast<std::vector<int> &>(m_up_stairs_path_q_maps_.at(cur_z)(cur_x, cur_y));
                        if (dst_q.empty()) { dst_q.resize(fsa->num_states, -1); }     // initialize
                        nq = dst_q[cur_q];                                            // read from the cache
                        auto &path = m_up_stairs_path_maps_.at(cur_z)(cur_x, cur_y);  // path to the upstairs portal
                        if (nq < 0) {                                                 // not computed yet
                            nq = cur_q;                                               // initialize
                            bool encounter_sink_state = false;
                            for (auto &point: path) {  // compute the next LTL state
                                nq = static_cast<int>(m_fsa_->GetNextState(nq, m_label_maps_.at(cur_z)(point[0], point[1])));
                                if (m_fsa_->IsSinkState(nq)) {
                                    encounter_sink_state = true;
                                    break;
                                }
                            }
                            if (!encounter_sink_state) {  // one more step to go upstairs
                                nq = static_cast<int>(m_fsa_->GetNextState(
                                    nq,
                                    m_label_maps_.at(floor_num_up)(floor_up->down_stairs_portal.value()[0], floor_up->down_stairs_portal.value()[1])));
                            }
                            dst_q[cur_q] = nq;  // write to the cache
                        }
                        if (!m_fsa_->IsSinkState(nq)) {
                            next_env_state->grid[0] = floor_up->down_stairs_portal.value()[0];
                            next_env_state->grid[1] = floor_up->down_stairs_portal.value()[1];
                            next_env_state->grid[2] = floor_num_up;
                            // nq is already computed or read from the cache
                            next_env_state->metric = GridToMetric(next_env_state->grid);
                            successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kOcc), m_floor_up_action_id_});
                        }
                    }
                }
                if (floor_num_down >= 0) {  // go downstairs
                    auto &floor_down = m_scene_graph_->floors.at(floor_num_down);
                    const double &cost = m_down_stairs_cost_maps_.at(cur_z)(cur_x, cur_y);
                    if (std::isinf(cost)) { return successors; }  // no path to go downstairs (should not happen, but just in case
                    auto next_env_state = std::make_shared<EnvironmentState>();
                    next_env_state->grid.resize(4);
                    int &nq = next_env_state->grid[3];

                    auto &dst_q = const_cast<std::vector<int> &>(m_down_stairs_path_q_maps_.at(cur_z)(cur_x, cur_y));
                    if (dst_q.empty()) { dst_q.resize(fsa->num_states, -1); }       // initialize
                    nq = dst_q[cur_q];                                              // read from the cache
                    auto &path = m_down_stairs_path_maps_.at(cur_z)(cur_x, cur_y);  // path to the downstairs portal
                    if (nq < 0) {                                                   // not computed yet
                        nq = cur_q;
                        bool encounter_sink_state = false;
                        for (auto &point: path) {
                            nq = static_cast<int>(m_fsa_->GetNextState(nq, m_label_maps_.at(cur_z)(point[0], point[1])));
                            if (m_fsa_->IsSinkState(nq)) {
                                encounter_sink_state = true;
                                break;
                            }
                        }
                        if (!encounter_sink_state) {  // one more step to go downstairs
                            nq = static_cast<int>(m_fsa_->GetNextState(
                                nq,
                                m_label_maps_.at(floor_num_down)(floor_down->up_stairs_portal.value()[0], floor_down->up_stairs_portal.value()[1])));
                        }
                        dst_q[cur_q] = nq;  // write to the cache
                    }
                    if (!m_fsa_->IsSinkState(nq)) {
                        next_env_state->grid[0] = floor_down->up_stairs_portal.value()[0];
                        next_env_state->grid[1] = floor_down->up_stairs_portal.value()[1];
                        next_env_state->grid[2] = floor_num_down;
                        // nq is already computed or read from the cache
                        next_env_state->metric = GridToMetric(next_env_state->grid);
                        successors.emplace_back(next_env_state, cost, std::vector{static_cast<int>(scene_graph::Node::Type::kOcc), m_floor_down_action_id_});
                    }
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
    EnvironmentLTLSceneGraph::GenerateLabelMaps() {
        const auto t0 = std::chrono::high_resolution_clock::now();

        // initialize the label maps
        std::unordered_map<int, Eigen::MatrixX<std::bitset<32>>> label_maps = {};
        for (int i = 0; i < m_scene_graph_->num_floors; ++i) { label_maps[i].resize(m_grid_map_info_->Shape(0), m_grid_map_info_->Shape(1)); }

        const auto *setting = reinterpret_cast<Setting *>(m_setting_.get());
        const std::shared_ptr<FiniteStateAutomaton::Setting> fsa = setting->fsa;
        const auto &atomic_propositions = setting->atomic_propositions;
        for (int i = 0; i < m_scene_graph_->num_floors; ++i) {  // each floor
            const int rows = m_grid_map_info_->Shape(0);
            const int cols = m_grid_map_info_->Shape(1);
            const std::size_t num_propositions = fsa->atomic_propositions.size();
            Eigen::MatrixX<std::bitset<32>> &label_map = label_maps[i];
#pragma omp parallel for collapse(2) default(none) shared(rows, cols, num_propositions, label_map, i, atomic_propositions, fsa)
            for (int r = 0; r < rows; ++r) {      // each row of the map
                for (int c = 0; c < cols; ++c) {  // each column of the map
                    auto &bitset = label_map(r, c);
                    for (std::size_t j = 0; j < num_propositions; ++j) {  // each atomic proposition
                        const std::shared_ptr<AtomicProposition> &proposition = atomic_propositions.at(fsa->atomic_propositions[j]);
                        bitset.set(j, EvaluateAtomicProposition(r, c, i, proposition));
                    }
                }
            }
            m_label_maps_[i] = label_map.cast<uint32_t>();
        }

        const auto t1 = std::chrono::high_resolution_clock::now();
        ERL_INFO("GenerateLabelMaps: %f ms", std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    bool
    EnvironmentLTLSceneGraph::EvaluateReachObject(const int x, const int y, const int floor_num, const int uuid, const double reach_distance) {
        auto half_rows = static_cast<int>(reach_distance / m_grid_map_info_->Resolution(0));
        if (half_rows == 0) { half_rows = 1; }
        auto half_cols = static_cast<int>(reach_distance / m_grid_map_info_->Resolution(1));
        if (half_cols == 0) { half_cols = 1; }

        const int object_id = m_scene_graph_->GetNode<scene_graph::Object>(uuid)->id;
        auto &cat_map = m_cat_maps_[floor_num];
        int roi_min_x = x - half_rows;
        int roi_min_y = y - half_cols;
        int roi_max_x = x + half_rows;
        int roi_max_y = y + half_cols;
        if (roi_min_x < 0) { roi_min_x = 0; }
        if (roi_min_y < 0) { roi_min_y = 0; }
        if (roi_max_x >= cat_map.rows) { roi_max_x = cat_map.rows - 1; }
        if (roi_max_y >= cat_map.cols) { roi_max_y = cat_map.cols - 1; }
        for (int r = roi_min_x; r <= roi_max_x; ++r) {
            for (int c = roi_min_y; c <= roi_max_y; ++c) {
                if (cat_map.at<int>(r, c) == object_id) { return true; }
            }
        }
        return false;
    }

    std::vector<std::shared_ptr<EnvironmentState>>
    EnvironmentLTLSceneGraph::ConvertPath(const std::vector<std::array<int, 2>> &path, const int floor_num, int cur_q) const {
        std::vector<std::shared_ptr<EnvironmentState>> next_env_states;
        next_env_states.reserve(path.size() + 1);
        for (auto &point: path) {
            auto next_env_state = std::make_shared<EnvironmentState>();
            next_env_state->grid.resize(4);
            int &nx = next_env_state->grid[0];
            int &ny = next_env_state->grid[1];
            int &nz = next_env_state->grid[2];
            int &nq = next_env_state->grid[3];
            nx = point[0];
            ny = point[1];
            nz = floor_num;
            nq = static_cast<int>(m_fsa_->GetNextState(cur_q, m_label_maps_.at(nz)(nx, ny)));
            cur_q = nq;  // update cur_q
            next_env_state->metric = GridToMetric(next_env_state->grid);
            next_env_states.push_back(next_env_state);
        }
        return next_env_states;
    }

    std::vector<std::shared_ptr<EnvironmentState>>
    EnvironmentLTLSceneGraph::GetPathToFloor(const int xg, const int yg, const int floor_num, int cur_q, const int next_floor_num) const {
        ERL_DEBUG_ASSERT(std::abs(floor_num - next_floor_num) == 1, "floor_num and next_floor_num should differ by 1.");
        std::vector<std::shared_ptr<EnvironmentState>> next_env_states;
        if (floor_num < next_floor_num) {  // go upstairs
            auto &path = m_up_stairs_path_maps_.at(floor_num)(xg, yg);
            next_env_states = ConvertPath(path, floor_num, cur_q);
        } else {  // go downstairs
            auto &path = m_down_stairs_path_maps_.at(floor_num)(xg, yg);
            next_env_states = ConvertPath(path, floor_num, cur_q);
        }
        const auto next_env_state = std::make_shared<EnvironmentState>();
        const auto &floor = m_scene_graph_->floors.at(next_floor_num);
        next_env_state->grid.resize(4);
        int &nx = next_env_state->grid[0];
        int &ny = next_env_state->grid[1];
        int &nz = next_env_state->grid[2];
        int &nq = next_env_state->grid[3];
        if (floor_num < next_floor_num) {  // go upstairs
            nx = floor->down_stairs_portal.value()[0];
            ny = floor->down_stairs_portal.value()[1];
            nz = next_floor_num;
        } else {  // go downstairs
            nx = floor->up_stairs_portal.value()[0];
            ny = floor->up_stairs_portal.value()[1];
            nz = next_floor_num;
        }
        if (!next_env_states.empty()) { cur_q = next_env_states.back()->grid[3]; }
        nq = static_cast<int>(m_fsa_->GetNextState(cur_q, m_label_maps_.at(next_floor_num)(nx, ny)));
        next_env_state->metric = GridToMetric(next_env_state->grid);
        next_env_states.push_back(next_env_state);
        return next_env_states;
    }

}  // namespace erl::env
