#include <chrono>
#include <memory>
#include "erl_search_planning/amra_star.hpp"

namespace erl::search_planning::amra_star {

    void
    Output::Save(const std::filesystem::path &file_path) const {
        std::ofstream ofs(file_path);
        ERL_ASSERTM(ofs.is_open(), "Failed to open file: %s", file_path.string().c_str());
        std::size_t num_successful_plans = paths.size();
        ofs << "AMRA* solution" << std::endl
            << "num_successful_plans: " << num_successful_plans << std::endl
            << "latest_plan_itr: " << latest_plan_itr << std::endl
            << "num_heuristics: " << num_heuristics << std::endl
            << "num_resolution_levels: " << num_resolution_levels << std::endl
            << "num_expansions: " << num_expansions << std::endl
            << "w1_solve: " << w1_solve << std::endl
            << "w2_solve: " << w2_solve << std::endl
            << "search_time: " << search_time << std::endl;

        // save solutions and their cost, actions, etc. for each plan iteration
        long d = 0;
        for (auto &[plan_itr, goal_index]: goal_indices) {
            auto &path = paths.at(plan_itr);
            auto &action_coords = actions_coords.at(plan_itr);
            d = path.rows();
            long n = path.cols();

            ofs << "plan_itr: " << plan_itr << std::endl
                << "w1: " << w1_values.at(plan_itr) << std::endl
                << "w2: " << w2_values.at(plan_itr) << std::endl
                << "goal_index: " << goal_index << std::endl
                << "cost: " << costs.at(plan_itr) << std::endl
                << "num_waypoints: " << n << std::endl
                << "path: " << std::endl;
            ofs << "pos[0]";
            for (int i = 1; i < d; ++i) { ofs << ", pos[" << i << "]"; }
            ofs << std::endl;
            for (int i = 0; i < n; ++i) {
                ofs << path(0, i);
                for (int j = 1; j < d; ++j) { ofs << ", " << path(j, i); }
                ofs << std::endl;
            }
            ofs << "num_actions: " << action_coords.size() << std::endl << "action_coords: " << std::endl;
            ofs << "coord[0]";
            std::size_t m = action_coords.front().size();
            ERL_ASSERTM(m > 0, "action_coord is empty");
            for (std::size_t i = 1; i < m; ++i) { ofs << ", coord[" << i << "]"; }
            ofs << std::endl;
            for (const auto &action_coord: action_coords) {
                ofs << action_coord[0];
                for (std::size_t i = 1; i < m; ++i) { ofs << ", " << action_coord[i]; }
                ofs << std::endl;
            }
        }

        // save opened_states, closed_states, inconsistent_states
        std::size_t cnt = 1;
        for (const auto &[plan_itr, opened_states_at_plan_itr]: opened_states) {
            for (const auto &[heuristic_id, opened_states_at_heuristic_id]: opened_states_at_plan_itr) { cnt += opened_states_at_heuristic_id.size(); }
        }
        ofs << "opened_states: " << std::endl << cnt << std::endl << "expand_itr, heuristic_id, pos[0]";
        for (long i = 1; i < d; ++i) { ofs << ", pos[" << i << "]"; }
        ofs << std::endl;
        for (const auto &[expand_itr, opened_states_at_expand_itr]: opened_states) {
            for (const auto &[heuristic_id, opened_states_at_heuristic_id]: opened_states_at_expand_itr) {
                for (const auto &state: opened_states_at_heuristic_id) {
                    ofs << expand_itr << ", " << heuristic_id << ", " << state[0];
                    for (long i = 1; i < d; ++i) { ofs << ", " << state[i]; }
                    ofs << std::endl;
                }
            }
        }

        cnt = 1;
        for (const auto &[expand_itr, closed_states_at_expand_itr]: closed_states) { cnt += closed_states_at_expand_itr.size(); }
        ofs << "closed_states: " << std::endl << cnt << std::endl << "expand_itr, action_resolution_level, pos[0]";
        for (long i = 1; i < d; ++i) { ofs << ", pos[" << i << "]"; }
        ofs << std::endl;
        for (const auto &[expand_itr, closed_states_at_expand_itr]: closed_states) {
            for (const auto &[action_resolution_level, state]: closed_states_at_expand_itr) {
                ofs << expand_itr << ", " << action_resolution_level << ", " << state[0];
                for (long i = 1; i < d; ++i) { ofs << ", " << state[i]; }
                ofs << std::endl;
            }
        }

        cnt = 1;
        for (const auto &[expand_itr, inconsistent_states_at_expand_itr]: inconsistent_states) { cnt += inconsistent_states_at_expand_itr.size(); }
        ofs << "inconsistent_states: " << std::endl << cnt << std::endl << "expand_itr, pos[0]";
        for (long i = 1; i < d; ++i) { ofs << ", pos[" << i << "]"; }
        ofs << std::endl;
        for (const auto &[expand_itr, inconsistent_states_at_expand_itr]: inconsistent_states) {
            for (const auto &state: inconsistent_states_at_expand_itr) {
                ofs << expand_itr << ", " << state[0];
                for (long i = 1; i < d; ++i) { ofs << ", " << state[i]; }
                ofs << std::endl;
            }
        }
        ofs.close();
    }

    AMRAStar::AMRAStar(std::shared_ptr<PlanningInterfaceMultiResolutions> planning_interface, std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)),
          m_planning_interface_(std::move(planning_interface)),
          m_expand_itr_(m_planning_interface_->GetNumHeuristics()),
          m_open_queues_(m_planning_interface_->GetNumHeuristics()),
          m_output_(std::make_shared<Output>()) {

        if (!m_setting_) { m_setting_ = std::make_shared<Setting>(); }

        std::size_t num_resolution_levels = m_planning_interface_->GetNumResolutionLevels();

        auto start_env_state = m_planning_interface_->GetStartState();
        m_start_state_ = std::make_shared<State>(
            m_plan_itr_,
            start_env_state,
            num_resolution_levels,
            m_planning_interface_->GetInResolutionLevelFlags(start_env_state),
            m_planning_interface_->GetHeuristicValues(start_env_state));
        GetState(start_env_state) = m_start_state_;

        const int num_goals = m_planning_interface_->GetNumGoals();
        m_goal_states_.reserve(num_goals);
        for (int i = 0; i < num_goals; ++i) {
            auto goal_env_state = m_planning_interface_->GetGoalState(i);
            m_goal_states_.push_back(std::make_shared<State>(
                m_plan_itr_,
                goal_env_state,
                num_resolution_levels,
                m_planning_interface_->GetInResolutionLevelFlags(goal_env_state),
                m_planning_interface_->GetHeuristicValues(goal_env_state)));
            GetState(goal_env_state) = m_goal_states_.back();
        }

        for (auto &queue: m_open_queues_) { queue.reserve(20000); }
    }

    std::shared_ptr<Output>
    AMRAStar::Plan() {
        {
            if (int goal_index = m_planning_interface_->ReachGoal(m_start_state_->env_state); goal_index >= 0) {
                SaveOutput({m_start_state_, goal_index});
                return m_output_;
            }
        }

        m_w1_ = m_setting_->w1_init;  // L43
        m_w2_ = m_setting_->w2_init;  // L43

        const std::size_t num_heuristics = m_planning_interface_->GetNumHeuristics();
        const std::size_t num_resolution_levels = m_planning_interface_->GetNumResolutionLevels();
        m_expand_itr_.setZero();
        m_total_expand_itr_ = 1;

        m_plan_itr_++;
        for (auto &goal_state: m_goal_states_) { ReinitState(goal_state); }
        ReinitState(m_start_state_);                                    // L45
        m_start_state_->g_value = 0.;                                   // L44
        for (auto &open_queue: m_open_queues_) { open_queue.clear(); }  // L46-47
        m_inconsistent_states_.clear();
        m_inconsistent_states_.insert(m_start_state_);  // L48

        m_search_time_ = 0ns;
        std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
        while (m_search_time_ < m_setting_->time_limit && (m_w1_ >= m_setting_->w1_final && m_w2_ >= m_setting_->w2_final)) {  // L49
            if (m_setting_->log) {
                m_output_->w1_values[m_plan_itr_] = m_w1_;
                m_output_->w2_values[m_plan_itr_] = m_w2_;
            }
            for (auto &state: m_inconsistent_states_) {  // L50
                state->RemoveFromClosed(0, m_planning_interface_->GetResolutionHeuristicIds(0));
                InsertOrUpdate(state, 0, GetKeyValue(state, 0));  // L51
            }
            m_inconsistent_states_.clear();  // L52
            // update other open queues using the anchor-level open queue
            for (auto &queue_item: m_open_queues_[0]) {  // L53
                std::shared_ptr<State> &state = queue_item->state;
                for (std::size_t heuristic_id = 1; heuristic_id < num_heuristics; ++heuristic_id) {  // L54
                    const std::size_t resolution_level = m_planning_interface_->GetResolutionAssignment(heuristic_id);
                    if (!state->InResolutionLevel(resolution_level)) { continue; }  // L55
                    state->RemoveFromClosed(resolution_level, m_planning_interface_->GetResolutionHeuristicIds(resolution_level));
                    InsertOrUpdate(state, heuristic_id, GetKeyValue(state, heuristic_id));  // L56
                }
            }

            // m_w1_ may be changed, so we need to update the open queues
            for (std::size_t heuristic_id = 0; heuristic_id < num_heuristics; ++heuristic_id) { RebuildOpenQueue(heuristic_id); }

            // empty all close sets
            for (const auto &[hashing, state]: m_states_hash_map_) {
                for (std::size_t res_level = 0; res_level < num_resolution_levels; ++res_level) {                     // L57
                    state->RemoveFromClosed(res_level, m_planning_interface_->GetResolutionHeuristicIds(res_level));  // L58
                }
            }

            // improve path
            std::chrono::nanoseconds elapsed_time;
            auto goal_info = ImprovePath(start_time, elapsed_time);  // L59
            m_search_time_ += elapsed_time;
            start_time = std::chrono::system_clock::now();
            SaveOutput(goal_info);                                                           // L60
            if (goal_info.second < 0 || m_search_time_ > m_setting_->time_limit) { break; }  // fail to find a solution or time out
            ERL_INFO(
                "Solved with (w1, w2) = (%f, %f) | expansions = %s | time = %f sec | cost = %f",
                m_w1_,
                m_w2_,
                common::EigenToNumPyFmtString(m_expand_itr_.transpose()).c_str(),
                m_search_time_.count() / 1e9,
                m_output_->costs[m_plan_itr_]);

            if (m_w1_ == m_setting_->w1_final && m_w2_ == m_setting_->w2_final) { break; }  // L61-62
            m_w1_ = std::max(m_w1_ * m_setting_->w1_decay_factor, m_setting_->w1_final);    // L63
            m_w2_ = std::max(m_w2_ * m_setting_->w2_decay_factor, m_setting_->w2_final);    // L63
            m_plan_itr_++;
        }

        if (int goal_index = m_planning_interface_->ReachGoal(m_start_state_->env_state); goal_index >= 0) {
            SaveOutput({m_start_state_, goal_index});
            return m_output_;
        }
        return m_output_;
    }

    std::pair<std::shared_ptr<State>, int>
    AMRAStar::ImprovePath(const std::chrono::system_clock::time_point &start_time, std::chrono::nanoseconds &elapsed_time) {
        elapsed_time = 0ns;
        const std::size_t num_heuristics = m_planning_interface_->GetNumHeuristics();
        auto &open_anchor = m_open_queues_[0];
        while (!open_anchor.empty() && open_anchor.top()->f_value < std::numeric_limits<double>::infinity()) {  // L25
            elapsed_time = std::chrono::system_clock::now() - start_time;
            if (elapsed_time + m_search_time_ > m_setting_->time_limit) { return {nullptr, -1}; }

            for (std::size_t heuristic_id = 1; heuristic_id < num_heuristics; ++heuristic_id) {  // L26
                if (open_anchor.empty()) { return {nullptr, -1}; }
                const double f_check = m_w2_ * open_anchor.top()->f_value;
                if (auto min_goal_itr = std::min_element(  // get the goal of minimum g_value
                        m_goal_states_.begin(),
                        m_goal_states_.end(),
                        [](const std::shared_ptr<State> &a, const std::shared_ptr<State> &b) { return a->g_value < b->g_value; });
                    f_check >= (*min_goal_itr)->g_value) {
                    return {*min_goal_itr, static_cast<int>(std::distance(m_goal_states_.begin(), min_goal_itr))};
                }

                // state in the open queue of heuristic_id may be invalid when it is moved to CLOSE by another heuristic

                auto &open = m_open_queues_[heuristic_id];
                while (!open.empty() || !open_anchor.empty()) {
                    if (!open.empty() && open.top()->f_value <= f_check) {
                        std::shared_ptr<State> state = open.top()->state;
                        if (int goal_index = m_planning_interface_->ReachGoal(state->env_state); goal_index >= 0) { return {state, goal_index}; }
                        open.pop();
                        if (!state->InOpened(heuristic_id, m_planning_interface_->GetResolutionAssignment(heuristic_id))) {
                            continue;  // state is moved to CLOSED by another heuristic
                        }
                        Expand(state, heuristic_id);
                        m_expand_itr_[static_cast<long>(heuristic_id)]++;
                        m_total_expand_itr_++;
                        break;
                    }
                    std::shared_ptr<State> state = open_anchor.top()->state;
                    if (int goal_index = m_planning_interface_->ReachGoal(state->env_state); goal_index >= 0) { return {state, goal_index}; }
                    open_anchor.pop();
                    Expand(state, 0);
                    m_expand_itr_[0]++;
                    m_total_expand_itr_++;
                    break;
                }
            }
        }
        return {nullptr, -1};
    }

    void
    AMRAStar::Expand(const std::shared_ptr<State> &parent, const std::size_t heuristic_id) {

        std::size_t resolution_level = m_planning_interface_->GetResolutionAssignment(heuristic_id);  // L4
        if (heuristic_id == 0) {
            ERL_DEBUG_ASSERT(!parent->InClosed(0), "parent is already in anchor-level closed set.");  // the heuristic for the anchor level must be consistent
            parent->SetClosed(0, m_total_expand_itr_);                                                // anchor level, consistent heuristic
        } else {                                                                                      // L5
            // comment the following assert because the heuristic may be inconsistent
            // ERL_DEBUG_ASSERT(!parent->InClosed(resolution_level), "parent is already in CLOSED of resolution level %d.", int(resolution_level));
            ERL_DEBUG_ASSERT(parent->InOpened(heuristic_id, resolution_level), "parent is not in OPENED of heuristic %d.", int(heuristic_id));  // L6 to L8
            parent->SetClosed(resolution_level, m_total_expand_itr_);  // parent is also removed from other opened sets assigned to the same resolution level
        }
        if (m_setting_->log) {
            ERL_ASSERTM(
                m_output_->closed_states[m_total_expand_itr_].insert({resolution_level, parent->env_state->metric}).second,
                "state already exists in closed set.");
        }

        ERL_DEBUG_ASSERT(m_planning_interface_->ReachGoal(parent->env_state) < 0, "should not expand from a goal parent.");
        std::vector<env::Successor> successors = m_planning_interface_->GetSuccessors(parent->env_state, resolution_level);
        const std::size_t num_resolution_levels = m_planning_interface_->GetNumResolutionLevels();
        const std::size_t num_heuristics = m_planning_interface_->GetNumHeuristics();
        for (auto &successor: successors) {  // L9
            std::shared_ptr<State> &child = GetState(successor.env_state);
            if (!child) {
                child = std::make_shared<State>(
                    m_plan_itr_,
                    successor.env_state,
                    num_resolution_levels,
                    m_planning_interface_->GetInResolutionLevelFlags(successor.env_state),
                    m_planning_interface_->GetHeuristicValues(successor.env_state));
            }

            if (const double tentative_g_value = parent->g_value + successor.cost; tentative_g_value < child->g_value) {  // L10
                child->g_value = tentative_g_value;                                                                       // L11
                child->SetParent(parent, successor.action_coords);                                                        // L12
                // in anchor-level closed set, inconsistency detected, re-open it
                if (child->InClosed(0)) {                  // L13
                    m_inconsistent_states_.insert(child);  // L14
                    if (m_setting_->log) { m_output_->inconsistent_states[m_total_expand_itr_].push_back(child->env_state->metric); }
                    continue;
                }
                const double f0 = GetKeyValue(child, 0);
                InsertOrUpdate(child, 0, f0);                                                            // L16
                for (std::size_t h_id = 1; h_id < num_heuristics; ++h_id) {                              // L17
                    const std::size_t res_level = m_planning_interface_->GetResolutionAssignment(h_id);  // L18
                    if (!child->InResolutionLevel(res_level)) { continue; }                              // L19-20
                    if (child->InClosed(res_level)) { continue; }                                        // L21
                    const double f_h_id = GetKeyValue(child, h_id);
                    if (f_h_id > m_w2_ * f0) { continue; }  // L22
                    InsertOrUpdate(child, h_id, f_h_id);    // L23
                }
            }
        }
    }

    void
    AMRAStar::RecoverPath(const std::pair<std::shared_ptr<State>, int> &goal_info) {
        m_output_->latest_plan_itr = m_plan_itr_;
        m_output_->w1_solve = m_w1_;
        m_output_->w2_solve = m_w2_;

        ERL_ASSERTM(m_output_->paths.find(m_plan_itr_) == m_output_->paths.end(), "path already exists for plan iteration %d.", m_plan_itr_);
        const std::shared_ptr<State> &goal_state = goal_info.first;
        int goal_index = goal_info.second;
        m_output_->costs[m_plan_itr_] = goal_state->g_value;

        std::shared_ptr<State> node;
        if (m_planning_interface_->IsVirtualGoal(goal_state->env_state)) {  // virtual goal state is used!
            const auto true_goal_env_state = m_planning_interface_->GetPath(goal_state->env_state, goal_state->action_coords)[0];
            goal_index = m_planning_interface_->IsMetricGoal(true_goal_env_state);  // get the true goal index
            node = GetState(true_goal_env_state);
        } else {
            node = goal_state;
        }
        m_output_->goal_indices[m_plan_itr_] = goal_index;

        ERL_DEBUG(
            "Reach goal[%d/%d] (metric: %s, grid: %s) from metric start %s at expansion_itr %lu plan_itr %u.",
            goal_index,
            m_planning_interface_->GetNumGoals(),
            common::EigenToNumPyFmtString(node->env_state->metric.transpose()).c_str(),
            common::EigenToNumPyFmtString(node->env_state->grid.transpose()).c_str(),
            common::EigenToNumPyFmtString(m_planning_interface_->GetStartState()->metric.transpose()).c_str(),
            m_total_expand_itr_,
            m_plan_itr_);

        auto &actions_coords = m_output_->actions_coords[m_plan_itr_];
        actions_coords.clear();
        while (node->parent != nullptr) {
            actions_coords.push_front(node->action_coords);
            node = node->parent;
        }
        std::vector<std::vector<std::shared_ptr<env::EnvironmentState>>> path_segments;
        path_segments.reserve(actions_coords.size());
        long num_path_states = 0;
        auto state = m_start_state_->env_state;
        for (auto &action_coords: actions_coords) {
            ERL_DEBUG("state: %s", common::EigenToNumPyFmtString(state->metric.transpose()).c_str());
            ERL_DEBUG("action_coords: %s", common::AsString(action_coords).c_str());
            std::vector<std::shared_ptr<env::EnvironmentState>> path_segment = m_planning_interface_->GetPath(state, action_coords);
            if (path_segment.empty()) { continue; }
            ERL_DEBUG("arrive: %s", common::EigenToNumPyFmtString(path_segment.back()->metric.transpose()).c_str());
            path_segments.push_back(path_segment);
            num_path_states += static_cast<long>(path_segment.size());
            state = path_segment.back();
        }

        auto &path = m_output_->paths[m_plan_itr_];
        path.resize(state->metric.size(), num_path_states + 1);
        path.col(0) = m_planning_interface_->GetStartState()->metric;
        long index = 1;
        for (auto &path_segment: path_segments) {
            const auto num_states = static_cast<long>(path_segment.size());
            for (long i = 0; i < num_states; ++i) { path.col(index++) = path_segment[i]->metric; }
        }
    }

    void
    AMRAStar::SaveOutput(const std::pair<std::shared_ptr<State>, int> &goal_info) {
        if (goal_info.second >= 0) { RecoverPath(goal_info); }
        m_output_->num_heuristics = m_planning_interface_->GetNumHeuristics();
        m_output_->num_resolution_levels = m_planning_interface_->GetNumResolutionLevels();
        m_output_->num_expansions = m_total_expand_itr_;
        m_output_->search_time = static_cast<double>(m_search_time_.count()) / 1.e9;
    }
}  // namespace erl::search_planning::amra_star
