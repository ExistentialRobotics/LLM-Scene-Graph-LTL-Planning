#pragma once

// Implementation of AMRA*: Anytime Multi-Resolution Multi-HeuristicBase A*

#include <limits>
#include <cstdint>
#include <memory>
#include <boost/heap/d_ary_heap.hpp>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include "erl_common/yaml.hpp"
#include "erl_common/eigen.hpp"
#include "heuristic.hpp"
#include "planning_interface_multi_resolutions.hpp"

using namespace std::chrono_literals;

namespace erl::search_planning::amra_star {

    struct State;

    struct PriorityQueueItem {
        double f_value = std::numeric_limits<double>::infinity();
        std::shared_ptr<State> state = nullptr;

        PriorityQueueItem() = default;

        PriorityQueueItem(double f, std::shared_ptr<State> s)
            : f_value(f),
              state(std::move(s)) {}
    };

    template<typename T>
    struct Greater {
        bool
        operator()(const std::shared_ptr<T>& s1, const std::shared_ptr<T>& s2) const {
            if (std::abs(s1->f_value - s2->f_value) < 1.e-6) {
                // f value is too close, compare g value
                return s1->state->g_value > s2->state->g_value;
            }
            return s1->f_value > s2->f_value;
        }
    };

    using PriorityQueue = boost::heap::
        d_ary_heap<std::shared_ptr<PriorityQueueItem>, boost::heap::mutable_<true>, boost::heap::arity<8>, boost::heap::compare<Greater<PriorityQueueItem>>>;

    struct State {
        uint32_t plan_itr = 0;  // iteration of the plan that generated/updated this state
        std::shared_ptr<env::EnvironmentState> env_state;
        std::vector<uint64_t> iteration_opened;
        std::vector<PriorityQueue::handle_type> open_queue_keys;
        std::vector<uint64_t> iteration_closed;
        double g_value = std::numeric_limits<double>::infinity();
        std::vector<double> h_values;
        std::vector<bool> in_resolution_levels;   // flags to indicate whether the state is in the resolution level
        std::shared_ptr<State> parent = nullptr;  // parent state
        std::vector<int> action_coords = {};      // action coords = (env_action_coords, env_res_level) that generates this state from its parent

        State(
            uint32_t plan_itr_in,
            std::shared_ptr<env::EnvironmentState> env_state_in,
            std::size_t num_resolution_levels,
            std::vector<bool> in_resolution_level_flags,
            std::vector<double> h_vals)
            : plan_itr(plan_itr_in),
              env_state(std::move(env_state_in)),
              iteration_opened(h_vals.size(), 0),
              open_queue_keys(h_vals.size()),
              iteration_closed(num_resolution_levels, 0),  // +1 for the anchor resolution level
              h_values(std::move(h_vals)),
              in_resolution_levels(std::move(in_resolution_level_flags)) {
            ERL_DEBUG_ASSERT(
                this->in_resolution_levels.size() == num_resolution_levels,
                "in_resolution_level_flags.size() == %zu, num_resolution_levels = %zu",
                this->in_resolution_levels.size(),
                num_resolution_levels);
            ERL_DEBUG_ASSERT(this->in_resolution_levels[0], "in_resolution_level_flags[0] != true");
        }

        [[nodiscard]] bool
        InResolutionLevel(std::size_t resolution_level) const {
            return in_resolution_levels[resolution_level];
        }

        [[nodiscard]] inline bool
        InOpened(std::size_t open_set_id, std::size_t close_set_id) const {
            // 1. if state is just moved into OPEN_i, then iteration_opened[i] > 0 and for other heuristics assigned to the same resolution,
            // iteration_opened[j] == 0 and any iteration_closed[j] == 0, i.e. iteration_opened[i] > any iteration_closed[j] of the same resolution.
            // 2. if state is moved into OPEN_i because it has not been in CLOSE_res(i), then iteration_opened[i] > iteration_closed[res(i)]
            return iteration_opened[open_set_id] > iteration_closed[close_set_id];
        }

        [[nodiscard]] inline bool
        InClosed(std::size_t closed_set_id) const {
            // as long as the state is moved into CLOSE_res(i), then iteration_closed[res(i)] > 0, where close_set_id = res(i).
            return iteration_closed[closed_set_id] > 0;
        }

        inline void
        SetOpened(std::size_t open_set_id, uint64_t opened_itr) {
            iteration_opened[open_set_id] = opened_itr;
        }

        inline void
        SetClosed(std::size_t close_set_id, uint64_t closed_itr) {
            iteration_closed[close_set_id] = closed_itr;
        }

        inline void
        RemoveFromClosed(std::size_t close_set_id, const std::vector<std::size_t>& open_set_ids) {
            if (!InClosed(close_set_id)) { return; }
            for (auto open_set_id: open_set_ids) { iteration_opened[open_set_id] = 0; }
            iteration_closed[close_set_id] = 0;
        }

        inline void
        SetParent(std::shared_ptr<State> parent_in, std::vector<int> action_coords_in) {
            parent = std::move(parent_in);
            action_coords = std::move(action_coords_in);
        }

        inline void
        Reset() {
            iteration_opened.resize(iteration_opened.size(), 0);
            iteration_closed.resize(iteration_closed.size(), 0);
            g_value = std::numeric_limits<double>::infinity();
            parent = nullptr;
            action_coords.clear();
        }
    };

    struct Output {
        uint32_t latest_plan_itr = -1;                                        // latest plan iteration
        std::map<uint32_t, int> goal_indices = {};                            // plan_itr -> goal_index
        std::map<uint32_t, Eigen::MatrixXd> paths = {};                       // plan_itr -> path
        std::map<uint32_t, std::list<std::vector<int>>> actions_coords = {};  // plan_itr -> actions_coords
        std::map<uint32_t, double> costs = {};                                // plan_itr -> cost

        // statistics
        uint32_t num_heuristics = 0;
        uint32_t num_resolution_levels = 0;
        uint64_t num_expansions = 0;
        double w1_solve = -1.0;
        double w2_solve = -1.0;
        double search_time = 0.;

        // logging
        std::map<uint32_t, double> w1_values;                                                 // plan_itr -> w1
        std::map<uint32_t, double> w2_values;                                                 // plan_itr -> w2
        std::map<uint32_t, std::map<std::size_t, std::list<Eigen::VectorXd>>> opened_states;  // total_expand_itr -> heuristic_id -> list of states
        std::map<uint32_t, std::map<std::size_t, Eigen::VectorXd>> closed_states;             // total_expand_itr -> action_resolution_level -> list of states
        std::map<uint32_t, std::list<Eigen::VectorXd>> inconsistent_states;                   // total_expand_itr -> list of states

        void
        Save(const std::filesystem::path& file_path) const;
    };

    class AMRAStar {

    public:
        struct Setting : common::Yamlable<Setting> {
            std::chrono::nanoseconds time_limit = 10000000s;
            double w1_init = 10;
            double w2_init = 20;
            double w1_final = 1;
            double w2_final = 1;
            double w1_decay_factor = 0.5;
            double w2_decay_factor = 0.5;
            bool log = false;
        };

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        std::shared_ptr<PlanningInterfaceMultiResolutions> m_planning_interface_ = nullptr;
        uint32_t m_plan_itr_ = 0;
        uint64_t m_total_expand_itr_ = 0;
        Eigen::VectorX<uint64_t> m_expand_itr_;

        double m_w1_ = 0;
        double m_w2_ = 0;

        std::chrono::nanoseconds m_search_time_ = 0ns;

        std::vector<PriorityQueue> m_open_queues_ = {};
        absl::flat_hash_map<std::size_t, std::shared_ptr<State>> m_states_hash_map_ = {};
        absl::flat_hash_set<std::shared_ptr<State>> m_inconsistent_states_ = {};
        std::shared_ptr<State> m_start_state_ = nullptr;
        std::vector<std::shared_ptr<State>> m_goal_states_ = {};
        std::shared_ptr<Output> m_output_ = nullptr;

    public:
        explicit AMRAStar(std::shared_ptr<PlanningInterfaceMultiResolutions> planning_interface, std::shared_ptr<Setting> setting = nullptr);

        std::shared_ptr<Output>
        Plan();

    private:
        std::pair<std::shared_ptr<State>, int>
        ImprovePath(const std::chrono::system_clock::time_point& start_time, std::chrono::nanoseconds& elapsed_time);

        void
        Expand(const std::shared_ptr<State>& parent, std::size_t heuristic_id);

        std::shared_ptr<State>&
        GetState(const std::shared_ptr<env::EnvironmentState>& env_state) {
            return m_states_hash_map_[m_planning_interface_->StateHashing(env_state)];
        }

        void
        ReinitState(const std::shared_ptr<State>& state) const {
            if (state->plan_itr == m_plan_itr_) { return; }
            state->Reset();
            state->plan_itr = m_plan_itr_;
            // recompute h-value
            std::size_t num_heuristics = m_planning_interface_->GetNumHeuristics();
            for (std::size_t heuristic_id = 0; heuristic_id < num_heuristics; ++heuristic_id) {
                state->h_values[heuristic_id] = (*m_planning_interface_->GetHeuristic(heuristic_id))(*state->env_state);
            }
        }

        [[nodiscard]] double
        GetKeyValue(const std::shared_ptr<State>& state, std::size_t heuristic_id) const {
            return state->g_value + m_w1_ * state->h_values[heuristic_id];
        }

        void
        InsertOrUpdate(const std::shared_ptr<State>& state, std::size_t heuristic_id, double f_value) {
            std::size_t resolution_level = m_planning_interface_->GetResolutionAssignment(heuristic_id);
            if (state->InOpened(heuristic_id, resolution_level)) {
                // state is already in open, update its f-value
                (*state->open_queue_keys[heuristic_id])->f_value = f_value;
                m_open_queues_[heuristic_id].increase(state->open_queue_keys[heuristic_id]);
            } else {
                // state is not in open, insert it into open
                state->open_queue_keys[heuristic_id] = m_open_queues_[heuristic_id].push(std::make_shared<PriorityQueueItem>(f_value, state));
                state->SetOpened(heuristic_id, m_total_expand_itr_);
                if (m_setting_->log) { m_output_->opened_states[m_total_expand_itr_][heuristic_id].push_back(state->env_state->metric); }
            }
        }

        void
        RebuildOpenQueue(std::size_t heuristic_id) {
            auto& open_queue = m_open_queues_[heuristic_id];
            PriorityQueue new_open_queue;
            new_open_queue.reserve(20000);
            for (auto& queue_item: open_queue) {
                queue_item->f_value = GetKeyValue(queue_item->state, heuristic_id);
                queue_item->state->open_queue_keys[heuristic_id] = new_open_queue.push(queue_item);
            }
            open_queue.swap(new_open_queue);
        }

        /**
         * @brief Recover the path from the start state to the reached goal state
         * @param goal_info (goal state, goal index)
         */
        void
        RecoverPath(const std::pair<std::shared_ptr<State>, int>& goal_info);

        /**
         * @brief Save the output of the search
         * @param goal_info (goal state, goal index)
         */
        void
        SaveOutput(const std::pair<std::shared_ptr<State>, int>& goal_info);
    };
}  // namespace erl::search_planning::amra_star

namespace YAML {
    template<>
    struct convert<erl::search_planning::amra_star::AMRAStar::Setting> {
        static Node
        encode(const erl::search_planning::amra_star::AMRAStar::Setting& rhs) {
            Node node;
            node["time_limit"] = static_cast<double>(rhs.time_limit.count()) / 1.e9;
            node["w1_init"] = rhs.w1_init;
            node["w2_init"] = rhs.w2_init;
            node["w1_final"] = rhs.w1_final;
            node["w2_final"] = rhs.w2_final;
            node["w1_decay_factor"] = rhs.w1_decay_factor;
            node["w2_decay_factor"] = rhs.w2_decay_factor;
            node["log"] = rhs.log;
            return node;
        }

        static bool
        decode(const Node& node, erl::search_planning::amra_star::AMRAStar::Setting& rhs) {
            rhs.time_limit = std::chrono::nanoseconds(static_cast<std::size_t>(node["time_limit"].as<double>() * 1.e9));
            rhs.w1_init = node["w1_init"].as<double>();
            rhs.w2_init = node["w2_init"].as<double>();
            rhs.w1_final = node["w1_final"].as<double>();
            rhs.w2_final = node["w2_final"].as<double>();
            rhs.w1_decay_factor = node["w1_decay_factor"].as<double>();
            rhs.w2_decay_factor = node["w2_decay_factor"].as<double>();
            rhs.log = node["log"].as<bool>();
            return true;
        }
    };

    inline Emitter&
    operator<<(Emitter& out, const erl::search_planning::amra_star::AMRAStar::Setting& rhs) {
        out << BeginMap;
        out << Key << "time_limit" << Value << static_cast<double>(rhs.time_limit.count()) / 1.e9;
        out << Key << "w1_init" << Value << rhs.w1_init;
        out << Key << "w2_init" << Value << rhs.w2_init;
        out << Key << "w1_final" << Value << rhs.w1_final;
        out << Key << "w2_final" << Value << rhs.w2_final;
        out << Key << "w1_decay_factor" << Value << rhs.w1_decay_factor;
        out << Key << "w2_decay_factor" << Value << rhs.w2_decay_factor;
        out << Key << "log" << Value << rhs.log;
        out << EndMap;
        return out;
    }
}  // namespace YAML
