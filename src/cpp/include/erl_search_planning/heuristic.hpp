#pragma once

#include "erl_common/assert.hpp"
#include "erl_common/eigen.hpp"
#include "erl_common/csv.hpp"
#include "erl_env/environment_state.hpp"
#include "erl_env/finite_state_automaton.hpp"

#include <boost/heap/d_ary_heap.hpp>

namespace erl::search_planning {

    class HeuristicBase {
    protected:
        Eigen::VectorXd m_goal_;
        Eigen::VectorXd m_goal_tolerance_;
        double m_terminal_cost_ = 0.0;

    public:
        HeuristicBase() = default;

        HeuristicBase(Eigen::VectorXd goal, Eigen::VectorXd goal_tolerance, double terminal_cost = 0.0)
            : m_goal_(std::move(goal)),
              m_goal_tolerance_(std::move(goal_tolerance)),
              m_terminal_cost_(terminal_cost) {
            ERL_ASSERTM(this->m_goal_.size() > 0, "goal dimension is zero.");
            ERL_ASSERTM(this->m_goal_tolerance_.size() == this->m_goal_.size(), "goal tolerance dimension is not equal to goal dimension.");
        }

        virtual ~HeuristicBase() = default;

        virtual double
        operator()(const env::EnvironmentState &state) const = 0;
    };

    template<long Dim>
    struct EuclideanDistanceHeuristic : public HeuristicBase {

        EuclideanDistanceHeuristic(Eigen::VectorXd goal, Eigen::VectorXd goal_tolerance, double terminal_cost = 0.0)
            : HeuristicBase(std::move(goal), std::move(goal_tolerance), terminal_cost) {
            if (Dim != Eigen::Dynamic) { ERL_ASSERTM(m_goal_.size() >= Dim, "goal dimension is fewer than %ld.", Dim); }
        }

        inline double
        operator()(const env::EnvironmentState &state) const override {
            ERL_ASSERTM(state.metric.size() == m_goal_.size(), "state dimension is not equal to goal dimension.");
            long n = Dim == Eigen::Dynamic ? state.metric.size() : Dim;
            double distance = 0.0;
            for (long i = 0; i < n; ++i) {
                double diff = std::abs(state.metric[i] - m_goal_[i]);
                diff = std::max(diff - m_goal_tolerance_[i], 0.0);
                distance += diff * diff;
            }
            distance = std::sqrt(distance) + m_terminal_cost_;
            return distance;
        }
    };

    template<long Dim>
    struct ManhattanDistanceHeuristic : public HeuristicBase {

        ManhattanDistanceHeuristic(Eigen::VectorXd goal, Eigen::VectorXd goal_tolerance, double terminal_cost = 0.0)
            : HeuristicBase(std::move(goal), std::move(goal_tolerance), terminal_cost) {
            if (Dim != Eigen::Dynamic) { ERL_ASSERTM(m_goal_.size() == Dim, "goal dimension is not equal to %ld.", Dim); }
        }

        inline double
        operator()(const env::EnvironmentState &state) const override {
            ERL_ASSERTM(state.metric.size() == m_goal_.size(), "state dimension is not equal to goal dimension.");
            long n = Dim == Eigen::Dynamic ? state.metric.size() : Dim;
            double distance = 0.0;
            for (long i = 0; i < n; ++i) {
                double diff = std::abs(state.metric[i] - m_goal_[i]);
                diff = std::max(diff - m_goal_tolerance_[i], 0.0);
                distance += diff;
            }
            distance += m_terminal_cost_;
            return distance;
        }
    };

    struct DictionaryHeuristic : public HeuristicBase {
        std::unordered_map<long, double> heuristic_dictionary;
        std::function<long(const env::EnvironmentState &)> state_hashing_func;
        bool assert_on_missing = true;

        DictionaryHeuristic(
            const std::string &csv_path,
            std::function<long(const env::EnvironmentState &)> state_hashing_func_in,
            bool assert_on_missing_in = true)
            : state_hashing_func(std::move(state_hashing_func_in)),
              assert_on_missing(assert_on_missing_in) {
            ERL_ASSERTM(!csv_path.empty(), "csv_path is empty.");
            ERL_ASSERTM(std::filesystem::exists(csv_path), "%s does not exist.", csv_path.c_str());
            ERL_ASSERTM(state_hashing_func != nullptr, "state_hashing is nullptr.");
            // Read csv file of two columns. The first column is the state hashing, and the second one is the heuristic value.
            std::vector<std::vector<std::string>> lines = common::LoadCsvFile(csv_path.c_str());
            std::size_t num_lines = lines.size();
            ERL_ASSERTM(num_lines > 0, "No heuristic values are provided.");
            for (std::size_t i = 0; i < num_lines; ++i) {
                auto &line = lines[i];
                ERL_ASSERTM(line.size() == 2, "Each line should be <state hashing>, <heuristic value>.");
                long state_hashing;
                double heuristic_value;
                try {
                    state_hashing = std::stol(line[0]);
                } catch (const std::invalid_argument &e) {
                    ERL_FATAL("%s. Failed to convert %s to a long number.", e.what(), line[0].c_str());
                } catch (const std::out_of_range &e) { ERL_FATAL("%s. The number, %s, is out of the range of signed long.", e.what(), line[0].c_str()); }
                try {
                    heuristic_value = std::stod(line[1]);
                } catch (const std::invalid_argument &e) {
                    ERL_FATAL("%s. Failed to convert %s to a double number.", e.what(), line[1].c_str());
                } catch (const std::out_of_range &e) { ERL_FATAL("%s. The number, %s, is out of the range of double.", e.what(), line[1].c_str()); }
                ERL_ASSERTM(heuristic_dictionary.insert({state_hashing, heuristic_value}).second, "Duplicated state hashing %ld.", state_hashing);
            }
        }

        double
        operator()(const env::EnvironmentState &state) const override {
            long state_hashing = state_hashing_func(state);
            auto it = heuristic_dictionary.find(state_hashing);
            if (it == heuristic_dictionary.end()) {
                if (assert_on_missing) {
                    ERL_FATAL(
                        "The state (metric: %s, grid: %s, hashing: %ld) is not found in the dictionary.",
                        common::EigenToNumPyFmtString(state.metric).c_str(),
                        common::EigenToNumPyFmtString(state.grid).c_str(),
                        state_hashing);
                }
                return std::numeric_limits<double>::infinity();
            } else {
                return it->second;
            }
        }
    };

    struct MultiGoalsHeuristic : HeuristicBase {
        std::vector<std::shared_ptr<HeuristicBase>> goal_heuristics;

        explicit MultiGoalsHeuristic(std::vector<std::shared_ptr<HeuristicBase>> goal_heuristics_in)
            : goal_heuristics(std::move(goal_heuristics_in)) {  // may be empty
            std::size_t num_goals = goal_heuristics.size();
            for (std::size_t i = 0; i < num_goals; ++i) { ERL_ASSERTM(goal_heuristics[i] != nullptr, "goal_heuristics_in[%d] is nullptr.", static_cast<int>(i)); }
        }

        double
        operator()(const env::EnvironmentState &state) const override {
            if (state.grid[0] == env::VirtualStateValue::kGoal) { return 0.0; }  // virtual goal
            double min_h = std::numeric_limits<double>::infinity();
            for (const auto &heuristic: goal_heuristics) {
                const double h = (*heuristic)(state);
                if (h < min_h) { min_h = h; }
            }
            return min_h;
        }
    };
}  // namespace erl::search_planning
