#include "erl_search_planning/planning_interface_multi_resolutions.hpp"

namespace erl::search_planning {
    PlanningInterfaceMultiResolutions::PlanningInterfaceMultiResolutions(
        std::shared_ptr<env::EnvironmentMultiResolution> environment_multi_resolution,
        std::vector<std::pair<std::shared_ptr<HeuristicBase>, std::size_t>> heuristics,
        Eigen::VectorXd metric_start_coords,
        const std::vector<Eigen::VectorXd> &metric_goals_coords,
        std::vector<Eigen::VectorXd> metric_goals_tolerances,
        std::vector<double> terminal_costs)
        : m_environment_multi_resolution_(std::move(environment_multi_resolution)),  // m_envs_(std::move(environments)),
          m_heuristics_(std::move(heuristics)),
          m_init_start_(std::move(metric_start_coords)),
          m_goals_tolerances_(std::move(metric_goals_tolerances)),
          m_terminal_costs_(std::move(terminal_costs)),
          m_multiple_goals_(metric_goals_coords.size() > 1) {

        // check environments
        ERL_ASSERTM(m_environment_multi_resolution_ != nullptr, "environment_anchor is nullptr.");

        // check goals and goals tolerances
        std::size_t num_goals = metric_goals_coords.size();
        ERL_ASSERTM(num_goals > 0, "at least one goal must be provided.");
        if (m_multiple_goals_) {
            if (m_goals_tolerances_.size() == 1) {
                ERL_INFO("only one goal tolerance is provided, copying it to all goals.");
                m_goals_tolerances_.resize(num_goals, m_goals_tolerances_[0]);
            } else if (m_goals_tolerances_.size() != num_goals) {
                ERL_FATAL("number of goal tolerances must be 1 or equal to the number of goals.");
            }
            if (m_terminal_costs_.size() == 1) {
                ERL_INFO("only one terminal cost is provided, copying it to all goals.");
                m_terminal_costs_.resize(num_goals, m_terminal_costs_[0]);
            } else if (m_terminal_costs_.size() != num_goals) {
                ERL_FATAL("number of terminal costs must be 1 or equal to the number of goals.");
            }
        }

        // check heuristics
        std::size_t num_resolution_levels = m_environment_multi_resolution_->GetNumResolutionLevels();
        std::size_t num_heuristics = m_heuristics_.size();
        m_heuristic_ids_by_resolution_level_.clear();
        m_heuristic_ids_by_resolution_level_.resize(num_resolution_levels);
        for (std::size_t i = 0; i < num_heuristics; ++i) {
            ERL_ASSERTM(m_heuristics_[i].first != nullptr, "heuristics[%d] is nullptr.", static_cast<int>(i));
            ERL_ASSERTM(
                !m_multiple_goals_ || std::dynamic_pointer_cast<MultiGoalsHeuristic>(m_heuristics_[i].first) != nullptr,
                "heuristics[%d] must be derived from MultiGoalsHeuristic when multiple goals are provided.",
                static_cast<int>(i));
            m_heuristic_ids_by_resolution_level_[m_heuristics_[i].second].push_back(i);
        }
        for (std::size_t i = 0; i < num_resolution_levels; ++i) {
            ERL_ASSERTM(!m_heuristic_ids_by_resolution_level_[i].empty(), "resolution level %d has no heuristic.", static_cast<int>(i));
        }
        ERL_ASSERTM(m_heuristics_[0].second == 0, "the first heuristic must be assigned to the anchor level (resolution level 0).");
        ERL_ASSERTM(m_heuristic_ids_by_resolution_level_[0].size() == 1, "the anchor level must have exactly one heuristic.");

        SetStart();
        SetGoals(metric_goals_coords);
    }
}  // namespace erl::search_planning
