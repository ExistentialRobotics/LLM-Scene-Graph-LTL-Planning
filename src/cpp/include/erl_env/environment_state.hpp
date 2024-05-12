#pragma once

#include "erl_common/eigen.hpp"

namespace erl::env {
    enum VirtualStateValue { kStart = -2, kGoal = -1 };  // used for multi-start and multi-goal search

    struct EnvironmentState {
        Eigen::VectorXd metric = {};
        Eigen::VectorXi grid = {};  // use signed int to allow for virtual states

        EnvironmentState() = default;

        explicit EnvironmentState(Eigen::VectorXd metric_state)
            : metric(std::move(metric_state)) {}

        explicit EnvironmentState(Eigen::VectorXi grid_state)
            : grid(std::move(grid_state)) {}

        EnvironmentState(Eigen::VectorXd metric_state, Eigen::VectorXi grid_state)
            : metric(std::move(metric_state)),
              grid(std::move(grid_state)) {}
    };
}  // namespace erl::env
