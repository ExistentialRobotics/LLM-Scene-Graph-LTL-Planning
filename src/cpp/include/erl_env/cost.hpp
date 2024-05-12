#pragma once

#include "erl_common/assert.hpp"
#include "erl_common/grid_map.hpp"
#include "environment_state.hpp"

namespace erl::env {

    struct CostBase {
        virtual ~CostBase() = default;

        virtual double
        operator()(const EnvironmentState& state1, const EnvironmentState& state2) const = 0;
    };

    struct EuclideanDistanceCost final : CostBase {
        double
        operator()(const EnvironmentState& state1, const EnvironmentState& state2) const override {
            ERL_ASSERTM(state1.metric.size() == state2.metric.size(), "state dimension is not equal to goal dimension.");
            long n = state1.metric.size();
            double distance = 0.0;
            for (long i = 0; i < n; ++i) {
                double diff = state1.metric[i] - state2.metric[i];
                distance += diff * diff;
            }
            distance = std::sqrt(distance);
            return distance;
        }
    };

    struct Se2Cost : CostBase {
        double w_theta = 0;

        explicit Se2Cost(double w_theta_in = 1.0)
            : w_theta(w_theta_in) {}

        double
        operator()(const EnvironmentState& state1, const EnvironmentState& state2) const override {
            ERL_ASSERTM(state1.metric.size() == 3, "state1 should be [x, y, theta].");
            ERL_ASSERTM(state2.metric.size() == 3, "state2 should be [x, y, theta].");

            double diff_x = state1.metric[0] - state2.metric[0];
            double diff_y = state1.metric[1] - state2.metric[1];
            double diff_theta = std::abs(state1.metric[2] - state2.metric[2]);
            diff_theta = std::min(diff_theta, 2 * M_PI - diff_theta);
            double distance = std::sqrt(diff_x * diff_x + diff_y * diff_y + w_theta * diff_theta * diff_theta);
            return distance;
        }
    };

    struct ManhattanDistanceCost : CostBase {
        double
        operator()(const EnvironmentState& state1, const EnvironmentState& state2) const override {
            ERL_ASSERTM(state1.metric.size() == state2.metric.size(), "state dimension is not equal to goal dimension.");
            long n = state1.metric.size();
            double distance = 0.0;
            for (long i = 0; i < n; ++i) {
                double diff = std::abs(state1.metric[i] - state2.metric[i]);
                distance += diff;
            }
            return distance;
        }
    };

    template<typename DataType, int Dim>
    struct MapCost : CostBase {

        typedef common::GridMap<DataType, Dim> MapType;
        MapType map;

        explicit MapCost(MapType map_in)
            : map(std::move(map_in)) {}

        double
        operator()(const EnvironmentState&, const EnvironmentState& state2) const override {
            return map.data[state2.grid];
        }
    };

    struct CostSum : CostBase {

        std::vector<std::shared_ptr<CostBase>> costs;

        explicit CostSum(std::vector<std::shared_ptr<CostBase>> costs_in)
            : costs(std::move(costs_in)) {}

        double
        operator()(const EnvironmentState& state1, const EnvironmentState& state2) const override {
            return std::accumulate(costs.begin(), costs.end(), 0.0, [&](double acc, const auto& cost_func) { return acc + (*cost_func)(state1, state2); });
        }
    };

}  // namespace erl::env
