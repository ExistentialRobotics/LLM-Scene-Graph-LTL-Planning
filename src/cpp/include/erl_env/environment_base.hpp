#pragma once

#include "erl_common/eigen.hpp"
#include "erl_common/assert.hpp"
#include "erl_common/exception.hpp"
#include "erl_common/grid_map.hpp"
#include "environment_state.hpp"
#include "cost.hpp"

#include <opencv2/core.hpp>

#include <functional>
#include <memory>
#include <vector>
#include <list>
#include <map>

namespace erl::env {

    struct Successor {
        std::shared_ptr<EnvironmentState> env_state = nullptr;
        double cost = 0.0;
        std::vector<int> action_coords = {};

        Successor() = default;

        Successor(std::shared_ptr<EnvironmentState> state, double cost, std::vector<int> action_coords)
            : env_state(std::move(state)),
              cost(cost),
              action_coords(std::move(action_coords)) {
            ERL_ASSERTM(env_state != nullptr, "state is nullptr");
        }

        Successor(Eigen::VectorXd env_metric_state, Eigen::VectorXi env_grid_state, double cost, std::vector<int> action_coords)
            : env_state(std::make_shared<EnvironmentState>(std::move(env_metric_state), std::move(env_grid_state))),
              cost(cost),
              action_coords(std::move(action_coords)) {}
    };

    /**
     * @brief EnvironmentBase is a virtual interface for search-based planning on a metric space. Thus, the
     * EnvironmentBase includes a set of map parameters such as grid cell resolution, and a collision checker. The
     * internal state representation is discrete grid coordinate.
     */
    class EnvironmentBase {

    protected:
        std::shared_ptr<CostBase> m_distance_cost_func_;
        double m_time_step_ = 0.01;  // 10ms

    public:
        explicit EnvironmentBase(std::shared_ptr<CostBase> distance_cost_func = nullptr, double time_step = 0.01)
            : m_distance_cost_func_(std::move(distance_cost_func)),
              m_time_step_(time_step) {
            if (m_distance_cost_func_ == nullptr) {
                ERL_INFO("distance_cost_func is nullptr, use Euclidean distance as default cost function.");
                m_distance_cost_func_ = std::make_shared<EuclideanDistanceCost>();
            }
        }

        virtual ~EnvironmentBase() = default;

        [[nodiscard]] std::shared_ptr<CostBase>
        GetDistanceCostFunc() const {
            return m_distance_cost_func_;
        }

        [[nodiscard]] double
        GetTimeStep() const {
            return m_time_step_;
        }

        [[nodiscard]] virtual std::size_t
        GetStateSpaceSize() const = 0;

        [[nodiscard]] virtual std::size_t
        GetActionSpaceSize() const = 0;

        /**
         * Apply an action on the given environment state to get the next state. No collision check guarantee!
         * @param env_state
         * @param action_coords
         * @return
         */
        [[nodiscard]] virtual std::vector<std::shared_ptr<EnvironmentState>>
        ForwardAction(const std::shared_ptr<const EnvironmentState> &env_state, const std::vector<int> &action_coords) const = 0;

        /**
         * Get reachable next environment states with the current state. Collision check is applied.
         * @param env_state the current environment state
         * @return vector of reachable next environment states
         */
        [[nodiscard]] virtual std::vector<Successor>
        GetSuccessors(const std::shared_ptr<EnvironmentState> &env_state) const = 0;

        [[nodiscard]] virtual bool
        InStateSpace(const std::shared_ptr<EnvironmentState> &env_state) const = 0;

        [[nodiscard]] virtual uint32_t
        StateHashing(const std::shared_ptr<env::EnvironmentState> &env_state) const = 0;

        [[nodiscard]] virtual Eigen::VectorXi
        MetricToGrid(const Eigen::Ref<const Eigen::VectorXd> &metric_state) const = 0;

        [[nodiscard]] virtual Eigen::VectorXd
        GridToMetric(const Eigen::Ref<const Eigen::VectorXi> &grid_state) const = 0;

        [[nodiscard]] virtual cv::Mat
        ShowPaths(const std::map<int, Eigen::MatrixXd> & paths, bool block) const = 0;

    protected:
        static void
        InitializeGridMap2D(const std::shared_ptr<common::GridMapUnsigned2D> &grid_map, cv::Mat &initialized_grid_map) {
            // x to the bottom, y to the right, along y first
            initialized_grid_map = cv::Mat(grid_map->info->Shape(0), grid_map->info->Shape(1), CV_8UC1, cv::Scalar(0));
            int size = grid_map->info->Size();
            auto begin = grid_map->data.GetDataPtr();
            auto end = begin + size;
            std::copy(begin, end, initialized_grid_map.data);  // both erl::common::GridMapUnsigned2D and cv::Mat are row-major.
        }

        static void
        InflateGridMap2D(
            const cv::Mat &original_grid_map,
            cv::Mat &inflated_grid_map,
            const std::shared_ptr<common::GridMapInfo2D> &grid_map_info,
            const Eigen::Ref<const Eigen::Matrix2Xd> &shape_metric_vertices);
    };

}  // namespace erl::env
