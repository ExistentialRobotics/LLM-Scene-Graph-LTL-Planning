#pragma once

#include "environment_base.hpp"

namespace erl::env {

    class EnvironmentMultiResolution : public EnvironmentBase {

    public:
        EnvironmentMultiResolution()
            : EnvironmentBase(nullptr) {}  // just use the interface of EnvironmentBase, no need to use the distance cost function

        [[nodiscard]] virtual std::size_t
        GetNumResolutionLevels() const = 0;

        [[nodiscard]] virtual std::vector<Successor>
        GetSuccessorsAtLevel(const std::shared_ptr<EnvironmentState> &env_state, std::size_t resolution_level) const = 0;

        [[nodiscard]] virtual bool
        InStateSpaceAtLevel(const std::shared_ptr<EnvironmentState> &env_state, std::size_t resolution_level) const = 0;
    };

}  // namespace erl::env
