#pragma once

#include <string>
#include <exception>

struct NotImplemented final : std::exception {

    std::string msg;

    explicit NotImplemented(const std::string& not_implemented_name)
        : msg(not_implemented_name + " not implemented") {}

    [[nodiscard]] const char*
    what() const noexcept override {
        return msg.c_str();
    }
};
