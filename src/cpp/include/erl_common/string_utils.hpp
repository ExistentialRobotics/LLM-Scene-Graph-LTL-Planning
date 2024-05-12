#pragma once

#include <memory>
#include <sstream>
#include <cxxabi.h>
#include <typeinfo>

#ifdef __GNUG__
inline std::string
demangle(const char *name) {
    int status = -4;  // some arbitrary value to eliminate the compiler warning
    std::unique_ptr<char, void (*)(void *)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
    return (status == 0) ? res.get() : name;
}
#else
inline std::string
demangle(const char *name) {
    return name;
}
#endif

template<typename T>
std::string
type_name() {
    return demangle(typeid(T).name());
}

namespace erl::common {

#define ERL_FORMAT_STRING(...)                              \
    ([&]() {                                                \
        char buffer[4096];                                  \
        std::snprintf(buffer, sizeof(buffer), __VA_ARGS__); \
        return std::string(buffer);                         \
    }())

    template<typename... Args>
    std::string
    AsString(Args... args) {
        std::stringstream ss;
        (ss << ... << args);  // https://en.cppreference.com/w/cpp/language/fold
        return ss.str();
    }

#define ERL_AS_STRING(x) #x

    template<typename T>
    std::string
    AsString(const std::vector<T> &vec) {
        if (vec.empty()) { return "vector<" + type_name<T>() + ">(size:0)[]"; }
        std::stringstream ss;
        std::size_t n = vec.size();
        ss << "vector<" << type_name<T>() << ">(size:" << n << ")[" << vec[0];
        for (std::size_t i = 1; i < n; ++i) { ss << ", " << vec[i]; }
        ss << "]";
        return ss.str();
    }
}  // namespace erl::common
