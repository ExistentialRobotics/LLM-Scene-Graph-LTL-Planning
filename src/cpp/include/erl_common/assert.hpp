#pragma once

#include "color_print.hpp"
#include "string_utils.hpp"

#include <iostream>
#include <mutex>
#include <memory>

static std::mutex g_print_mutex;

#if defined(ERL_ROS_VERSION_1) || defined(ERL_ROS_VERSION_2)
#include <ros/console.h>
#include <ros/assert.h>
#define ERL_FATAL(...)                ROS_FATAL(__VA_ARGS__)
#define ERL_ERROR(...)                ROS_ERROR(__VA_ARGS__)
#define ERL_WARN(...)                 ROS_WARN(__VA_ARGS__)
#define ERL_WARN_ONCE(...)            ROS_WARN_ONCE(__VA_ARGS__)
#define ERL_WARN_COND(condition, ...) ROS_WARN_COND(condition, __VA_ARGS__)
#define ERL_INFO(...)                 ROS_INFO(__VA_ARGS__)
#define ERL_DEBUG(...)                ROS_DEBUG(__VA_ARGS__)
#ifdef ROS_ASSERT_ENABLED
#define ERL_ASSERT(expr) ROS_ASSERT(expr)
#define ERL_ASSERTM(expr, ...) \
    do { ROS_ASSERT_MSG(expr, __VA_ARGS__); } while (false)
#endif
#else
#define ERL_FATAL(...)                                                                      \
    do {                                                                                    \
        g_print_mutex.lock();                                                               \
        std::cout << erl::common::PrintError("[ERROR]: ", __FILE__, ':', __LINE__) << ": "; \
        printf(__VA_ARGS__);                                                                \
        std::cout << std::endl;                                                             \
        g_print_mutex.unlock();                                                             \
        exit(1);                                                                            \
    } while (false)

#define ERL_ERROR(...)                                                                      \
    do {                                                                                    \
        g_print_mutex.lock();                                                               \
        std::cout << erl::common::PrintError("[ERROR]: ", __FILE__, ':', __LINE__) << ": "; \
        printf(__VA_ARGS__);                                                                \
        g_print_mutex.unlock();                                                             \
    } while (false)

#define ERL_WARN(...)                                                                        \
    do {                                                                                     \
        g_print_mutex.lock();                                                                \
        std::cout << erl::common::PrintWarning("[WARN]: ", __FILE__, ':', __LINE__) << ": "; \
        printf(__VA_ARGS__);                                                                 \
        std::cout << std::endl;                                                              \
        g_print_mutex.unlock();                                                              \
    } while (false)

#define ERL_WARN_ONCE(...)          \
    do {                            \
        static bool warned = false; \
        if (!warned) {              \
            warned = true;          \
            ERL_WARN(__VA_ARGS__);  \
        }                           \
    } while (false)

#define ERL_WARN_COND(condition, ...)             \
    do {                                          \
        if (condition) { ERL_WARN(__VA_ARGS__); } \
    } while (false)

#define ERL_INFO(...)                                                                     \
    do {                                                                                  \
        g_print_mutex.lock();                                                             \
        std::cout << erl::common::PrintInfo("[INFO]: ", __FILE__, ':', __LINE__) << ": "; \
        printf(__VA_ARGS__);                                                              \
        std::cout << std::endl << std::flush;                                             \
        g_print_mutex.unlock();                                                           \
    } while (false)

#ifndef NDEBUG
#define ERL_DEBUG(...)                                                                     \
    do {                                                                                   \
        g_print_mutex.lock();                                                              \
        std::cout << erl::common::PrintInfo("[DEBUG]: ", __FILE__, ':', __LINE__) << ": "; \
        printf(__VA_ARGS__);                                                               \
        std::cout << std::endl;                                                            \
        g_print_mutex.unlock();                                                            \
    } while (false)
#define ERL_DEBUG_ASSERT(expr, ...) ERL_ASSERTM(expr, __VA_ARGS__)
#else
#define ERL_DEBUG(...)              ((void) 0)
#define ERL_DEBUG_ASSERT(expr, ...) (void) 0
#endif
#endif

#define ERL_WARN_ONCE_COND(condition, ...) \
    do {                                   \
        static bool warned = false;        \
        if (!warned && (condition)) {      \
            warned = true;                 \
            ERL_WARN(__VA_ARGS__);         \
        }                                  \
    } while (false)

#ifndef ERL_ASSERTM
#define ERL_ASSERTM(expr, ...)                                                                                                                    \
    do {                                                                                                                                          \
        if (!(expr)) {                                                                                                                            \
            std::stringstream _ss;                                                                                                                \
            _ss << erl::common::PrintError("Assertion failed: (", #expr, ") at ", __FILE__, ":", __LINE__, ": ", ERL_FORMAT_STRING(__VA_ARGS__)); \
            g_print_mutex.lock();                                                                                                                 \
            std::cout << std::flush;                                                                                                              \
            g_print_mutex.unlock();                                                                                                               \
            throw std::runtime_error(_ss.str());                                                                                                  \
        }                                                                                                                                         \
    } while (false)
#endif

#ifndef ERL_ASSERT
#define ERL_ASSERT(expr) ERL_ASSERTM(expr, "Assertion %s failed.", #expr)
#endif

#ifndef NDEBUG
#define ERL_DEBUG_ASSERT(expr, ...) ERL_ASSERTM(expr, __VA_ARGS__)
#else
#define ERL_DEBUG_ASSERT(expr, ...) (void) 0
#endif
