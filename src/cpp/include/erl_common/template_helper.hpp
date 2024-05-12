#pragma once

#include <memory>

/// Check if T is an instantiation of the template `Class`. For example:
/// `is_instantiation<shared_ptr, T>` is true if `T == shared_ptr<U>` where U can be anything.
template<template<typename...> class Class, typename T>
struct is_instantiation : std::false_type {};

template<template<typename...> class Class, typename... Us>
struct is_instantiation<Class, Class<Us...>> : std::true_type {};

/// Check if T is std::shared_ptr<U> where U can be anything
template<typename T>
using IsSharedPtr = is_instantiation<std::shared_ptr, T>;

/// Check if T is std::unique_ptr<U> where U can be anything
template<typename T>
using IsUniquePtr = is_instantiation<std::unique_ptr, T>;

/// Check if T is std::weak_ptr<U> where U can be anything
template<typename T>
using IsWeakPtr = is_instantiation<std::weak_ptr, T>;

/// Check if T is smart pointer (std::shared_ptr, std::unique_ptr, std::weak_ptr)
template<typename T>
using IsSmartPtr = std::disjunction<IsSharedPtr<T>, IsUniquePtr<T>, IsWeakPtr<T>>;

#define ERL_SMART_PTR_TYPEDEFS(T) \
    using Ptr = std::shared_ptr<T>; \
    using ConstPtr = std::shared_ptr<const T>; \
    using WeakPtr = std::weak_ptr<T>; \
    using ConstWeakPtr = std::weak_ptr<const T>; \
    using UniquePtr = std::unique_ptr<T>; \
    using ConstUniquePtr = std::unique_ptr<const T>
