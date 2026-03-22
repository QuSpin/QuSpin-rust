// Copyright 2024 Phillip Weinberg
#pragma once

#include <stdexcept>
#include <variant>

// TODO: replace this with std::optional

namespace quspin { namespace detail {

template<typename T>
struct Optional {
    std::variant<T, std::monostate> value;

    Optional() : value(std::monostate()) {}

    Optional(const T &value) : value(value) {}

    Optional(const std::monostate value) : value(value) {}

    Optional(const Optional<T> &result) : value(result.value) {}

    bool has_value() const noexcept { return std::holds_alternative<T>(value); }

    template<typename Callable>
    T get(Callable &&functor = []() { return T(); }) const noexcept {
      if (has_value()) {
        return std::get<T>(value);
      } else {
        return functor();
      }
    }

    T get() const {
      if (has_value()) {
        return std::get<T>(value);
      } else {
        throw std::runtime_error(
            "Optional does not have a value, no value provided");
      }
    }
};

}}  // namespace quspin::detail
