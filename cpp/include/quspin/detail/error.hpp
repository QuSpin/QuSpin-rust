// Copyright 2024 Phillip Weinberg
#pragma once

#include <stdexcept>
#include <string>
#include <variant>

namespace quspin { namespace detail {

enum class ErrorType { RuntimeError, ValueError, IndexError };

struct Error {
    ErrorType type;
    std::string e;

    Error(ErrorType type, std::string e) : type(type), e(e) {}

    void throw_error() const {
      switch (type) {
        case ErrorType::RuntimeError:
          throw std::runtime_error(e);
        case ErrorType::ValueError:
          throw std::invalid_argument(e);
        case ErrorType::IndexError:
          throw std::out_of_range(e);
      }
    }
};

template<typename T>
struct ErrorOr {
    std::variant<T, Error> val;

    ErrorOr(const Error &e) : val(e) {}

    ErrorOr(const T &t) : val(t) {}

    bool has_error() const { return std::holds_alternative<Error>(val); }

    T get_value() const { return std::get<T>(val); }

    Error get_error() const { return std::get<Error>(val); }
};

struct ReturnVoidError : ErrorOr<std::monostate> {
    ReturnVoidError() : ErrorOr(std::monostate()) {}

    ReturnVoidError(const Error &err) : ErrorOr(err) {}
};

template<typename T, typename Visitor, typename... Args>
T visit_or_error(Visitor visitor, Args... args) {
  ErrorOr<T> out = std::visit(visitor, args...);

  if (out.has_error()) {
    out.get_error().throw_error();
  }
  return out.get_value();
}

}}  // namespace quspin::detail
