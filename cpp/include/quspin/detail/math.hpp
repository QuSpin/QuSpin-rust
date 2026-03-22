// Copyright 2024 Phillip Weinberg
#pragma once

#include <cmath>
#include <complex>
#include <concepts>
#include <quspin/detail/cast.hpp>
#include <quspin/detail/operators.hpp>

namespace quspin { namespace detail {

template<typename T>
T abs(const complex_wrapper<T> &A) {
  return std::hypot(A.real(), A.imag());
}

template<typename T>
  requires std::unsigned_integral<T>
constexpr T abs(const T &A) {
  return A;
}

template<typename T>
T abs(const T &A) {
  return (A >= 0 ? A : -A);
}

template<typename T>
double abs_squared(const complex_wrapper<T> &A) {  // abs(A)^2
  const T real = A.real();
  const T imag = A.imag();
  return real * real + imag * imag;
}

template<typename T>
decltype(auto) abs_squared(const T &A) {
  if constexpr (std::is_integral_v<T>) {
    return cast<double>(A * A);
  } else {
    return A * A;
  }
}

}}  // namespace quspin::detail
