// Copyright 2024 Phillip Weinberg
// Copyright 2024 Phillip Weinberg
#pragma once

#include <quspin/array/detail/array.hpp>
#include <quspin/scalar/detail/reference.hpp>
#include <quspin/scalar/detail/scalar.hpp>

namespace quspin { namespace detail {

template<typename lhs_t, typename rhs_t>
void check(const array<lhs_t> &lhs, const array<rhs_t> &rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("Invalid array sizes");
  }

  if (lhs.ndim() != rhs.ndim()) {
    throw std::runtime_error("Invalid number of dimensions");
  }

  if (lhs.shape() != rhs.shape()) {
    throw std::runtime_error("Invalid array shapes");
  }
}

template<typename lhs_t, typename rhs_t>
void check(const scalar<lhs_t> &, const array<rhs_t> &) {}

template<typename lhs_t, typename rhs_t>
void check(const array<lhs_t> &, const scalar<rhs_t> &) {}

template<typename lhs_t, typename rhs_t>
void check(const scalar<lhs_t> &, const scalar<rhs_t> &) {}

template<typename lhs_t, typename rhs_t>
void check(const reference<lhs_t> &, const array<rhs_t> &) {}

template<typename lhs_t, typename rhs_t>
void check(const array<lhs_t> &, const reference<rhs_t> &) {}

template<typename lhs_t, typename rhs_t>
void check(const reference<lhs_t> &, const reference<rhs_t> &) {}

template<typename lhs_t, typename rhs_t>
void check(const scalar<lhs_t> &, const reference<rhs_t> &) {}

template<typename lhs_t, typename rhs_t>
void check(const reference<lhs_t> &, const scalar<rhs_t> &) {}

}}  // namespace quspin::detail
