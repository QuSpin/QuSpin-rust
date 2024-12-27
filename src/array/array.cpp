// Copyright 2024 Phillip Weinberg
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <quspin/array/array.hpp>
#include <quspin/array/detail/array.hpp>
#include <quspin/detail/error.hpp>
#include <quspin/detail/operators.hpp>
#include <quspin/dtype/dtype.hpp>
#include <quspin/scalar/reference.hpp>
#include <quspin/scalar/scalar.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace quspin {

// implementation of Array

template Array::Array(const detail::array<int8_t> &);
template Array::Array(const detail::array<uint8_t> &);
template Array::Array(const detail::array<int16_t> &);
template Array::Array(const detail::array<uint16_t> &);
template Array::Array(const detail::array<uint32_t> &);
template Array::Array(const detail::array<int32_t> &);
template Array::Array(const detail::array<uint64_t> &);
template Array::Array(const detail::array<int64_t> &);
template Array::Array(const detail::array<float> &);
template Array::Array(const detail::array<double> &);
template Array::Array(const detail::array<detail::cfloat> &);
template Array::Array(const detail::array<detail::cdouble> &);

Array::Array(const std::vector<std::size_t> &shape, const DType &dtype)
    : DTypeObject<detail::arrays>(std::visit(
          [&shape](const auto &dtype) {
            using T = typename std::decay_t<decltype(dtype)>::value_type;
            detail::array<T> arr(shape);
            return detail::arrays(arr);
          },
          dtype.get_variant_obj())) {}

Array::Array(const std::vector<std::size_t> &shape,
             const std::vector<std::size_t> &stride, const DType &dtype,
             void *data)
    : DTypeObject<detail::arrays>(std::visit(
          [&shape, &stride, &data](const auto &dtype) {
            using T = typename std::decay_t<decltype(dtype)>::value_type;
            detail::array<T> arr(shape, stride, static_cast<T *>(data));
            return detail::arrays(arr);
          },
          dtype.get_variant_obj())) {}

template Array::Array(std::initializer_list<int8_t>);
template Array::Array(std::initializer_list<uint8_t>);
template Array::Array(std::initializer_list<int16_t>);
template Array::Array(std::initializer_list<uint16_t>);
template Array::Array(std::initializer_list<int32_t>);
template Array::Array(std::initializer_list<uint32_t>);
template Array::Array(std::initializer_list<int64_t>);
template Array::Array(std::initializer_list<uint64_t>);
template Array::Array(std::initializer_list<float>);
template Array::Array(std::initializer_list<double>);
template Array::Array(std::initializer_list<detail::cfloat>);
template Array::Array(std::initializer_list<detail::cdouble>);

bool Array::is_contiguous() const {
  return std::visit(
      [](const auto &internals) { return internals.is_contiguous(); },
      internals_);
}

std::vector<std::size_t> Array::shape() const {
  return std::visit([](const auto &internals) { return internals.shape(); },
                    internals_);
}

std::size_t Array::shape(const std::size_t &dim) const {
  return std::visit(
      [dim](const auto &internals) { return internals.shape(dim); },
      internals_);
}

std::vector<std::size_t> Array::strides() const {
  return std::visit([](const auto &internals) { return internals.strides(); },
                    internals_);
}

std::size_t Array::strides(const std::size_t &dim) const {
  return std::visit(
      [dim](const auto &internals) { return internals.strides(dim); },
      internals_);
}

std::size_t Array::ndim() const {
  return std::visit([](const auto &internals) { return internals.ndim(); },
                    internals_);
}

std::size_t Array::size() const {
  return std::visit([](const auto &internals) { return internals.size(); },
                    internals_);
}

const Scalar Array::operator[](std::vector<std::size_t> &index) const {
  return std::visit(
      [&index](const auto &internals) { return Scalar(internals.at(index)); },
      internals_);
}

Reference Array::operator[](std::vector<std::size_t> &index) {
  return std::visit(
      [&index](auto &&internals) { return Reference(internals.at(index)); },
      internals_);
}

const Scalar Array::operator[](
    std::initializer_list<std::size_t> indices) const {
  return std::visit(
      [&indices](const auto &internals) {
        return Scalar(internals.at(indices));
      },
      internals_);
}

Reference Array::operator[](std::initializer_list<std::size_t> indices) {
  auto index = std::vector<std::size_t>(indices);
  return std::visit(
      [&index](auto &&internals) { return Reference(internals.at(index)); },
      internals_);
}

Array Array::copy() const {
  return std::visit(
      [](const auto &internals) { return Array(internals.copy()); },
      internals_);
}

Array Array::astype(const DType &dtype) const {
  Array out(shape(), dtype);

  std::visit(
      [](auto &&from, auto &&to) -> void {
        using to_t = std::decay_t<decltype(to)>::value_type;

        auto func = [](auto &&value) { return detail::cast<to_t>(value); };
        std::transform(from.begin(), from.end(), to.begin(), func);

        return;
      },
      get_variant_obj(), out.get_variant_obj());

  return out;
}

}  // namespace quspin
