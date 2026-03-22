// Copyright 2024 Phillip Weinberg
#pragma once

#include <cstddef>
#include <quspin/array/detail/array.hpp>
#include <quspin/detail/operators.hpp>
#include <quspin/dtype/detail/dtype.hpp>
#include <quspin/dtype/dtype.hpp>
#include <quspin/scalar/detail/scalar.hpp>
#include <quspin/scalar/reference.hpp>
#include <quspin/scalar/scalar.hpp>
#include <string>
#include <vector>

namespace quspin {

class Array : public DTypeObject<detail::arrays> {
    using DTypeObject<detail::arrays>::internals_;

    static detail::arrays default_value() {
      return detail::arrays(detail::array<double>());
    }

  public:

    Array() : DTypeObject<detail::arrays>(default_value()) {}

    Array(const std::vector<std::size_t> &shape,
          const std::vector<std::size_t> &stride, const DType &dtype,
          void *data);
    Array(const std::vector<std::size_t> &shape, const DType &dtype = DType());

    Array(std::initializer_list<std::size_t> shape,
          std::initializer_list<std::size_t> stride, const DType &dtype,
          void *data)
        : Array(std::vector<std::size_t>(shape),
                std::vector<std::size_t>(stride), dtype, data) {}

    Array(std::initializer_list<std::size_t> shape,
          const DType &dtype = DType())
        : Array(std::vector<std::size_t>(shape), dtype) {}

    template<PrimativeTypes T>
    Array(const detail::array<T> &array)
        : DTypeObject<detail::arrays>(detail::arrays(array)) {}

    template<PrimativeTypes T>
    Array(std::initializer_list<T> values)
        : DTypeObject<detail::arrays>(
              detail::arrays(detail::array<T>(values))) {}

    bool is_contiguous() const;
    std::vector<std::size_t> shape() const;
    std::size_t shape(const std::size_t &) const;
    std::vector<std::size_t> strides() const;
    std::size_t strides(const std::size_t &) const;
    std::size_t ndim() const;
    std::size_t size() const;

    const Scalar operator[](std::vector<std::size_t> &index) const;
    Reference operator[](std::vector<std::size_t> &index);
    const Scalar operator[](std::initializer_list<std::size_t> index) const;
    Reference operator[](std::initializer_list<std::size_t> index);
    Array copy() const;
    Array astype(const DType &) const;
};
}  // namespace quspin
