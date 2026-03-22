// Copyright 2024 Phillip Weinberg
#pragma once

#include <functional>
#include <quspin/detail/cast.hpp>
#include <quspin/detail/operators.hpp>
#include <quspin/detail/type_concepts.hpp>
#include <quspin/dtype/detail/dtype.hpp>
#include <quspin/dtype/dtype.hpp>
#include <quspin/scalar/detail/scalar.hpp>
#include <variant>

namespace quspin {

class Scalar : public DTypeObject<detail::scalars> {
    using DTypeObject<detail::scalars>::internals_;

    detail::scalars default_value() {
      return detail::scalars(detail::scalar<double>());
    }

  public:

    Scalar() : DTypeObject<detail::scalars>(default_value()) {}

    Scalar(const Scalar &scalar) = default;

    template<PrimativeTypes T>
    Scalar(const T &value)
        : DTypeObject<detail::scalars>(detail::scalars(detail::scalar(value))) {
    }

    template<PrimativeTypes T>
    operator T() const;

    template<PrimativeTypes T>
    Scalar &operator=(const T &value);

    template<typename Op>
    static Scalar binary_op(const Scalar &lhs, const Scalar &rhs, Op &&op);
    template<PrimativeTypes T>
    Scalar operator+(const T &other) const;
    Scalar operator+(const Scalar &other) const;
    template<PrimativeTypes T>
    Scalar operator-(const T &other) const;
    Scalar operator-(const Scalar &other) const;
    template<PrimativeTypes T>
    Scalar operator*(const T &other) const;
    Scalar operator*(const Scalar &other) const;
    template<PrimativeTypes T>
    Scalar operator/(const T &other) const;
    Scalar operator/(const Scalar &other) const;
};

template<class T>
concept ScalarTypes = PrimativeTypes<T> || std::same_as<T, Scalar>;

}  // namespace quspin
