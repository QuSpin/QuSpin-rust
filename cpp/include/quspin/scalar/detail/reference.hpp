// Copyright 2024 Phillip Weinberg
#pragma once

#include <quspin/detail/operators.hpp>
#include <quspin/dtype/detail/dtype.hpp>
#include <variant>

namespace quspin { namespace detail {

template<typename T>
class reference : public typed_object<T> {
    T *ref_;

  public:

    reference() : ref_(nullptr) {}

    reference(T &ref) : ref_(&ref) {}

    operator T &() const { return *ref_; }

    T &operator=(const T &value) {
      *ref_ = value;
      return *ref_;
    }

    const T &get() const { return *ref_; }
};

template<typename T>
struct value_type<reference<T>> {
    using type = T;
};

using references =
    std::variant<reference<int8_t>, reference<uint8_t>, reference<int16_t>,
                 reference<uint16_t>, reference<uint32_t>, reference<int32_t>,
                 reference<uint64_t>, reference<int64_t>, reference<float>,
                 reference<double>, reference<cfloat>, reference<cdouble>>;

}}  // namespace quspin::detail
