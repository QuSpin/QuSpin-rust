// Copyright 2024 Phillip Weinberg
#pragma once

#include <cstdint>
#include <quspin/detail/operators.hpp>
#include <quspin/detail/type_concepts.hpp>
#include <variant>

namespace quspin { namespace detail {

template<PrimativeTypes T>
struct typed_object {
  public:

    using value_type = T;
};

template<PrimativeTypes T>
struct dtype : public typed_object<T> {
    dtype() = default;
    dtype(const dtype &) = default;
    dtype &operator=(const dtype &) = default;
    dtype(dtype &&) = default;
    dtype &operator=(dtype &&) = default;

    static T default_value() { return T(); }
};

template<PrimativeTypes... Ts>
dtype<std::common_type_t<Ts...>> result_dtype(dtype<Ts>...) {
  return dtype<std::common_type_t<Ts...>>();
}

template<PrimativeTypes... Ts>
std::common_type<Ts...> object_result_type(typed_object<Ts>...) {
  return std::common_type<Ts...>();
}

template<typename T>
struct value_type {
    using type = T;
};

template<typename T>
using value_type_t = typename value_type<T>::type;

template<typename T>
  requires PrimativeTypes<T>
struct value_type<dtype<T>> {
    using type = T;
};

using dtypes = std::variant<dtype<int8_t>, dtype<uint8_t>, dtype<int16_t>,
                            dtype<uint16_t>, dtype<uint32_t>, dtype<int32_t>,
                            dtype<uint64_t>, dtype<int64_t>, dtype<float>,
                            dtype<double>, dtype<cfloat>, dtype<cdouble>>;

}}  // namespace quspin::detail
