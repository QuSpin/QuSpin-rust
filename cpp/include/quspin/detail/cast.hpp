// Copyright 2024 Phillip Weinberg
#pragma once

#include <complex>
#include <quspin/detail/type_concepts.hpp>

namespace quspin { namespace detail {

template<Floating T>
using complex_wrapper = std::complex<T>;

typedef std::complex<float> cfloat;
typedef std::complex<double> cdouble;
typedef std::complex<long double> cldouble;

template<PrimativeTypes T>
struct is_complex : std::false_type {};

template<Floating T>
struct is_complex<std::complex<T>> : std::true_type {};

template<PrimativeTypes T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template<PrimativeTypes... Ts>
using upcast_t = std::common_type_t<Ts...>;

template<PrimativeTypes from, PrimativeTypes to>
struct can_safe_cast : std::false_type {};

template<PrimativeTypes from, PrimativeTypes to>
inline constexpr bool can_safe_cast_v = can_safe_cast<from, to>::value;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

template<PrimativeTypes To, PrimativeTypes From>
inline To cast(const From &value) {
  if constexpr (is_complex_v<From> && !is_complex_v<To>) {
    return static_cast<To>(value.real());
  } else {
    return static_cast<To>(value);
  }
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

template<>
struct can_safe_cast<int8_t, int8_t> : std::true_type {};

template<>
struct can_safe_cast<int8_t, int16_t> : std::true_type {};

template<>
struct can_safe_cast<int8_t, int32_t> : std::true_type {};

template<>
struct can_safe_cast<int8_t, int64_t> : std::true_type {};

template<>
struct can_safe_cast<int8_t, float> : std::true_type {};

template<>
struct can_safe_cast<int8_t, double> : std::true_type {};

template<>
struct can_safe_cast<int8_t, cfloat> : std::true_type {};

template<>
struct can_safe_cast<int8_t, cdouble> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, uint8_t> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, int16_t> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, uint16_t> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, int32_t> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, uint32_t> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, int64_t> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, uint64_t> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, float> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, double> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, cfloat> : std::true_type {};

template<>
struct can_safe_cast<uint8_t, cdouble> : std::true_type {};

template<>
struct can_safe_cast<int16_t, int16_t> : std::true_type {};

template<>
struct can_safe_cast<int16_t, int32_t> : std::true_type {};

template<>
struct can_safe_cast<int16_t, int64_t> : std::true_type {};

template<>
struct can_safe_cast<int16_t, float> : std::true_type {};

template<>
struct can_safe_cast<int16_t, double> : std::true_type {};

template<>
struct can_safe_cast<int16_t, cfloat> : std::true_type {};

template<>
struct can_safe_cast<int16_t, cdouble> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, uint16_t> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, int32_t> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, uint32_t> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, int64_t> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, uint64_t> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, float> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, double> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, cfloat> : std::true_type {};

template<>
struct can_safe_cast<uint16_t, cdouble> : std::true_type {};

template<>
struct can_safe_cast<int32_t, int32_t> : std::true_type {};

template<>
struct can_safe_cast<int32_t, int64_t> : std::true_type {};

template<>
struct can_safe_cast<int32_t, double> : std::true_type {};

template<>
struct can_safe_cast<int32_t, cdouble> : std::true_type {};

template<>
struct can_safe_cast<uint32_t, uint32_t> : std::true_type {};

template<>
struct can_safe_cast<uint32_t, int64_t> : std::true_type {};

template<>
struct can_safe_cast<uint32_t, uint64_t> : std::true_type {};

template<>
struct can_safe_cast<uint32_t, double> : std::true_type {};

template<>
struct can_safe_cast<uint32_t, cdouble> : std::true_type {};

template<>
struct can_safe_cast<int64_t, int64_t> : std::true_type {};

template<>
struct can_safe_cast<int64_t, double> : std::true_type {};

template<>
struct can_safe_cast<int64_t, cdouble> : std::true_type {};

template<>
struct can_safe_cast<uint64_t, uint64_t> : std::true_type {};

template<>
struct can_safe_cast<uint64_t, double> : std::true_type {};

template<>
struct can_safe_cast<uint64_t, cdouble> : std::true_type {};

template<>
struct can_safe_cast<float, float> : std::true_type {};

template<>
struct can_safe_cast<float, double> : std::true_type {};

template<>
struct can_safe_cast<float, cfloat> : std::true_type {};

template<>
struct can_safe_cast<float, cdouble> : std::true_type {};

template<>
struct can_safe_cast<double, double> : std::true_type {};

template<>
struct can_safe_cast<double, cdouble> : std::true_type {};

template<>
struct can_safe_cast<cfloat, cfloat> : std::true_type {};

template<>
struct can_safe_cast<cfloat, cdouble> : std::true_type {};

template<>
struct can_safe_cast<cdouble, cdouble> : std::true_type {};

}}  // namespace quspin::detail
