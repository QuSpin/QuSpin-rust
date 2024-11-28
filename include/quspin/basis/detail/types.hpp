// Copyright 2024 Phillip Weinberg
#pragma once

#include <cinttypes>
#include <concepts>

#ifdef USE_BOOST
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/numeric/conversion/cast.hpp>
#endif

namespace quspin::details::basis {

using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;

using dit_integer_t = int;

#ifdef USE_BOOST

using uint128_t = boost::multiprecision::uint128_t;
using uint256_t = boost::multiprecision::uint256_t;
using uint512_t = boost::multiprecision::uint512_t;
using uint1024_t = boost::multiprecision::uint1024_t;

using uint2048_t =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<
        2048UL, 2048UL, boost::multiprecision::unsigned_magnitude,
        boost::multiprecision::unchecked, void> >;

using uint4096_t =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<
        4096UL, 4096UL, boost::multiprecision::unsigned_magnitude,
        boost::multiprecision::unchecked, void> >;

using uint8192_t =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<
        8192UL, 8192UL, boost::multiprecision::unsigned_magnitude,
        boost::multiprecision::unchecked, void> >;

using uint16384_t =
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<
        16384UL, 16384UL, boost::multiprecision::unsigned_magnitude,
        boost::multiprecision::unchecked, void> >;

template<typename I>
concept BasisPrimativeTypes =
    std::same_as<I, uint32_t> || std::same_as<I, uint64_t> ||
    std::same_as<I, uint128_t> || std::same_as<I, uint256_t> ||
    std::same_as<I, uint512_t> || std::same_as<I, uint1024_t> ||
    std::same_as<I, uint2048_t> || std::same_as<I, uint4096_t> ||
    std::same_as<I, uint4096_t> || std::same_as<I, uint8192_t> ||
    std::same_as<I, uint16384_t>;

#else

template<typename I>
concept BasisPrimativeTypes =
    std::same_as<I, uint32_t> || std::same_as<I, uint64_t>;

#endif

}  // namespace quspin::details::basis
