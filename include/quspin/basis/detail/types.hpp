// Copyright 2024 Phillip Weinberg
#pragma once

#include <bitset>
#include <cinttypes>
#include <concepts>

namespace quspin::detail::basis {

using dit_integer_t = int;

using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;
using uint128_t = std::bitset<128>;
using uint256_t = std::bitset<256>;
using uint512_t = std::bitset<512>;
using uint1024_t = std::bitset<1024>;
using uint2048_t = std::bitset<2048>;
using uint4096_t = std::bitset<4096>;
using uint8192_t = std::bitset<8192>;
using uint16384_t = std::bitset<16384>;

template<typename I>
concept BasisPrimativeTypes =
    std::same_as<I, uint32_t> || std::same_as<I, uint64_t> ||
    std::same_as<I, uint128_t> || std::same_as<I, uint256_t> ||
    std::same_as<I, uint512_t> || std::same_as<I, uint1024_t> ||
    std::same_as<I, uint2048_t> || std::same_as<I, uint4096_t> ||
    std::same_as<I, uint4096_t> || std::same_as<I, uint8192_t> ||
    std::same_as<I, uint16384_t>;

}  // namespace quspin::detail::basis
