// Copyright 2024 Phillip Weinberg
#pragma once

#include <bit>
#include <concepts>
#include <quspin/basis/detail/bitbasis/cast.hpp>
#include <quspin/basis/detail/types.hpp>
#include <valarray>

namespace quspin::details::basis {

template<typename I>
struct bit_info {};

template<BasisPrimativeTypes I>
struct bit_info<I> {
  public:

    using bit_index_type = int;
    using value_type = I;
    static constexpr bit_index_type bits = std::numeric_limits<I>::digits;
    static constexpr bit_index_type ld_bits =
        std::bit_width(static_cast<std::size_t>(bits - 1));
    static constexpr bit_index_type bytes = bits / 8;
};

template<BasisPrimativeTypes I>
struct bit_info<std::valarray<I>> : public bit_info<I> {};

static_assert(bit_info<uint32_t>::bits == 32);
static_assert(bit_info<uint32_t>::ld_bits == 5);
static_assert(bit_info<uint32_t>::bytes == 4);

static_assert(bit_info<uint64_t>::bits == 64);
static_assert(bit_info<uint64_t>::ld_bits == 6);
static_assert(bit_info<uint64_t>::bytes == 8);

#ifdef USE_BOOST

static_assert(bit_info<uint128_t>::bits == 128);
static_assert(bit_info<uint128_t>::ld_bits == 7);
static_assert(bit_info<uint128_t>::bytes == 16);

static_assert(bit_info<uint256_t>::bits == 256);
static_assert(bit_info<uint256_t>::ld_bits == 8);
static_assert(bit_info<uint256_t>::bytes == 32);

static_assert(bit_info<uint512_t>::bits == 512);
static_assert(bit_info<uint512_t>::ld_bits == 9);
static_assert(bit_info<uint512_t>::bytes == 64);

static_assert(bit_info<uint1024_t>::bits == 1024);
static_assert(bit_info<uint1024_t>::ld_bits == 10);
static_assert(bit_info<uint1024_t>::bytes == 128);

static_assert(bit_info<uint2048_t>::bits == 2048);
static_assert(bit_info<uint2048_t>::ld_bits == 11);
static_assert(bit_info<uint2048_t>::bytes == 256);

static_assert(bit_info<uint4096_t>::bits == 4096);
static_assert(bit_info<uint4096_t>::ld_bits == 12);
static_assert(bit_info<uint4096_t>::bytes == 512);

static_assert(bit_info<uint8192_t>::bits == 8192);
static_assert(bit_info<uint8192_t>::ld_bits == 13);
static_assert(bit_info<uint8192_t>::bytes == 1024);

static_assert(bit_info<uint16384_t>::bits == 16384);
static_assert(bit_info<uint16384_t>::ld_bits == 14);
static_assert(bit_info<uint16384_t>::bytes == 2048);

#endif

}  // namespace quspin::details::basis
