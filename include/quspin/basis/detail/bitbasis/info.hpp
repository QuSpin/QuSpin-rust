// Copyright 2024 Phillip Weinberg
#pragma once

#include <bit>
#include <concepts>
#include <quspin/basis/detail/bitbasis/cast.hpp>
#include <quspin/basis/detail/types.hpp>
#include <valarray>

namespace quspin::detail::basis {

template<typename I>
struct bit_info {};

template<>
struct bit_info<bits32_t> {
  public:

    using bit_index_type = int;
    using value_type = bits32_t;
    static constexpr bit_index_type bits = 32;
    static constexpr bit_index_type ld_bits =
        std::bit_width(static_cast<std::size_t>(bits - 1));
    static constexpr bit_index_type bytes = bits / 8;
};

template<>
struct bit_info<bits64_t> {
  public:

    using bit_index_type = int;
    using value_type = bits64_t;
    static constexpr bit_index_type bits = 64;
    static constexpr bit_index_type ld_bits =
        std::bit_width(static_cast<std::size_t>(bits - 1));
    static constexpr bit_index_type bytes = bits / 8;
};

template<std::size_t num_bits>
struct bit_info<bitset<num_bits>> {
  public:

    using bit_index_type = int;
    using value_type = bitset<num_bits>;
    static constexpr bit_index_type bits = num_bits;
    static constexpr bit_index_type ld_bits =
        std::bit_width(static_cast<std::size_t>(bits - 1));
    static constexpr bit_index_type bytes = bits / 8;
};

template<BasisPrimativeTypes I>
struct bit_info<std::valarray<I>> : public bit_info<I> {};

static_assert(bit_info<bits32_t>::bits == 32);
static_assert(bit_info<bits32_t>::ld_bits == 5);
static_assert(bit_info<bits32_t>::bytes == 4);

static_assert(bit_info<bits64_t>::bits == 64);
static_assert(bit_info<bits64_t>::ld_bits == 6);
static_assert(bit_info<bits64_t>::bytes == 8);

static_assert(bit_info<bits128_t>::bits == 128);
static_assert(bit_info<bits128_t>::ld_bits == 7);
static_assert(bit_info<bits128_t>::bytes == 16);

static_assert(bit_info<bits256_t>::bits == 256);
static_assert(bit_info<bits256_t>::ld_bits == 8);
static_assert(bit_info<bits256_t>::bytes == 32);

static_assert(bit_info<bits512_t>::bits == 512);
static_assert(bit_info<bits512_t>::ld_bits == 9);
static_assert(bit_info<bits512_t>::bytes == 64);

static_assert(bit_info<bits1024_t>::bits == 1024);
static_assert(bit_info<bits1024_t>::ld_bits == 10);
static_assert(bit_info<bits1024_t>::bytes == 128);

static_assert(bit_info<bits2048_t>::bits == 2048);
static_assert(bit_info<bits2048_t>::ld_bits == 11);
static_assert(bit_info<bits2048_t>::bytes == 256);

static_assert(bit_info<bits4096_t>::bits == 4096);
static_assert(bit_info<bits4096_t>::ld_bits == 12);
static_assert(bit_info<bits4096_t>::bytes == 512);

static_assert(bit_info<bits8192_t>::bits == 8192);
static_assert(bit_info<bits8192_t>::ld_bits == 13);
static_assert(bit_info<bits8192_t>::bytes == 1024);

static_assert(bit_info<bits16384_t>::bits == 16384);
static_assert(bit_info<bits16384_t>::ld_bits == 14);
static_assert(bit_info<bits16384_t>::bytes == 2048);

}  // namespace quspin::detail::basis
