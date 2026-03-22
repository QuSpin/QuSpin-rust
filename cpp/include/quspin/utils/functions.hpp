// Copyright 2024 Phillip Weinberg
#pragma once

#include <complex>
#include <vector>

namespace quspin {

template<size_t base, size_t N>
struct integer_pow {
    enum pow : size_t {
      value = base * static_cast<size_t>(integer_pow<base, N - 1>::value)
    };
};

template<size_t base>
struct integer_pow<base, 0u> {
    enum pow : size_t { value = 1 };
};

static_assert(integer_pow<2, 0>::value == 1);
static_assert(integer_pow<2, 1>::value == 2);
static_assert(integer_pow<2, 2>::value == 4);
static_assert(integer_pow<2, 3>::value == 8);
static_assert(integer_pow<2, 4>::value == 16);

static_assert(integer_pow<3, 0>::value == 1);
static_assert(integer_pow<3, 1>::value == 3);
static_assert(integer_pow<3, 2>::value == 9);
static_assert(integer_pow<3, 3>::value == 27);
static_assert(integer_pow<3, 4>::value == 81);

}  // namespace quspin
