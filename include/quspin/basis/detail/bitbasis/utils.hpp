// Copyright 2024 Phillip Weinberg
#pragma once

#include <quspin/basis/detail/bitbasis/info.hpp>

namespace quspin::detail::basis {

template<class T>
typename bit_info<T>::bit_index_type bit_pos(
    T x, typename bit_info<T>::bit_index_type *idx) {
  typename bit_info<T>::bit_index_type *idx0 = idx;
  typename bit_info<T>::bit_index_type n = 0;
  typename bit_info<T>::bit_index_type p = 0;
  do {
    if (x & 1) *(idx++) = n;
    n++;
  } while (x >>= 1);
  return (typename bit_info<T>::bit_index_type)(idx - idx0);
}

template<class T>
int inline bit_count(T v, const int l) {
  // v = v & (((~(T)0) >> 1) >> (bit_info<T>::bits - 1 - l));
  const T m = (~T(0));
  v = v & (m >> (bit_info<T>::bits - l));
  v = v - ((v >> 1) & (T) ~(T)0 / 3);                              // temp
  v = (v & (T) ~(T)0 / 15 * 3) + ((v >> 2) & (T) ~(T)0 / 15 * 3);  // temp
  v = (v + (v >> 4)) & (T) ~(T)0 / 255 * 15;                       // temp
  T res =
      (T)(v * ((T) ~(T)0 / 255)) >> ((bit_info<T>::bytes - 1) * 8);  // count
  return (int)res;
}

}  // namespace quspin::detail::basis

#ifdef QUSPIN_UNIT_TESTS

namespace quspin::detail::basis {

template int bit_pos<uint8_t>(uint8_t, int *);
template int bit_count<uint8_t>(uint8_t, const int);

}  // namespace quspin::detail::basis

TEST_SUITE("quspin/basis/bitbasis/utils.h") {
  using namespace quspin::detail::basis;

  TEST_CASE("bit_pos") {
    uint8_t val = 0b10110010;
    int pos[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    int npos = bit_pos<uint8_t>(val, pos);

    CHECK(npos == 4);
    CHECK(pos[0] == 1);
    CHECK(pos[1] == 4);
    CHECK(pos[2] == 5);
    CHECK(pos[3] == 7);
    CHECK(pos[4] == 0);
    CHECK(pos[5] == 0);
    CHECK(pos[6] == 0);
    CHECK(pos[7] == 0);
  }

  TEST_CASE("bit_count") {
    using namespace quspin::detail::basis;
    uint8_t val = 0b10110010;

    CHECK(bit_count<uint8_t>(val, 0) == 0);
    CHECK(bit_count<uint8_t>(val, 1) == 0);
    CHECK(bit_count<uint8_t>(val, 2) == 1);
    CHECK(bit_count<uint8_t>(val, 3) == 1);
    CHECK(bit_count<uint8_t>(val, 4) == 1);
    CHECK(bit_count<uint8_t>(val, 5) == 2);
    CHECK(bit_count<uint8_t>(val, 6) == 3);
    CHECK(bit_count<uint8_t>(val, 7) == 3);
    CHECK(bit_count<uint8_t>(val, 8) == 4);
  }
}

#endif
