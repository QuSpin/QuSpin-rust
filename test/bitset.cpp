
#include <array>
#include <cassert>
#include <concepts>
#include <iostream>
#include <quspin/basis/detail/types.hpp>

int main() {
  using namespace quspin::detail::basis;
  using bitset_t = bitset<64, unsigned char>;

  assert((bitset_t(1230) << 36) == bitset_t(1230ull << 36));
  assert((bitset_t(2346020) << 23) == bitset_t(2346020ull << 23));
  assert((bitset_t(1232132) << 42) == bitset_t(1232132ull << 42));
  assert((bitset_t(1230) >> 2) == bitset_t(1230ull >> 2));
  assert((bitset_t(1230) & bitset_t(123)) == bitset_t(1230ull & 123ull));
  assert((bitset_t(1230) | bitset_t(123)) == bitset_t(1230ull | 123ull));
  assert((bitset_t(1230) ^ bitset_t(123)) == bitset_t(1230ull ^ 123ull));
  assert((~bitset_t(0)) == bitset_t(~uint64_t(0)));

  return 0;
}
