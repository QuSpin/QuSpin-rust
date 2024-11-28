// Copyright 2024 Phillip Weinberg
#pragma once

#include <concepts>
#include <quspin/basis/detail/bitbasis/benes.hpp>
#include <quspin/basis/detail/bitbasis/dit_manip.hpp>

namespace quspin::details::basis {

template<typename bitset_t>
class perm_dit_locations  // permutation of dit locations
{
  private:

    benes::tr_benes<bitset_t> benes;

  public:

    perm_dit_locations(const int lhss, const std::vector<int>& perm) {
      // number of bits to store lhss worth of data:
      const dit_integer_t bits = constants::bits[lhss];

      assert(perm.size() <= bit_info<bitset_t>::bits / bits);

      benes::ta_index<bitset_t> index;
      for (int i = 0; i < bit_info<bitset_t>::bits; i++) {
        index[i] = benes::no_index;
      }

      // permute chucks of bits of length 'bits'
      for (size_t i = 0; i < perm.size(); i++) {
        const int dst = perm[i];
        const int src = i;
        for (int j = 0; j < bits; j++) {
          const int srcbit = bits * src + j;
          const int dstbit = bits * dst + j;
          index[srcbit] = dstbit;
        }
      }

      benes::gen_benes(&benes, index);
    }

    perm_dit_locations(const perm_dit_locations<bitset_t>& other) = default;
    perm_dit_locations& operator=(const perm_dit_locations<bitset_t>& other) =
        default;

    ~perm_dit_locations() {}

    inline bitset_t app(const bitset_t& s) const {
      return benes::benes_fwd(&benes, s);
    }

    inline bitset_t inv(const bitset_t& s) const {
      return benes::benes_bwd(&benes, s);
    }
};

template<typename bitset_t>
class perm_dynamic_dit_values {
  private:

    int lhss;
    std::vector<int> perm;
    std::vector<int> inv;
    std::vector<int> locs;
    dynamic_bit_manip<bitset_t> manip;

  public:

    perm_dit_values(const int lhss, const std::vector<int>& perm,
                    const std::vector<int>& locs)
        : lhss(lhss), perm(perm), inv(perm.size()), locs(locs), manip(lhss) {
      for (size_t i = 0; i < perm.size(); i++) {
        inv[perm[i]] = i;
      }
    }

    perm_dit_values(const perm_dit_values<bitset_t>& other) = default;
    perm_dit_values& operator=(const perm_dit_values<bitset_t>& other) =
        default;

    ~perm_dit_values() {}

    inline bitset_t app(const bitset_t& s) const {
      bitset_t out = 0;
      for (const auto loc : locs) {
        const int sub_str = manip.get_sub_bitstring(s, loc);
        out = manip.set_sub_bitstring(out, perm[sub_str], loc);
      }
      return out;
    }

    inline bitset_t inv(const bitset_t& s) const {
      bitset_t out = 0;
      for (const auto loc : locs) {
        const int sub_str = manip.get_sub_bitstring(s, loc);
        out = manip.set_sub_bitstring(out, inv[sub_str], loc);
      }
      return out;
    }
};

template<typename bitset_t, std::size_t lhss>
  requires(lhss > 1)
class perm_dit_values {
  private:

    std::array<int, lhss> perm;
    std::array<int, lhss> inv;
    std::vector<int> locs;
    dit_manip<lhss> manip;

  public:

    perm_dit_values(const std::array<int, lhss>& perm,
                    const std::vector<int>& locs)
        : perm(perm), locs(locs) {
      for (size_t i = 0; i < lhss; i++) {
        inv[perm[i]] = i;
      }
    }

    perm_dit_values(const perm_dit_values<bitset_t, lhss>& other) = default;
    perm_dit_values& operator=(const perm_dit_values<bitset_t, lhss>& other) =
        default;

    ~perm_dit_values() {}

    inline bitset_t app(const bitset_t& s) const {
      bitset_t out = 0;
      for (const auto loc : locs) {
        const int sub_str = manip.get_sub_bitstring(s, loc);
        out = manip.set_sub_bitstring(out, perm[sub_str], loc);
      }
      return out;
    }

    inline bitset_t inv(const bitset_t& s) const {
      bitset_t out = 0;
      for (const auto loc : locs) {
        const int sub_str = manip.get_sub_bitstring(s, loc);
        out = manip.set_sub_bitstring(out, inv[sub_str], loc);
      }
      return out;
    }
};

template<typename bitset_t>
class perm_dit_values<bitset_t, 2> {
  private:

    bitset_t mask;

  public:

    perm_dit_values(const bitset_t mask) : mask(mask) {}

    perm_dit_values(const perm_dit_values<bitset_t, 2>& other) = default;
    perm_dit_values& operator=(const perm_dit_values<bitset_t, 2>& other) =
        default;

    ~perm_dit_values() {}

    inline bitset_t app(const bitset_t& s) const { return s ^ mask; }

    inline bitset_t inv(const bitset_t& s) const { return s ^ mask; }
};

}  // namespace quspin::details::basis
