// Copyright 2024 Phillip Weinberg
#pragma once

#include <boost/container/small_vector.hpp>
#include <concepts>
#include <quspin/basis/detail/bitbasis/benes.hpp>
#include <quspin/basis/detail/bitbasis/dit_manip.hpp>

namespace quspin::detail::basis {

template<typename bitset_t>
class perm_dit_locations  // permutation of dit locations
{
  private:

    benes::tr_benes<bitset_t> benes;

  public:

    perm_dit_locations(const int lhss, const std::vector<int>& perm) {
      // number of bits to store lhss worth of data:
      const dit_integer_t bits = constants::bits[lhss];

      assert(perm.size() <=
             static_cast<unsigned>(bit_info<bitset_t>::bits) / bits);

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

    int lhss_;
    boost::container::small_vector<uint8_t, 32> perm_;
    boost::container::small_vector<uint8_t, 32> inv_perm_;
    boost::container::small_vector<int, 32> locs_;
    dynamic_dit_manip manip_;

  public:

    template<typename Container>
    perm_dynamic_dit_values(const std::size_t lhss, const Container& perm,
                            const Container& locs)
        : lhss_(lhss),
          perm_(perm.size()),
          inv_perm_(perm.size()),
          locs_(locs.size()),
          manip_(lhss) {
      std::copy(perm.begin(), perm.end(), perm_.begin());
      std::copy(locs.begin(), locs.end(), locs_.begin());
      for (std::size_t i = 0; i < lhss; i++) {
        inv_perm_[perm_[i]] = i;
      }
    }

    perm_dynamic_dit_values(const perm_dynamic_dit_values& other) = default;
    perm_dynamic_dit_values& operator=(const perm_dynamic_dit_values& other) =
        default;

    ~perm_dynamic_dit_values() {}

    inline bitset_t app(const bitset_t& s) const {
      bitset_t out = 0;
      for (const auto loc : locs_) {
        const int sub_str = manip_.get_sub_bitstring(s, loc);
        out = manip_.set_sub_bitstring(out, perm_[sub_str], loc);
      }
      return out;
    }

    inline bitset_t inv(const bitset_t& s) const {
      bitset_t out = 0;
      for (const auto loc : locs_) {
        const int sub_str = manip_.get_sub_bitstring(s, loc);
        out = manip_.set_sub_bitstring(out, inv_perm_[sub_str], loc);
      }
      return out;
    }
};

template<typename bitset_t, std::size_t lhss>
  requires(lhss > 1)
class perm_dit_values {
  private:

    std::array<int, lhss> perm;
    std::array<int, lhss> inv_perm;
    std::vector<int> locs;
    dit_manip<lhss> manip;

  public:

    perm_dit_values(const std::array<int, lhss>& perm,
                    const std::vector<int>& locs)
        : perm(perm), locs(locs) {
      for (size_t i = 0; i < lhss; i++) {
        inv_perm[perm[i]] = i;
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
        out = manip.set_sub_bitstring(out, inv_perm[sub_str], loc);
      }
      return out;
    }
};

template<typename bitset_t>
class perm_dit_mask {
  private:

    bitset_t mask;

  public:

    perm_dit_mask(const bitset_t mask) : mask(mask) {}

    perm_dit_mask(const perm_dit_mask& other) = default;
    perm_dit_mask& operator=(const perm_dit_mask& other) = default;

    ~perm_dit_mask() {}

    inline bitset_t app(const bitset_t& s) const { return s ^ mask; }

    inline bitset_t inv(const bitset_t& s) const { return s ^ mask; }
};

}  // namespace quspin::detail::basis
