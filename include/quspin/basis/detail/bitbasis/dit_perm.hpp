// Copyright 2024 Phillip Weinberg
#pragma once

#include <array>
#include <concepts>
#include <quspin/basis/detail/bitbasis/benes.hpp>
#include <quspin/basis/detail/bitbasis/dit_manip.hpp>
#include <quspin/detail/default_containers.hpp>

namespace quspin::detail::basis {

template<typename bitset_t>
  requires BasisPrimativeTypes<bitset_t>
class perm_dit_locations  // permutation of dit locations
{
  private:

    benes::tr_benes<bitset_t> benes;

  public:

    template<std::integral IntType>
    perm_dit_locations(const std::size_t lhss,
                       const std::vector<IntType>& perm) {
      // number of bits to store lhss worth of data:
      const std::size_t bits = constants::bits[lhss];

      assert(perm.size() <=
             static_cast<unsigned>(bit_info<bitset_t>::bits) / bits);

      benes::ta_index<bitset_t> index;
      for (std::size_t i = 0; i < bit_info<bitset_t>::bits; i++) {
        index[i] = benes::no_index;
      }

      // permute chucks of bits of length 'bits'
      for (std::size_t i = 0; i < perm.size(); i++) {
        const int dst = perm[i];
        const int src = i;
        for (std::size_t j = 0; j < bits; j++) {
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
  requires BasisPrimativeTypes<bitset_t>
class dynamic_perm_dit_values {
  private:

    std::size_t lhss_;
    svector_t<uint8_t, 6> perm_;
    svector_t<std::size_t, 64> locs_;

  public:

    template<typename Container>
    dynamic_perm_dit_values(const std::size_t lhss, const Container& perm,
                            const Container& locs)
        : lhss_(lhss),
          perm_(perm.begin(), perm.end()),
          locs_(locs.begin(), perm.end()) {}

    dynamic_perm_dit_values(const dynamic_perm_dit_values& other) = default;
    dynamic_perm_dit_values& operator=(const dynamic_perm_dit_values& other) =
        default;

    ~dynamic_perm_dit_values() {}

    inline bitset_t app(const bitset_t& s) const {
      dynamic_dit_manip manip(lhss_);
      bitset_t out = 0;
      for (const auto loc : locs_) {
        const int sub_str = manip.get_sub_bitstring(s, loc);
        out = manip.set_sub_bitstring(out, perm_[sub_str], loc);
      }
      return out;
    }
};

template<typename bitset_t, std::size_t lhss>
  requires(lhss > 2) && BasisPrimativeTypes<bitset_t>
class perm_dit_values {
  private:

    std::array<uint8_t, lhss> perm;
    std::array<std::size_t, bit_info<bitset_t>::bits> locs;
    const std::size_t num_locs;

  public:

    template<typename Container>
    perm_dit_values(const Container& perm, const Container& locs)
        : num_locs(std::distance(locs.begin(), locs.end())) {
      assert(perm.size() == lhss);
      assert(num_locs <= bit_info<bitset_t>::bits);
      std::copy(perm.begin(), perm.end(), this->perm.begin());
      std::copy(locs.begin(), locs.end(), this->locs.begin());
    }

    perm_dit_values(const perm_dit_values& other) = default;
    perm_dit_values& operator=(const perm_dit_values& other) = default;

    ~perm_dit_values() {}

    inline bitset_t app(const bitset_t& s) const {
      dit_manip<lhss> manip;

      bitset_t out = s;
      for (std::size_t i = 0; i < num_locs; i++) {
        const std::size_t loc = locs[i];
        const std::size_t sub_str = manip.get_sub_bitstring(s, loc);
        out = manip.set_sub_bitstring(out, perm[sub_str], loc);
      }
      return out;
    }
};

template<typename bitset_t>
  requires BasisPrimativeTypes<bitset_t>
class perm_dit_mask {
  private:

    bitset_t mask;

  public:

    perm_dit_mask(const bitset_t mask) : mask(mask) {}

    perm_dit_mask(const perm_dit_mask& other) = default;
    perm_dit_mask& operator=(const perm_dit_mask& other) = default;

    ~perm_dit_mask() {}

    inline bitset_t app(const bitset_t& s) const { return s ^ mask; }
};

template<typename bitset_t>
  requires BasisPrimativeTypes<bitset_t>
class dynamic_higher_spin_inv {
  private:

    std::vector<int> locs_;
    std::size_t lhss_;

  public:

    dynamic_higher_spin_inv(const std::vector<int>& locs,
                            const std::size_t lhss)
        : locs_(locs), lhss_(lhss) {}

    dynamic_higher_spin_inv(const dynamic_higher_spin_inv& other) = default;
    dynamic_higher_spin_inv& operator=(const dynamic_higher_spin_inv& other) =
        default;

    ~dynamic_higher_spin_inv() {}

    inline bitset_t app(const bitset_t& s) const {
      dynamic_dit_manip manip(lhss_);
      bitset_t out = 0;
      for (const auto loc : locs_) {
        const auto local_spin = manip.get_sub_bitstring(s, loc);
        out = manip.set_sub_bitstring(out, lhss_ - local_spin - 1, loc);
      }
      return out;
    }
};

template<typename bitset_t, std::size_t lhss>
  requires(lhss > 2) && BasisPrimativeTypes<bitset_t>
class higher_spin_inv {
  private:

    std::array<std::size_t, bit_info<bitset_t>::bits> locs_;
    const std::size_t num_locs;

  public:

    higher_spin_inv(const std::vector<int>& locs)
        : num_locs(std::distance(locs.begin(), locs.end())) {
      assert(num_locs <= bit_info<bitset_t>::bits);
      std::copy(locs.begin(), locs.end(), this->locs_.begin());
    }

    higher_spin_inv(const higher_spin_inv& other) = default;
    higher_spin_inv& operator=(const higher_spin_inv& other) = default;

    ~higher_spin_inv() {}

    inline bitset_t app(const bitset_t& s) const {
      dit_manip<lhss> manip(lhss);
      bitset_t out = 0;
      for (std::size_t i = 0; i < num_locs; i++) {
        const auto loc = locs_[i];
        const auto local_spin = manip.get_sub_bitstring(s, loc);
        out = manip.set_sub_bitstring(out, lhss - local_spin - 1, loc);
      }
      return out;
    }
};

}  // namespace quspin::detail::basis
