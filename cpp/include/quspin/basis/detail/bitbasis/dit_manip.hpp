// Copyright 2024 Phillip Weinberg
#pragma once

#include <array>
#include <iterator>
#include <quspin/basis/detail/bitbasis/info.hpp>
#include <quspin/basis/detail/types.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace quspin::detail::basis {

namespace constants {

static const unsigned bits[256] = {
    1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};

static const unsigned mask[256] = {
    1,   1,   1,   3,   3,   7,   7,   7,   7,   15,  15,  15,  15,  15,  15,
    15,  15,  31,  31,  31,  31,  31,  31,  31,  31,  31,  31,  31,  31,  31,
    31,  31,  31,  63,  63,  63,  63,  63,  63,  63,  63,  63,  63,  63,  63,
    63,  63,  63,  63,  63,  63,  63,  63,  63,  63,  63,  63,  63,  63,  63,
    63,  63,  63,  63,  63,  127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
    127, 127, 127, 127, 127, 127, 127, 127, 127, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255};

}  // namespace constants

// local degrees of freedom stored in contiguous chunks of bits

class dynamic_dit_manip {
    // tool for manipuating dit integers
    // use this inside of operator classes to
    // extract subspace of integer.
    std::size_t lhss_;
    unsigned mask;
    dit_integer_t bits;

  public:

    dynamic_dit_manip(const int lhss)
        : lhss_(lhss),
          mask(constants::mask[lhss]),
          bits(constants::bits[lhss]) {}

    template<BasisPrimativeTypes I>
    std::vector<dit_integer_t> to_vector(const I content,
                                         const std::size_t length = 0) const {
      const int niter = (length > 0 ? length : bit_info<I>::bits / bits);

      std::vector<dit_integer_t> out(niter);
      for (int i = 0; i < niter; ++i) {
        int shift = i * bits;
        out[i] = integer_cast<dit_integer_t, I>((content >> shift) & I(mask));
      }
      return out;
    }

    template<BasisPrimativeTypes I>
    std::string to_string(const I content, const std::size_t length = 0) const {
      auto dit_vec = to_vector(content, length);
      std::stringstream out;
      for (auto ele : dit_vec) {
        out << (int)ele << " ";
      }
      return out.str();
    }

    template<BasisPrimativeTypes I>
    std::size_t get_sub_bitstring(
        const I content, std::initializer_list<std::size_t> locs) const {
      return get_sub_bitstring(content, locs.begin(), locs.end());
    }

    template<BasisPrimativeTypes I>
    std::size_t get_sub_bitstring(const I content, const std::size_t i) const {
      return integer_cast<std::size_t, I>(
          (content >> static_cast<unsigned>(i * bits)) & I(mask));
    }

    template<BasisPrimativeTypes I, std::bidirectional_iterator Iterator>
      requires std::integral<std::iter_value_t<Iterator>>
    std::size_t get_sub_bitstring(const I content, Iterator locs_begin,
                                  Iterator locs_end) const {
      std::size_t out = 0;
      auto locs = locs_end;
      std::size_t lhss_pow = 1;
      --locs;
      while (locs != locs_begin) {
        out += lhss_pow * get_sub_bitstring(content, *locs);
        lhss_pow *= lhss_;
        --locs;
      }
      out += lhss_pow * get_sub_bitstring(content, *locs);

      return out;
    }

    template<BasisPrimativeTypes I>
    I set_sub_bitstring(const I content, const std::size_t in,
                        const std::size_t i) const {
      const std::size_t shift = i * bits;
      assert(shift < bit_info<I>::bits);
      const I cleared_content = content & ~(I(mask) << shift);
      return cleared_content | (I(in) << shift);
    }

    template<BasisPrimativeTypes I, std::forward_iterator Iterator>
      requires std::integral<std::iter_value_t<Iterator>>
    I set_sub_bitstring(const I content, const std::size_t in,
                        Iterator locs_begin, Iterator locs_end) const {
      I out = content;
      std::size_t in_ = in;
      for (Iterator loc = locs_begin; loc != locs_end; ++loc) {
        out = set_sub_bitstring(out, in_ % lhss_, *loc);
        in_ /= lhss_;
      }

      return out;
    }

    template<BasisPrimativeTypes I>
    I set_sub_bitstring(const I content, const std::size_t in,
                        std::initializer_list<std::size_t> locs) const {
      return set_sub_bitstring(content, in, locs.begin(), locs.end());
    }
};

template<std::size_t lhss = 2>
  requires(lhss > 1)
class dit_manip : public dynamic_dit_manip {
  public:

    explicit dit_manip() : dynamic_dit_manip(lhss) {}
};

}  // end namespace quspin::detail::basis
