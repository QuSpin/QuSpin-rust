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

static const quspin::detail::basis::dit_integer_t bits[256] = {
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
    int lhss_;
    unsigned mask;
    dit_integer_t bits;

  public:

    dynamic_dit_manip(const int lhss)
        : lhss_(lhss),
          mask(constants::mask[lhss]),
          bits(constants::bits[lhss]) {}

    template<BasisPrimativeTypes I>
    std::vector<dit_integer_t> to_vector(const I content,
                                         const int length = 0) const {
      const int niter = (length > 0 ? length : bit_info<I>::bits / bits);

      std::vector<dit_integer_t> out(niter);
      for (int i = 0; i < niter; ++i) {
        int shift = i * bits;
        out[i] = integer_cast<dit_integer_t, I>((content >> shift) & mask);
      }
      return out;
    }

    template<BasisPrimativeTypes I>
    std::string to_string(const I content, const int length = 0) const {
      auto dit_vec = to_vector(content, length);
      std::stringstream out;
      for (auto ele : dit_vec) {
        out << (int)ele << " ";
      }
      return out.str();
    }

    template<BasisPrimativeTypes I, std::integral J>
    J get_sub_bitstring(const I content, const J i) const {
      return integer_cast<J, I>((content >> (i * bits)) & mask);
    }

    template<BasisPrimativeTypes I, std::bidirectional_iterator Iterator>
      requires std::integral<std::iter_value_t<Iterator>>
    decltype(auto) get_sub_bitstring(const I content, Iterator locs_begin,
                                     Iterator locs_end) const {
      using out_t = typename std::iter_value_t<Iterator>;

      out_t out = 0;
      auto locs = locs_end;
      while (locs != locs_begin) {
        --locs;
        out += get_sub_bitstring(content, *locs) % lhss_;
        out *= lhss_;
      }
      out += get_sub_bitstring(content, *locs);

      return out;
    }

    template<BasisPrimativeTypes I, std::integral J>
    I set_sub_bitstring(const I content, const J in, const int i) const {
      const int shift = i * bits;
      if constexpr (std::is_same_v<I, J>) {
        return content ^ (((in << shift) ^ content) & (mask << shift));
      } else {
        return content ^ (((I(in) << shift) ^ content) & (mask << shift));
      }
    }

    template<BasisPrimativeTypes I, std::forward_iterator Iterator>
      requires std::integral<std::iter_value_t<Iterator>>
    I set_sub_bitstring(const I content, const int in, Iterator locs_begin,
                        Iterator locs_end) const {
      I out = content;
      I in_ = I(in);
      for (Iterator loc = locs_begin; loc != locs_end; ++loc) {
        out = set_sub_bitstring(out, in_, *loc);
        in_ /= lhss_;
      }

      return out;
    }
};

template<std::size_t lhss = 2>
  requires(lhss > 1)
class dit_manip : public dynamic_dit_manip {
  public:

    explicit dit_manip() : dynamic_dit_manip(lhss) {}
};

}  // end namespace quspin::detail::basis
