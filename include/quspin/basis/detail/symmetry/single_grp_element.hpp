// Copyright 2024 Phillip Weinberg
#pragma once

#include <complex>
#include <concepts>
#include <iostream>
#include <quspin/basis/detail/types.hpp>

namespace quspin::detail::basis {

template<typename bitset_t>
struct grp_result {
    bitset_t bits;
    std::complex<double> coeff;

  public:

    using bitset_type = bitset_t;

    grp_result(const bitset_t& bits) : bits(bits), coeff(1.0) {}

    grp_result(const bitset_t& bits, const std::complex<double>& coeff)
        : bits(bits), coeff(coeff) {}

    grp_result(const grp_result<bitset_t>& other) = default;
    grp_result(grp_result<bitset_t>&& other) = default;
    grp_result& operator=(const grp_result& other) = default;
    grp_result& operator=(grp_result&& other) = default;

    grp_result check_refstate(const grp_result& next);
    grp_result get_refstate(const grp_result& next);
};

template<typename bitset_t>
grp_result<bitset_t> grp_result<bitset_t>::check_refstate(
    const grp_result<bitset_t>& next) {
  const auto curr = *this;
  const double bits_equal = static_cast<double>(curr.bits == next.bits);
  const std::complex<double> norm = curr.coeff.real() + (bits_equal);
  return grp_result<bitset_t>(next.bits > curr.bits ? next.bits : curr.bits,
                              norm);
}

template<typename bitset_t>
grp_result<bitset_t> grp_result<bitset_t>::get_refstate(
    const grp_result<bitset_t>& next) {
  const auto curr = *this;
  return std::move(next.bits > curr.bits ? next : curr);
}

template<typename bit_operation_t, typename bitset_t>
concept bit_operation_requirements =
    requires(bit_operation_t bit_operation, bitset_t bitset) {
      { bit_operation.app(bitset) } -> std::same_as<bitset_t>;
    };

template<typename bit_operation_t, typename bitset_t>
  requires BasisPrimativeTypes<bitset_t> &&
           bit_operation_requirements<bit_operation_t, bitset_t>
struct grp_element {
    std::complex<double> grp_char;
    bit_operation_t dit_operator;

  public:

    grp_element(const std::complex<double>& grp_char,
                bit_operation_t dit_operator)
        : grp_char(grp_char), dit_operator(dit_operator) {}

    template<typename... Args>
    grp_element(const std::complex<double>& grp_char, Args&&... args)
        : grp_char(grp_char), dit_operator(args...) {}

    grp_element(const grp_element& other) = default;
    grp_element& operator=(const grp_element& other) = default;

    grp_result<bitset_t> apply(const grp_result<bitset_t>& in) const {
      return std::move(
          grp_result<bitset_t>(dit_operator.app(in.bits), in.coeff * grp_char));
    }
};

}  // namespace quspin::detail::basis
