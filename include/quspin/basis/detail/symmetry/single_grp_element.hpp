// Copyright 2024 Phillip Weinberg
#pragma once

#include <complex>
#include <concepts>
#include <quspin/basis/detail/types.hpp>

namespace quspin::details::basis {

template<typename bitset_t>
struct grp_result {
    bitset_t bits;
    std::complex<double> coeff;

  public:

    using bitet_type = bitset_t;

    grp_result(const bitset_t& bits) : bits(bits), coeff(1.0) {}

    grp_result(const grp_result<bitset_t>& other) = default;
    grp_result(grp_result<bitset_t>&& other) = default;
    grp_result& operator=(const grp_result<bitset_t>& other) = default;
    grp_result& operator=(grp_result<bitset_t>&& other) = default;
};

template<typename bitset_t>
grp_result<bitset_t> check_refstate(const grp_result<bitset_t>& curr,
                                    const grp_result<bitset_t>& next) {
  const double check_failed = static_cast<double>(next.bits > curr.bits);
  const double check_equal = static_cast<double>(next.bits == curr.bits);
  constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

  const std::complex<double> norm =
      curr.coeff + NaN * check_failed + next.coeff * check_equal;

  return std::move(grp_result<bitset_t>(curr.bits, norm));
}

template<typename bitset_t>
grp_result<bitset_t> get_refstate(const grp_result<bitset_t>& curr,
                                  const grp_result<bitset_t>& next) {
  return std::move(next.bits > curr.bits ? next : curr);
}

template<typename bit_operation_t, typename bitset_t>
concept bit_operation_requirements =
    requires(bit_operation_t bit_operation, bitset_t bitset) {
      { bit_operation.app(bitset) } -> std::same_as<bitset_t>;
    };

template<typename bit_operation_t, typename bitset_t>
  requires BasisPrimativeTypes<bitset_t> &&
           grp_result_requirements<bitset_t, coeff_t>
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

}  // namespace quspin::details::basis
