// Copyright 2024 Phillip Weinberg
#pragma once

#include <concepts>
#include <vector>

namespace quspin::detail::basis {

template<typename grp_result_t>
concept grp_result_requirements =
    requires(grp_result_t grp_result, const grp_result_t& other) {
      typename grp_result_t::bitset_type;
      { grp_result.check_refstate(other) } -> std::same_as<grp_result_t>;
      { grp_result.get_refstate(other) } -> std::same_as<grp_result_t>;
    };

template<typename grp_element_t, typename grp_result_t>
concept grp_element_requirements =
    requires(const grp_element_t grp_element, const grp_result_t& grp_result) {
      { grp_element.apply(grp_result) } -> std::same_as<grp_result_t>;
    };

template<typename grp_result_t, typename lattice_grp_element_t,
         typename local_grp_element_t>
  requires grp_element_requirements<lattice_grp_element_t, grp_result_t> &&
           grp_element_requirements<local_grp_element_t, grp_result_t> &&
           grp_result_requirements<grp_result_t>
struct grp {
    std::vector<lattice_grp_element_t> lattice_grp;
    std::vector<local_grp_element_t> local_grp;

  public:

    grp(const std::vector<lattice_grp_element_t>& lattice_grp = {},
        const std::vector<local_grp_element_t>& local_grp = {})
        : lattice_grp(lattice_grp), local_grp(local_grp) {}

    grp(const grp& other) = default;
    grp(grp&& other) = default;

    grp& operator=(const grp& other) = default;
    grp& operator=(grp&& other) = default;

    using bitset_type = typename grp_result_t::bitset_type;
    using get_result_t = std::pair<bitset_type, std::complex<double>>;
    using check_result_t = std::pair<bitset_type, double>;

    std::size_t lattice_size() const { return lattice_grp.size(); }

    std::size_t local_size() const { return local_grp.size(); }

    check_result_t check_refstate(
        const typename grp_result_t::bitset_type& input_bits) const {
      const grp_result_t input(input_bits);
      grp_result_t output(input_bits, 0.0);

      // Apply lattice operators only
      for (auto& lattice : lattice_grp) {
        const auto& next = lattice.apply(input);
        output = output.check_refstate(next);
      }
      // Apply local operators then lattice operators
      for (auto& local : local_grp) {
        const auto& next_local = local.apply(input);
        for (auto& lattice : lattice_grp) {
          const auto& next = lattice.apply(next_local);
          output = output.check_refstate(next);
        }
      }
      return std::move(std::make_pair(output.bits, output.coeff.real()));
    }

    get_result_t get_refstate(const typename grp_result_t::bitset_type& input,
                              const std::complex<double>& coeff =
                                  std::complex<double>(1.0, 0.0)) const {
      grp_result_t curr(input, coeff);

      // Apply lattice operators only
      for (const auto& lattice : lattice_grp) {
        const auto& next = lattice.apply(curr);
        curr = curr.get_refstate(next);
      }
      // Apply local operators then lattice operators
      for (const auto& local : local_grp) {
        const auto& next_local = local.apply(curr);
        for (auto& lattice : lattice_grp) {
          const auto& next = lattice.apply(next_local);
          curr = curr.get_refstate(next);
        }
      }
      return std::move(std::make_pair(curr.bits, curr.coeff));
    }
};

}  // namespace quspin::detail::basis
