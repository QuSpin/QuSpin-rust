// Copyright 2024 Phillip Weinberg
#pragma once

#include <concepts>
#include <vector>

namespace quspin::details::basis {

template<typename grp_element_t, typename grp_result_t>
concept grp_requirements =
    requires(grp_element_t grp_element, grp_result_t grp_result) {
      typename grp_result_t::bitset_type;
      { grp_element.apply(grp_result) } -> std::same_as<grp_result_t>;
      { check_refstate(grp_result, grp_result) } -> std::same_as<grp_result_t>;
      { get_refstate(grp_result, grp_result) } -> std::same_as<grp_result_t>;
    };

template<typename lattice_grp_element_t, typename local_grp_element_t,
         typename grp_result_t>
  requires grp_requirements<lattice_grp_element_t, grp_result_t> &&
           grp_requirements<local_grp_element_t, grp_result_t>
struct grp {
    std::vector<lattice_grp_element_t> lattice_grp;
    std::vector<local_grp_element_t> local_grp;

  public:

    grp(const std::vector<lattice_grp_element_t>& lattice_grp = {},
        const std::vector<local_grp_element_t>& local_grp = {})
        : lattice_grp(lattice_grp), local_grp(local_grp) {}

    grp(const grp& other) = default;
    grp& operator=(const grp& other) = default;

    using bitset_type = typename grp_result_t::bitset_type;

    grp_result_t check_refstate(
        const typename grp_result_t::bitset_type& input) {
      grp_result_t curr(nput);

      // Apply lattice operators only
      for (auto& lattice : lattice_grp) {
        auto next = lattice.apply(curr);
        curr = check_refstate(curr, next);
      }
      // Apply local operators then lattice operators
      for (auto& local : local_grp) {
        auto next_local = local.apply(curr);
        for (auto& lattice : lattice_grp) {
          auto next = lattice.apply(next_local);
          curr = check_refstate(curr, next);
        }
      }
      return std::move(curr);
    }

    grp_result_t get_refstate(const typename grp_result_t::bitset_type& input) {
      grp_result_t curr(input);

      // Apply lattice operators only
      for (auto& lattice : lattice_grp) {
        auto next = lattice.apply(curr);
        curr = get_refstate(curr, next);
      }
      // Apply local operators then lattice operators
      for (auto& local : local_grp) {
        auto next_local = local.apply(curr);
        for (auto& lattice : lattice_grp) {
          auto next = lattice.apply(next_local);
          curr = get_refstate(curr, next);
        }
      }
      return std::move(curr);
    }
};

}  // namespace quspin::details::basis

#ifdef QUSPIN_UNIT_TESTS

namespace quspin::details::basis {  // test cases

template class dit_perm<uint8_t>;
template dit_set<uint8_t> dit_perm<uint8_t>::app<double>(
    const dit_set<uint8_t>&, double&) const;
template dit_set<uint8_t> dit_perm<uint8_t>::inv<double>(
    const dit_set<uint8_t>&, double&) const;

template class perm_dit<uint8_t>;
template dit_set<uint8_t> perm_dit<uint8_t>::app<double>(
    const dit_set<uint8_t>&, double&) const;
template dit_set<uint8_t> perm_dit<uint8_t>::inv<double>(
    const dit_set<uint8_t>&, double&) const;

template class bit_perm<uint8_t>;
template bit_set<uint8_t> bit_perm<uint8_t>::app<double>(
    const bit_set<uint8_t>&, double&) const;
template bit_set<uint8_t> bit_perm<uint8_t>::inv<double>(
    const bit_set<uint8_t>&, double&) const;

template class perm_bit<uint8_t>;
template bit_set<uint8_t> perm_bit<uint8_t>::app<double>(
    const bit_set<uint8_t>&, double&) const;
template bit_set<uint8_t> perm_bit<uint8_t>::inv<double>(
    const bit_set<uint8_t>&, double&) const;

template class symmetry<bit_perm<uint8_t>, perm_bit<uint8_t>, bit_set<uint8_t>,
                        double>;

}  // namespace quspin::details::basis

TEST_SUITE("quspin/basis/symmetry.h") {
#include <memory>
  using namespace quspin::details::basis;

  TEST_CASE("bit_perm") {
    std::vector<int> perm = {1, 3, 2, 0};
    std::vector<int> inv = {3, 0, 2, 1};
    bit_set<uint8_t> bit_state({0, 1, 0, 1, 1, 1, 0, 0});
    bit_perm<uint8_t> bp(perm);
    double coeff = 1.0;

    auto result = bp.app(bit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "1 1 0 0 1 1 0 0 ");

    result = bp.inv(bit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "1 0 0 1 1 1 0 0 ");

    result = bp.app(bp.inv(bit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 0 1 1 1 0 0 ");

    result = bp.inv(bp.app(bit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 0 1 1 1 0 0 ");
  }

  TEST_CASE("dit_perm") {
    std::vector<int> perm = {1, 3, 2, 0};
    std::vector<int> inv = {3, 0, 2, 1};
    dit_set<uint8_t> dit_state({0, 1, 2, 0}, 3);
    dit_perm<uint8_t> dp(3, perm);
    double coeff = 1.0;

    auto result = dp.app(dit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "1 0 2 0 ");

    result = dp.inv(dit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 0 2 1 ");

    result = dp.app(dp.inv(dit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 2 0 ");

    result = dp.inv(dp.app(dit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 2 0 ");
  }

  TEST_CASE("perm_bit") {
    const uint8_t mask = 0b00110101;
    std::unique_ptr<perm_bit<uint8_t>> pb =
        std::make_unique<perm_bit<uint8_t>>(mask);
    bit_set<uint8_t> bit_state({0, 1, 0, 1, 1, 1, 0, 0});
    double coeff = 1.0;

    auto result = pb->app(bit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "1 1 1 1 0 0 0 0 ");

    result = pb->inv(bit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "1 1 1 1 0 0 0 0 ");

    result = pb->app(pb->inv(bit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 0 1 1 1 0 0 ");

    result = pb->inv(pb->app(bit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 0 1 1 1 0 0 ");

    std::vector<int> mask_vec = {1, 0, 1, 0, 1, 1, 0, 0};
    pb = std::make_unique<perm_bit<uint8_t>>(mask_vec);

    result = pb->app(bit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "1 1 1 1 0 0 0 0 ");

    result = pb->inv(bit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "1 1 1 1 0 0 0 0 ");

    result = pb->app(pb->inv(bit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 0 1 1 1 0 0 ");

    result = pb->inv(pb->app(bit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 0 1 1 1 0 0 ");
  }

  TEST_CASE("perm_dit") {
    std::vector<int> perm = {1, 2, 0};
    std::vector<int> inv = {2, 0, 1};
    dit_set<uint8_t> dit_state({0, 1, 2, 0}, 3);
    perm_dit<uint8_t> dp(3, {perm, perm}, {1, 3});
    double coeff = 1.0;

    auto result = dp.app(dit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 2 2 1 ");

    result = dp.inv(dit_state, coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 0 2 2 ");

    result = dp.app(dp.inv(dit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 2 0 ");

    result = dp.inv(dp.app(dit_state, coeff), coeff);
    CHECK(coeff == 1.0);
    CHECK(result.to_string() == "0 1 2 0 ");
  }
}

#endif
