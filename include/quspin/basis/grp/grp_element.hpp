// Copyright 2024 Phillip Weinberg
#pragma once

#include <concepts>
#include <quspin/basis/detail/bitbasis/dit_perm.hpp>
#include <quspin/basis/detail/symmetry/grp.hpp>
#include <quspin/basis/detail/symmetry/single_grp_element.hpp>
#include <quspin/basis/detail/types.hpp>
#include <sstream>
#include <variant>
#include <vector>

namespace quspin {

using perm_grp_t = std::tuple<std::complex<double>, std::vector<std::size_t>>;

// grp elements
template<typename element_t>
std::size_t get_perm_size(const std::vector<element_t> &grp) {
  std::size_t size = 0;
  for (const auto &ele : grp) {
    const auto &perm = std::get<std::vector<std::size_t>>(ele);
    size = std::max(size, perm.size());
  }
  for (const auto &ele : grp) {
    const auto &perm = std::get<std::vector<std::size_t>>(ele);
    if (perm.size() != size) {
      throw std::invalid_argument("Permutations must be the same size");
    }
  }
  return size;
}

template<typename bitset_t>
using lattice_grp_element =
    detail::basis::grp_element<detail::basis::perm_dit_locations<bitset_t>,
                               bitset_t>;

template<typename bitset_t>
std::vector<lattice_grp_element<bitset_t>> create_lattice_grp(
    const std::size_t lhss,
    const std::vector<std::tuple<std::complex<double>,
                                 std::vector<std::size_t>>> &lattice_grp) {
  std::vector<lattice_grp_element<bitset_t>> grp;
  for (const auto &[coeff, perm] : lattice_grp) {
    grp.emplace_back(coeff,
                     detail::basis::perm_dit_locations<bitset_t>(lhss, perm));
  }
  return grp;
}

template<typename bitset_t>
using bitflip_grp_element =
    detail::basis::grp_element<detail::basis::perm_dit_mask<bitset_t>,
                               bitset_t>;

template<typename bitset_t>
std::vector<bitflip_grp_element<bitset_t>> create_bitflip_grp(
    const std::vector<std::size_t> &locs, const int z) {
  using namespace detail::basis;

  std::vector<bitflip_grp_element<bitset_t>> grp;
  bitset_t mask = 0;
  for (const auto loc : locs) {
    mask |= (bitset_t(1) << loc);
  }
  perm_dit_mask<bitset_t> perm_mask(mask);
  grp.emplace_back(std::complex<double>(z, 0.0), perm_mask);
  return grp;
}

template<typename bitset_t>
using dynamic_local_grp_element =
    detail::basis::grp_element<detail::basis::dynamic_perm_dit_values<bitset_t>,
                               bitset_t>;

template<typename bitset_t>
std::vector<dynamic_local_grp_element<bitset_t>> create_dynamic_local_grp(
    const std::vector<perm_grp_t> &local_grp,
    const std::vector<std::size_t> &locs) {
  using namespace detail::basis;

  const std::size_t lhss = get_perm_size(local_grp);

  std::vector<dynamic_local_grp_element<bitset_t>> grp;
  for (const auto &[coeff, perm] : local_grp) {
    const auto dit_op = dynamic_perm_dit_values<bitset_t>(lhss, perm, locs);
    grp.emplace_back(coeff, dit_op);
  }
  return grp;
}

template<typename bitset_t>
using dynamic_spin_inversion_grp_element =
    detail::basis::grp_element<detail::basis::dynamic_higher_spin_inv<bitset_t>,
                               bitset_t>;

template<typename bitset_t>
std::vector<dynamic_spin_inversion_grp_element<bitset_t>>
create_dynamic_spin_inversion_grp(const std::vector<std::size_t> &locs,
                                  const std::size_t lhss, const int z) {
  std::vector<dynamic_spin_inversion_grp_element<bitset_t>> grp;
  using namespace detail::basis;

  dynamic_higher_spin_inv<bitset_t> spin_inv(locs, lhss);
  grp.emplace_back(std::complex<double>(z, 0.0), spin_inv);

  return grp;
}

template<typename bitset_t, std::size_t lhss>
using local_grp_element =
    detail::basis::grp_element<detail::basis::perm_dit_values<bitset_t, lhss>,
                               bitset_t>;

template<typename bitset_t, std::size_t lhss>
std::vector<local_grp_element<bitset_t, lhss>> create_local_grp(
    const std::vector<perm_grp_t> &local_grp,
    const std::vector<std::size_t> &locs) {
  using namespace detail::basis;

  const std::size_t size = get_perm_size(local_grp);
  if (lhss != size) {
    std::stringstream message;
    message << "Permutations must be the same size: " << size;
    throw std::invalid_argument(message.str());
  }
  std::vector<local_grp_element<bitset_t, lhss>> grp;
  for (const auto &[coeff, perm] : local_grp) {
    const auto dit_op = perm_dit_values<bitset_t, lhss>(perm, locs);
    grp.emplace_back(coeff, dit_op);
  }
  return grp;
}

template<typename bitset_t, std::size_t lhss>
using spin_inversion_grp_element =
    detail::basis::grp_element<detail::basis::higher_spin_inv<bitset_t, lhss>,
                               bitset_t>;

template<typename bitset_t, std::size_t lhss>
std::vector<spin_inversion_grp_element<bitset_t, lhss>>
create_spin_inversion_grp(const std::vector<std::size_t> &locs, const int z) {
  std::vector<spin_inversion_grp_element<bitset_t, lhss>> grp;
  using namespace detail::basis;

  higher_spin_inv<bitset_t, lhss> spin_inv(locs);
  grp.emplace_back(std::complex<double>(z, 0.0), spin_inv);

  return grp;
}

}  // namespace quspin
