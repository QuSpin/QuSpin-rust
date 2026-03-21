// Copyright 2024 Phillip Weinberg

#include <cassert>
#include <memory>
#include <quspin/basis/detail/space.hpp>
#include <quspin/basis/detail/symmetry/single_grp_element.hpp>
#include <quspin/basis/grp/grp_element.hpp>
#include <quspin/basis/hardcore.hpp>
#include <quspin/detail/error.hpp>
#include <quspin/detail/variant_container.hpp>
#include <quspin/operator.hpp>
#include <stdexcept>
#include <variant>
#include <vector>

namespace quspin {

space_t HardcoreBasis::make_space(
    const std::size_t N_max, const bool subspace,
    const std::vector<perm_grp_t> &loattice_perm_elements,
    const std::vector<std::size_t> &locs, const int z) {
  if (N_max < 33) {
    return make_space_typed<detail::basis::bits32_t>(
        N_max, subspace, loattice_perm_elements, locs, z);
  } else if (N_max < 64) {
    return make_space_typed<detail::basis::bits64_t>(
        N_max, subspace, loattice_perm_elements, locs, z);
  } else if (N_max < 128) {
    return make_space_typed<detail::basis::bits128_t>(
        N_max, subspace, loattice_perm_elements, locs, z);
  } else if (N_max < 256) {
    return make_space_typed<detail::basis::bits256_t>(
        N_max, subspace, loattice_perm_elements, locs, z);
  } else if (N_max < 1024) {
    return make_space_typed<detail::basis::bits1024_t>(
        N_max, subspace, loattice_perm_elements, locs, z);
  } else if (N_max < 4096) {
    return make_space_typed<detail::basis::bits4096_t>(
        N_max, subspace, loattice_perm_elements, locs, z);
  } else if (N_max < 16384) {
    return make_space_typed<detail::basis::bits16384_t>(
        N_max, subspace, loattice_perm_elements, locs, z);
  } else {
    throw std::invalid_argument("N_max too large for full space");
  }
}

std::size_t HardcoreBasis::size() const {
  using namespace detail::basis;
  return std::visit([](const auto &space) { return space->size(); },
                    internals_);
}

std::string HardcoreBasis::fock_state(const std::size_t index,
                                      const std::size_t N) const {
  using namespace detail::basis;
  return std::visit(
      [index, &N](const auto &space) {
        const auto &state = space->state_at(index);
        detail::basis::dit_manip<2> manip;
        return manip.to_string(state, N);
      },
      internals_);
}

template<typename OperatorType>
void HardcoreBasis::construct_from(
    const OperatorType &op,
    const std::vector<std::vector<std::size_t>> &seeds) {
  using namespace detail::basis;

  detail::visit_or_error<std::monostate>(
      [&seeds](const auto &op, auto space) {
        using space_t = typename std::decay_t<decltype(space)>::element_type;
        using bitset_t = typename space_t::bitset_type;

        if constexpr (std::is_same_v<bitset_t, bits32_t> ||
                      std::is_same_v<bitset_t, bits64_t>) {
          if constexpr (std::is_same_v<space_t,
                                       hardcore_boson_space<bitset_t>>) {
            return detail::ReturnVoidError(
                detail::Error(detail::ErrorType::ValueError,
                              "Basis is a full space, not a subspace."));
          } else {
            std::vector<bitset_t> seed_values =
                get_seed_values<bitset_t>(seeds);
            for (const auto &seed : seed_values) {
              space->build(op, seed);
            }
            return detail::ReturnVoidError();
          }
        } else {
          std::vector<bitset_t> seed_values = get_seed_values<bitset_t>(seeds);
          for (const auto &seed : seed_values) {
            space->build(op, seed);
          }
          return detail::ReturnVoidError();
        }
      },
      op.get_variant_obj(), get_variant_obj());
}

template<>
void HardcoreBasis::construct_from<PauliHamiltonian>(
    const PauliHamiltonian &op,
    const std::vector<std::vector<std::size_t>> &seeds);

}  // namespace quspin
