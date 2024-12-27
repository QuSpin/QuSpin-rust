// Copyright 2024 Phillip Weinberg
#pragma once

#include <cassert>
#include <memory>
#include <quspin/basis/detail/space.hpp>
#include <quspin/basis/detail/symmetry/single_grp_element.hpp>
#include <quspin/basis/grp/grp_element.hpp>
#include <quspin/detail/error.hpp>
#include <quspin/detail/variant_container.hpp>
#include <stdexcept>
#include <variant>
#include <vector>

namespace quspin {

template<typename bitset_t>
using hardcore_boson_grp =
    detail::basis::grp<detail::basis::grp_result<bitset_t>,
                       lattice_grp_element<bitset_t>,
                       bitflip_grp_element<bitset_t>>;

template<typename bitset_t, typename norm_t>
using hardcore_boson_symmetric_subspace =
    detail::basis::symmetric_subspace<hardcore_boson_grp<bitset_t>, bitset_t,
                                      norm_t>;

template<typename bitset_t>
using hardcore_boson_subspace = detail::basis::subspace<bitset_t>;

template<typename bitset_t>
using hardcore_boson_space = detail::basis::space<bitset_t>;

#ifdef USE_BOOST
using space_t = std::variant<
    std::shared_ptr<hardcore_boson_space<detail::basis::uint32_t>>,
    std::shared_ptr<hardcore_boson_space<detail::basis::uint64_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint32_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint64_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint128_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint256_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint1024_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint4096_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint16384_t>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint32_t, double>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint64_t, double>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint128_t, double>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint256_t, double>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint1024_t, double>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint4096_t, double>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint16384_t, double>>>;

#else
using space_t = std::variant<
    std::shared_ptr<hardcore_boson_space<detail::basis::uint32_t>>,
    std::shared_ptr<hardcore_boson_space<detail::basis::uint64_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint32_t>>,
    std::shared_ptr<hardcore_boson_subspace<detail::basis::uint64_t>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint32_t, double>>,
    std::shared_ptr<
        hardcore_boson_symmetric_subspace<detail::basis::uint64_t, double>>>;

#endif

class HardcoreBasis : public detail::VariantContainer<space_t> {
  private:

    using detail::VariantContainer<space_t>::internals_;

    template<typename bitset_t>
    static std::vector<bitset_t> get_seed_values(
        const std::vector<std::vector<std::size_t>> &seeds) {
      using namespace detail::basis;

      detail::basis::dit_manip<2> manip;

      std::vector<bitset_t> seed_values;

      for (const auto &seed : seeds) {
        if (seed.size() > bit_info<bitset_t>::bits) {
          throw std::invalid_argument(
              "Seed too large for reqyested system size");
        }

        bitset_t value = 0;
        std::size_t index = 0;
        for (const auto &val : seed) {
          value = manip.set_sub_bitstring(value, val, index);
          index++;
        }

        seed_values.push_back(value);
      }

      return seed_values;
    }

    template<typename bitset_t>
    static hardcore_boson_grp<bitset_t> create_grp(
        const std::vector<perm_grp_t> &loattice_perm_elements,
        const std::vector<std::size_t> &locs, const int z) {
      const std::vector<lattice_grp_element<bitset_t>> &lattice_grp =
          create_lattice_grp<bitset_t>(2, loattice_perm_elements);
      const std::vector<bitflip_grp_element<bitset_t>> &bitflip_grp =
          create_bitflip_grp<bitset_t>(locs, z);
      return hardcore_boson_grp<bitset_t>(lattice_grp, bitflip_grp);
    }

    template<typename bitset_t>
    static space_t make_space_typed(
        const std::size_t N_max, const bool subspace,
        const std::vector<perm_grp_t> &loattice_perm_elements,
        const std::vector<std::size_t> &locs, const int z) {
      using namespace detail::basis;

      static_assert(BasisPrimativeTypes<bitset_t>);

      if (loattice_perm_elements.size() > 0 || locs.size() > 0) {
        if (!subspace) {
          throw std::invalid_argument(
              "Symmmtries are a subspace, set subspace=true.");
        }
        hardcore_boson_grp<bitset_t> grp =
            create_grp<bitset_t>(loattice_perm_elements, locs, z);
        return space_t(
            std::make_shared<
                hardcore_boson_symmetric_subspace<bitset_t, double>>(grp));
      } else if (subspace) {
        return space_t(std::make_shared<hardcore_boson_subspace<bitset_t>>());
      } else if constexpr (std::is_same_v<bitset_t, detail::basis::uint32_t> ||
                           std::is_same_v<bitset_t, detail::basis::uint64_t>) {
        assert(N_max < 64);
        const std::size_t Ns = std::size_t(1) << N_max;
        return space_t(std::make_shared<hardcore_boson_space<bitset_t>>(Ns));
      } else {
        throw std::invalid_argument("N_max too large for full space");
      }
    }

    static space_t make_space(
        const std::size_t N_max, const bool subspace,
        const std::vector<perm_grp_t> &loattice_perm_elements,
        const std::vector<std::size_t> &locs, const int z) {
      if (N_max < 33) {
        return make_space_typed<detail::basis::uint32_t>(
            N_max, subspace, loattice_perm_elements, locs, z);
      } else if (N_max < 64) {
        return make_space_typed<detail::basis::uint64_t>(
            N_max, subspace, loattice_perm_elements, locs, z);
      }
#ifdef USE_BOOST
      else if (N_max < 128) {
        return make_space_typed<detail::basis::uint128_t>(
            N_max, subspace, loattice_perm_elements, locs, z);
      } else if (N_max < 256) {
        return make_space_typed<detail::basis::uint256_t>(
            N_max, subspace, loattice_perm_elements, locs, z);
      } else if (N_max < 1024) {
        return make_space_typed<detail::basis::uint1024_t>(
            N_max, subspace, loattice_perm_elements, locs, z);
      } else if (N_max < 4096) {
        return make_space_typed<detail::basis::uint4096_t>(
            N_max, subspace, loattice_perm_elements, locs, z);
      } else if (N_max < 16384) {
        return make_space_typed<detail::basis::uint16384_t>(
            N_max, subspace, loattice_perm_elements, locs, z);
      }
#endif
      else {
        throw std::invalid_argument("N_max too large for full space");
      }
    }

  public:

    HardcoreBasis(const std::size_t N_max, const bool subspace = false,
                  const std::vector<perm_grp_t> &loattice_perm_elements = {},
                  const std::vector<std::size_t> &locs = {}, const int z = 1)
        : detail::VariantContainer<space_t>(
              make_space(N_max, subspace, loattice_perm_elements, locs, z)) {}

    std::size_t size() const {
      using namespace detail::basis;
      return std::visit([](const auto &space) { return space->size(); },
                        internals_);
    }

    decltype(auto) fock_state(const std::size_t index,
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
    void construct_from(OperatorType &&op,
                        const std::vector<std::vector<std::size_t>> &seeds) {
      using namespace detail::basis;

      detail::visit_or_error<std::monostate>(
          [&seeds](const auto &op, auto space) {
            using space_t =
                typename std::decay_t<decltype(space)>::element_type;
            using bitset_t = typename space_t::bitset_type;

            static_assert(detail::basis::BasisPrimativeTypes<bitset_t>);

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
          },
          op.get_variant_obj(), get_variant_obj());
    }
};

}  // namespace quspin
