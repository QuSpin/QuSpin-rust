// Copyright 2024 Phillip Weinberg
#pragma once

#include <concepts>
#include <quspin/basis/detail/bitbasis/info.hpp>
#include <quspin/basis/detail/space.hpp>
#include <quspin/basis/detail/types.hpp>
#include <quspin/detail/variant_container.hpp>
#include <stdexcept>

namespace quspin {

enum class SpaceTypes {
  DIT_FULLSPACE,
  DIT_SUBSPACE,
  BIT_FULLSPACE,
  BIT_SUBSPACE
};

class Space : public details::VariantContainer<spaces> {
  private:

    template<std::integral I>
    static std::size_t int_pow(const I base, const I exp);

    template<typename I, typename J = std::size_t>
    static details::basis::spaces get_subspace_variant(const SpaceTypes type,
                                                       const int lhss,
                                                       const int N,
                                                       const J Ns_est);

    static details::basis::spaces get_space_variant(
        const SpaceTypes type, const int lhss, const int N,
        const std::size_t Ns_est = 0);

  public:

    Space() = default;

    Space(const spaces &internals) : VariantContainer(internals) {}

    Space(const SpaceTypes type, const int lhss, const int N,
          const std::size_t Ns_est = 0)
        : VariantContainer(get_space_variant(type, lhss, N, Ns_est)) {}
};

static std::size_t Space::int_pow(const int base, const int exp) {
  std::size_t result = 1;
  for (int i = 0; i < exp; i++) {
    result *= base;
  }
  return result;
}

template<typename I, typename J = std::size_t>
static details::basis::spaces Space::get_subspace_variant(const SpaceTypes type,
                                                          const int lhss,
                                                          const int N,
                                                          const J Ns_est) {
  using namespace details::basis;
  const J Ns = int_pow<I>(lhss, N);

  switch (type) {
    case SpaceTypes::DIT_FULLSPACE:
      return spaces(dit_fullspace<I, J>(lhss, Ns));
    case SpaceTypes::BIT_FULLSPACE:
      return spaces(bit_fullspace<I, J>(lhss, Ns));
    case SpaceTypes::DIT_SUBSPACE:
      return spaces(dit_subspace<I, J>(lhss, Ns_est));
    case SpaceTypes::BIT_SUBSPACE:
      return spaces(bit_subspace<I, J>(lhss, Ns_est));
    default:
      throw std::invalid_argument("Invalid space type.");
  }
}

static details::basis::spaces Space::get_space_variant(
    const SpaceTypes type, const int lhss, const int N,
    const std::size_t Ns_est) {
  using namespace details::basis;
  const int bit_size = constants::bits[lhss] * N;

  if (bit_size < bit_info<uint32_t>::bits) {
    return get_space_variant<uint32_t>(type, lhss, N, Ns_est);
  } else if (bit_size < bit_info<uint64_t>::bits) {
    return get_space_variant<uint64_t>(type, lhss, N, Ns_est);
  } else if (bit_size < bit_info<uint128_t>::bits) {
    return get_space_variant<uint128_t>(type, lhss, N, Ns_est);
  } else if (bit_size < bit_info<uint256_t>::bits) {
    return get_space_variant<uint256_t>(type, lhss, N, Ns_est);
  } else if (bit_size < bit_info<uint16384_t>::bits) {
    return get_space_variant<uint16384_t>(type, lhss, N, Ns_est);
  } else {
    throw std::invalid_argument("Space size too large.");
  }
}

}  // namespace quspin
