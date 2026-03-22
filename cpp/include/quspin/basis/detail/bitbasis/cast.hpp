// Copyright 2024 Phillip Weinberg
#pragma once

#include <concepts>
#include <limits>
#include <quspin/basis/detail/types.hpp>

namespace quspin::detail::basis {

template<typename J, BasisPrimativeTypes I>
  requires std::integral<J> && BasisPrimativeTypes<I>
J integer_cast(const I s) {
  if constexpr (std::is_same_v<I, bits32_t> || std::is_same_v<I, bits64_t>) {
    if (s < std::numeric_limits<J>::max()) {
      return static_cast<J>(s);
    } else {
      throw std::invalid_argument("Integer cast overflow");
    }
  } else {
    return static_cast<J>(s.to_ullong());
  }
}

}  // namespace quspin::detail::basis
