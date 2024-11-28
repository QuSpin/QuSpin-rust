// Copyright 2024 Phillip Weinberg
#pragma once

#include <concepts>
#include <limits>
#include <quspin/basis/detail/types.hpp>

namespace quspin::details::basis {

template<typename J, BasisPrimativeTypes I>
  requires std::integral<J> && BasisPrimativeTypes<I>
J integer_cast(const I s) {
  if (s > std::numeric_limits<J>::max()) {
    return -1;
  } else {
#ifdef USE_BOOST
    return boost::numeric_cast<J>(s);
#else
    return static_cast<J>(s);
#endif
  }
}

}  // namespace quspin::details::basis
