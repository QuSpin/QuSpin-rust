// Copyright 2024 Phillip Weinberg
#include <algorithm>
#include <array>
#include <numeric>
#include <quspin/array/array.hpp>
#include <quspin/array/kernels.hpp>
#include <quspin/detail/broadcast.hpp>
#include <quspin/detail/cast.hpp>
#include <quspin/detail/dispatch.hpp>
#include <quspin/detail/error.hpp>
#include <quspin/detail/math.hpp>
#include <quspin/detail/operators.hpp>
#include <quspin/detail/optional.hpp>
#include <type_traits>
#include <variant>

namespace quspin {

using OptionalArray = detail::Optional<Array>;

Array add(const Array &lhs, const Array &rhs,
          detail::Optional<Array> out /* = detail::Optional<Array>() */) {
  auto kernel = [](auto &out, const auto &lhs, const auto &rhs) {
    using out_t = typename std::decay_t<decltype(out)>::value_type;

    std::transform(lhs.begin(), lhs.end(), rhs.begin(), out.begin(),
                   [](const auto &lhs, const auto &rhs) {
                     return detail::cast<out_t>(lhs + rhs);
                   });
    return detail::ReturnVoidError();
  };
  return detail::dispatch_elementwise(kernel, out, lhs, rhs);
}

Scalar norm(const Array &arr) {
  auto kernel = [](const auto &arr) {
    using T = typename std::decay_t<decltype(arr)>::value_type;
    using norm_t = std::decay_t<decltype(detail::abs_squared(T()))>;

    norm_t result = std::accumulate(
        arr.begin(), arr.end(), norm_t(),
        [](auto &&acc, auto &&val) { return acc + detail::abs_squared(val); });

    return detail::ErrorOr<Scalar>(
        Scalar(detail::cast<double>(std::sqrt(result))));
  };

  auto static_args_check = [](const auto &) { return detail::ValidArgs(); };

  auto dynamic_args_check = [](const auto &arr) {
    if (arr.ndim() != 1) {
      return detail::ReturnVoidError(detail::Error(
          detail::ErrorType::ValueError, "Expecting 1-dimensional array"));
    }
    return detail::ReturnVoidError();
  };

  return detail::dispatch_scalar(kernel, static_args_check, dynamic_args_check,
                                 arr);
}

bool allclose(const Array &lhs, const Array &rhs,
              const double rtol /* = 1e-5 */, const double atol /* = 1e-8 */) {
  auto kernel = [&atol, &rtol](const auto &lhs, const auto &rhs) {
    using lhs_t = typename std::decay_t<decltype(lhs)>::value_type;
    using rhs_t = typename std::decay_t<decltype(rhs)>::value_type;
    using value_t = detail::upcast_t<lhs_t, rhs_t>;

    auto check = [&atol, &rtol](const auto &lhs, const auto &rhs) {
      const value_t l = detail::cast<value_t>(lhs);
      const value_t r = detail::cast<value_t>(rhs);

      return detail::abs(l - r) <=
             atol + rtol * std::max(detail::abs(l), detail::abs(r));
    };

    auto rhs_it = rhs.begin();
    for (auto lhs_it = lhs.begin(); lhs_it != lhs.end(); ++lhs_it, ++rhs_it) {
      if (!check(*lhs_it, *rhs_it)) {
        return detail::ErrorOr<Scalar>(Scalar(static_cast<uint8_t>(false)));
      }
    }
    return detail::ErrorOr<Scalar>(Scalar(static_cast<uint8_t>(true)));
  };

  auto static_args_check = [](const auto &...) { return detail::ValidArgs(); };

  auto dynamic_args_check = [](const auto &lhs, const auto &rhs) {
    if (lhs.shape() != rhs.shape()) {
      return detail::ReturnVoidError(
          detail::Error(detail::ErrorType::ValueError, "Incompatible shapes"));
    }
    return detail::ReturnVoidError();
  };

  Scalar result = detail::dispatch_scalar(kernel, static_args_check,
                                          dynamic_args_check, lhs, rhs);

  return static_cast<bool>(static_cast<uint8_t>(result));
}

}  // namespace quspin
