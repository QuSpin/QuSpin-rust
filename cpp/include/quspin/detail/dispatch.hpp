// Copyright 2024 Phillip Weinberg
#pragma once
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif

#include <any>
#include <quspin/array/array.hpp>
#include <quspin/array/detail/array.hpp>
#include <quspin/detail/error.hpp>
#include <quspin/detail/operators.hpp>
#include <quspin/detail/optional.hpp>
#include <quspin/scalar/scalar.hpp>

namespace quspin { namespace detail {

struct ValidArgs {
    static constexpr bool has_error() { return false; }
};

template<typename Kernel, typename StaticArgsCheck, typename DynamicArgsCheck,
         typename... Args>
void dispatch(Kernel &&kernel, StaticArgsCheck &&static_args_check,
              DynamicArgsCheck &&dynamic_args_check, Args... args) {
  visit_or_error<std::monostate>(
      [&kernel, &dynamic_args_check](auto &...args) {
        using static_args_check_return_t = decltype(static_args_check(args...));

        if constexpr (static_args_check_return_t::has_error()) {
          return ReturnVoidError(static_args_check_return_t::get_error());
        } else {
          using kernel_return_t = decltype(kernel(args...));
          using dynamic_args_check_return_t =
              decltype(dynamic_args_check(args...));
          static_assert(
              std::is_same_v<dynamic_args_check_return_t, ReturnVoidError>,
              "Dynamic args check must return ReturnVoidError");
          static_assert(std::is_same_v<kernel_return_t, ReturnVoidError>,
                        "Kernel must return ReturnVoidError");

          ReturnVoidError err = dynamic_args_check(args...);
          if (err.has_error()) {
            return err;
          } else {
            return kernel(args...);
          }
        }
      },
      args.get_variant_obj()...);
}

template<typename Kernel, typename ResultShape, typename ResultDType,
         typename StaticArgsCheck, class DynamicArgsCheck, typename... Args>
Array dispatch_array(Kernel &&kernel, ResultShape &&result_shape_func,
                     ResultDType &&result_dtype_func,
                     StaticArgsCheck &&static_args_check,
                     DynamicArgsCheck &&dynamic_args_check,
                     Optional<Array> out_option, Args... args) {
  std::vector<std::size_t> shape = result_shape_func(args...);
  DType output_dtype = result_dtype_func(args...);

  auto default_result = [&shape, &output_dtype]() {
    return Array(shape, output_dtype);
  };

  Array out = out_option.get(default_result);

  dispatch(kernel, static_args_check, dynamic_args_check, out, args...);

  return out;
}

template<typename Kernel>
Array dispatch_elementwise(Kernel &&kernel, Optional<Array> out,
                           const Array &lhs, const Array &rhs) {
  struct InvalidDtype {
      static constexpr bool has_error() { return true; }

      static Error get_error() {
        return Error(ErrorType::ValueError, "Incompatible out type");
      }
  };

  auto result_shape = [](const Array &lhs, const Array &) {
    return lhs.shape();
  };

  auto result_dtype_func = [](const Array &lhs, const Array &rhs) {
    std::vector<DType> dtypes = {lhs.dtype(), rhs.dtype()};
    return result_dtype(dtypes);
  };

  auto static_args_check = [](auto &out, const auto &lhs, const auto &rhs) {
    using lhs_t = typename std::decay_t<decltype(lhs)>::value_type;
    using rhs_t = typename std::decay_t<decltype(rhs)>::value_type;
    using out_t = typename std::decay_t<decltype(out)>::value_type;
    using result_t = std::common_type_t<lhs_t, rhs_t>;

    if constexpr (can_safe_cast_v<result_t, out_t>) {
      return ValidArgs();
    } else {
      return InvalidDtype();
    }
  };

  auto dynamic_args_check = [](auto &out, const auto &lhs, const auto &rhs) {
    if (lhs.shape() != rhs.shape() || lhs.shape() != out.shape()) {
      return ReturnVoidError(
          Error(ErrorType::ValueError, "Incompatible shapes"));
    }
    return ReturnVoidError();
  };

  return dispatch_array(kernel, result_shape, result_dtype_func,
                        static_args_check, dynamic_args_check, out, lhs, rhs);
}

template<typename Kernel, typename StaticArgsCheck, typename DynamicArgsCheck,
         typename... Args>
Scalar dispatch_scalar(Kernel &&kernel, StaticArgsCheck &&static_args_check,
                       DynamicArgsCheck &&dynamic_args_check,
                       const Args &...args) {
  return visit_or_error<Scalar>(
      [&kernel, &dynamic_args_check](const auto &...args) {
        using static_args_check_return_t = decltype(static_args_check(args...));

        if constexpr (static_args_check_return_t::has_error()) {
          return ErrorOr<Scalar>(static_args_check_return_t::get_error());
        } else {
          using kernel_return_t = decltype(kernel(args...));
          using dynamic_args_check_return_t =
              decltype(dynamic_args_check(args...));
          static_assert(
              std::is_same_v<dynamic_args_check_return_t, ReturnVoidError>,
              "Dynamic args check must return Scalar");
          static_assert(std::is_same_v<kernel_return_t, ErrorOr<Scalar>>,
                        "Kernel must return Scalar");

          ReturnVoidError err = dynamic_args_check(args...);

          if (err.has_error()) {
            return ErrorOr<Scalar>(err.get_error());
          } else {
            return kernel(args...);
          }
        }
      },
      args.get_variant_obj()...);
}

}}  // namespace quspin::detail

#ifdef _MSC_VER
#pragma warning(pop)
#endif
