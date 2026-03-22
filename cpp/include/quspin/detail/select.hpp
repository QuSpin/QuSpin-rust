// Copyright 2024 Phillip Weinberg
#pragma once

#include <quspin/detail/error.hpp>
#include <quspin/dtype/dtype.hpp>
#include <sstream>
#include <type_traits>
#include <variant>

namespace quspin { namespace detail {

template<typename Type, typename... Types>
struct is_one_of : std::disjunction<std::is_same<Types, Type>...> {};

template<typename Type, typename... Types>
inline constexpr bool is_one_of_v = is_one_of<Type, Types...>::value;

template<typename... Types, typename Variants>
DTypeObject<std::variant<Types...>> select(DTypeObject<Variants> &obj,
                                           std::string name = "") {
  static_assert(sizeof...(Types) > 0, "No types specified for select");
  using select_variant_t = std::variant<Types...>;
  using dtype_object_t = DTypeObject<select_variant_t>;

  return visit_or_error<dtype_object_t>(
      [&name](auto &&arg) -> ErrorOr<dtype_object_t> {
        using arg_t = std::remove_reference_t<decltype(arg)>;

        if constexpr (is_one_of_v<arg_t, Types...>) {
          select_variant_t select_arg = arg;
          dtype_object_t select_obj(select_arg);
          return ErrorOr<dtype_object_t>(select_obj);
        } else {
          std::stringstream error_msg;

          DType input_dtype = DType::of<arg_t>();
          auto expected_dtypes = std::vector<DType>{DType::of<Types>()...};
          error_msg << "Invalid type " << input_dtype.name();
          if (!name.empty()) {
            error_msg << " for " << name;
          }
          error_msg << ", expected one of the specified types: ";

          for (auto &dtype : expected_dtypes) {
            error_msg << dtype.name() << ", ";
          }
          Error error = Error(ErrorType::ValueError, error_msg.str());
          return ErrorOr<dtype_object_t>(error);
        }
      },
      obj.get_variant_obj());
}

}}  // namespace quspin::detail
