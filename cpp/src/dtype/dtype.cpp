// Copyright 2024 Phillip Weinberg

#include <cstdint>
#include <quspin/detail/operators.hpp>
#include <quspin/dtype/detail/dtype.hpp>
#include <quspin/dtype/dtype.hpp>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace quspin {

template DType::DType(const detail::dtype<int8_t> &);
template DType::DType(const detail::dtype<uint8_t> &);
template DType::DType(const detail::dtype<int16_t> &);
template DType::DType(const detail::dtype<uint16_t> &);
template DType::DType(const detail::dtype<int32_t> &);
template DType::DType(const detail::dtype<uint32_t> &);
template DType::DType(const detail::dtype<int64_t> &);
template DType::DType(const detail::dtype<uint64_t> &);
template DType::DType(const detail::dtype<float> &);
template DType::DType(const detail::dtype<double> &);
template DType::DType(const detail::dtype<detail::cfloat> &);
template DType::DType(const detail::dtype<detail::cdouble> &);

bool DType::operator==(const DType &dtype) const {
  return std::visit(
      [](auto &&lhs, auto &&rhs) {
        using lhs_t = typename std::decay_t<decltype(lhs)>::value_type;
        using rhs_t = typename std::decay_t<decltype(rhs)>::value_type;
        return std::is_same_v<lhs_t, rhs_t>;
      },
      internals_, dtype.get_variant_obj());
}

std::string DType::name() const {
  return std::visit(
      [](const auto &dtype) {
        using T = typename std::decay_t<decltype(dtype)>::value_type;
        if constexpr (std::is_same_v<T, int8_t>) {
          return std::string("int8");
        } else if constexpr (std::is_same_v<T, uint8_t>) {
          return std::string("uint8");
        } else if constexpr (std::is_same_v<T, int16_t>) {
          return std::string("int16");
        } else if constexpr (std::is_same_v<T, uint16_t>) {
          return std::string("uint16");
        } else if constexpr (std::is_same_v<T, int32_t>) {
          return std::string("int32");
        } else if constexpr (std::is_same_v<T, uint32_t>) {
          return std::string("uint32");
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return std::string("int64");
        } else if constexpr (std::is_same_v<T, uint64_t>) {
          return std::string("uint64)");
        } else if constexpr (std::is_same_v<T, float>) {
          return std::string("float");
        } else if constexpr (std::is_same_v<T, double>) {
          return std::string("double");
        } else if constexpr (std::is_same_v<T, detail::cfloat>) {
          return std::string("cfloat");
        } else {
          return std::string("cdouble");
        }
      },
      internals_);
}

DType result_dtype(std::vector<DType> &args) {
  switch (args.size()) {
    case 0:
      throw std::invalid_argument("Empty list of dtypes");
    case 1:
      return args[0];
    case 2:
      return std::visit(
          [](const auto &dtype1, const auto &dtype2) {
            return DType(detail::result_dtype(dtype1, dtype2));
          },
          args[0].get_variant_obj(), args[1].get_variant_obj());
    default:
      auto next_args = std::vector<DType>(args.begin() + 1, args.end());
      DType recurse = result_dtype(next_args);

      auto new_args = std::vector<DType>({args[0], recurse});
      return result_dtype(new_args);
  }
}

bool int_dtype(const DType &dtype) {
  return std::visit(
      [](const auto &dtype) {
        using T = typename std::decay_t<decltype(dtype)>::value_type;
        return std::is_integral_v<T>;
      },
      dtype.get_variant_obj());
}

bool float_dtype(const DType &dtype) {
  return std::visit(
      [](const auto &dtype) {
        using T = typename std::decay_t<decltype(dtype)>::value_type;
        return std::is_floating_point_v<T>;
      },
      dtype.get_variant_obj());
}

bool complex_dtype(const DType &dtype) {
  return std::visit(
      [](const auto &dtype) {
        using T = typename std::decay_t<decltype(dtype)>::value_type;
        return std::is_same_v<T, detail::cfloat> ||
               std::is_same_v<T, detail::cdouble>;
      },
      dtype.get_variant_obj());
}

}  // namespace quspin
