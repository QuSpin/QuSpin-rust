// Copyright 2024 Phillip Weinberg
#include <iostream>
#include <memory>
#include <quspin/array/array.hpp>
#include <quspin/basis/hardcore.hpp>
#include <quspin/detail/default_containers.hpp>
#include <quspin/detail/error.hpp>
#include <quspin/detail/type_concepts.hpp>
#include <quspin/dtype/dtype.hpp>
#include <quspin/operator.hpp>
#include <quspin/qmatrix/detail/qmatrix.hpp>
#include <quspin/qmatrix/qmatrix.hpp>
#include <stdexcept>
#include <variant>

namespace quspin {

detail::qmatrices QMatrix::default_value() {
  return detail::qmatrices(make_qmatrices<double, int32_t, uint8_t>());
}

// TODO switch to using shared pointer for variant
std::size_t QMatrix::dim() const {
  return detail::visit_or_error<std::size_t>(
      [](const auto &qmat) {
        return detail::ErrorOr<std::size_t>(qmat->dim());
      },
      internals_);
}

std::size_t QMatrix::num_coeff() const {
  return detail::visit_or_error<std::size_t>(
      [](const auto &qmat) {
        return detail::ErrorOr<std::size_t>(qmat->num_coeff());
      },
      internals_);
}

Array QMatrix::dot(const bool overwrite_out, const Array &coeff,
                   const Array &input, Array &output) const {
  detail::visit_or_error<std::monostate>(
      [overwrite_out](const auto qmat, const auto &coeff, const auto &input,
                      const auto &output) {
        using out_t = typename std::decay_t<decltype(output)>::value_type;
        using in_t = typename std::decay_t<decltype(input)>::value_type;
        using coeff_t = typename std::decay_t<decltype(coeff)>::value_type;
        using qmat_t = typename std::decay_t<decltype(qmat)>::element_type;
        using mat_t = typename qmat_t::value_type;

        if constexpr (std::is_same_v<out_t, in_t> &&
                      std::is_same_v<in_t, coeff_t> &&
                      std::convertible_to<detail::upcast_t<mat_t, out_t>,
                                          out_t>) {
          const detail::svector_t<coeff_t, 256> coeff_vec(coeff.begin(),
                                                          coeff.end());
          try {
            qmat->dot(overwrite_out, coeff_vec, input, output);
            return detail::ReturnVoidError();
          } catch (const std::runtime_error &e) {
            return detail::ReturnVoidError(
                detail::Error(detail::ErrorType::ValueError, e.what()));
          }
        } else {
          return detail::ReturnVoidError(detail::Error(
              detail::ErrorType::ValueError, "Invalid type for output array"));
        }
      },
      internals_, coeff.get_variant_obj(), input.get_variant_obj(),
      output.get_variant_obj());

  return output;
}

}  // namespace quspin
