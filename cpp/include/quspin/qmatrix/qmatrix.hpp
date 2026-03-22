// Copyright 2024 Phillip Weinberg
#pragma once

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
#include <stdexcept>
#include <variant>

namespace quspin {

class QMatrix : public DTypeObject<detail::qmatrices> {
  private:

    using DTypeObject<detail::qmatrices>::internals_;

    template<typename dtype_t, typename index_t, typename cindex_t,
             typename... Args>
      requires QMatrixTypes<dtype_t, index_t, cindex_t>
    static detail::qmatrices make_qmatrices(Args &&...args) {
      using qmat_t = detail::qmatrix<dtype_t, index_t, cindex_t>;
      return detail::qmatrices(
          std::make_shared<qmat_t>(std::forward<Args>(args)...));
    }

    static detail::qmatrices default_value();

    template<typename Hamiltonian, typename Basis>
    detail::qmatrices create_internals(const Hamiltonian &ham,
                                       const Basis &basis, const DType &dtype) {
      using namespace detail;
      return visit_or_error<qmatrices>(
          [](const auto &ham, const auto &basis, const auto &dtype) {
            using dtype_t = typename std::decay_t<decltype(dtype)>::value_type;
            using cindex_t = typename std::decay_t<decltype(ham)>::cindex_type;
            using result_t = ErrorOr<qmatrices>;

            if (basis->size() < std::numeric_limits<int32_t>::max()) {
              if constexpr (QMatrixTypes<dtype_t, int32_t, cindex_t>) {
                return result_t(
                    make_qmatrices<dtype_t, int32_t, cindex_t>(ham, *basis));
              } else {
                return result_t(
                    Error(ErrorType::ValueError, "Invalid type for qmatrix"));
              }
            } else if (basis->size() < std::numeric_limits<int64_t>::max()) {
              if constexpr (QMatrixTypes<dtype_t, int64_t, cindex_t>) {
                return result_t(
                    make_qmatrices<dtype_t, int64_t, cindex_t>(ham, *basis));
              } else {
                return result_t(
                    Error(ErrorType::ValueError, "Invalid type for qmatrix"));
              }
            } else {
              return result_t(
                  Error(ErrorType::ValueError, "Invalid basis size"));
            }
          },
          ham.get_variant_obj(), basis.get_variant_obj(),
          dtype.get_variant_obj());
    }

  public:

    QMatrix() : DTypeObject<detail::qmatrices>(default_value()) {}

    QMatrix(const PauliHamiltonian &ham, const HardcoreBasis &basis,
            const DType dtype)
        : DTypeObject<detail::qmatrices>(create_internals(ham, basis, dtype)) {}

    QMatrix(const QMatrix &) = default;
    QMatrix(QMatrix &&) = default;
    QMatrix &operator=(const QMatrix &) = default;
    QMatrix &operator=(QMatrix &&) = default;

    // TODO switch to using shared pointer for variant
    std::size_t dim() const;
    std::size_t num_coeff() const;
    Array dot(const bool overwrite_out, const Array &coeff, const Array &input,
              Array &output) const;
};

}  // namespace quspin
