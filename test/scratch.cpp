
#include <quspin/basis/grp/hardcore.hpp>
#include <quspin/dtype/dtype.hpp>
#include <quspin/operator.hpp>
#include <quspin/qmatrix/detail/qmatrix.hpp>

namespace qusoin {

class QMatrix : public DTypeObject<detail::qmatrices> {
    static detail::qmatrices default_value() {
      return detail::qmatrices(detail::qmatrix<double, int32_t, uint8_t>());
    }

    template<typename Hamiltonian, typename Basis>
    detail::qmatrices create_internals(const Hamiltonian &ham,
                                       const Basis &basis) {
      const auto &internals = ;
      const auto &basis_states = basis.internals();

      detail::qmatrices result;

      if (basis.size() < std::numeric_limits<int32_t>::max()) {
        result = std::move(std::visit(
            [](const auto &ham, const auto &basis, const auto &dtype) {
              using qmatrix_t = detail::value_type_t<decltype(dtype)>;
              using cindex_t =
                  typename std::decay_t<decltype(ham)>::cindex_type;

              detail::qmatrix<qmatrix_t, int32_t, cindex_t> qmat(ham, basis);
              return qmat;
            },
            ham.internals(), basis.internals(), dtype.internals()));
      } else if (basis.size() < std::numeric_limits<int64_t>::max()) {
        result = std::move(std::visit(
            [](const auto &ham, const auto &basis, const auto &dtype) {
              using qmatrix_t = detail::value_type_t<decltype(dtype)>;
              using cindex_t =
                  typename std::decay_t<decltype(ham)>::cindex_type;

              detail::qmatrix<qmatrix_t, int64_t, cindex_t> qmat(ham, basis);
              return qmat;
            },
            ham.internals(), basis.internals(), dtype.internals()));
      } else {
        throw std::invalid_argument("Invalid basis size");
      }

      return result;
    }

  public:

    QMatrix() : DTypeObject<detail::qmatrices>(default_value()) {}

    QMatrix(const PauliHamiltonian &ham, const HardcoreBasis &basis,
            const DType dtype)
        : DTypeObject<detail::qmatrices>(
              std::move(create_internals(ham, basis))) {}

    QMatrix(const detail::qmatrices &op);
    template<typename T, typename I, typename J>
    QMatrix(const detail::qmatrix<T, I, J> &op);

    std::size_t size() const;
    std::size_t num_coeff() const;
};

}  // namespace qusoin

int main() {}
