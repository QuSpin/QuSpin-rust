
#include <boost/container/small_vector.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <quspin/array/detail/array.hpp>
#include <quspin/basis/detail/bitbasis/dit_perm.hpp>
#include <quspin/basis/detail/space.hpp>
#include <quspin/basis/detail/symmetry/grp.hpp>
#include <quspin/basis/detail/symmetry/single_grp_element.hpp>
#include <quspin/qmatrix/detail/qmatrix.hpp>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quspin::detail::basis {

template<typename Operation, typename grp_t, typename bitset_t, typename norm_t,
         typename indptr_t>
void calc_rowptr(Operation &&op,
                 const symmetric_subspace<grp_t, bitset_t, norm_t> &basis,
                 indptr_t rowptr) {
  using indptr_value_type = typename indptr_t::value_type;
  assert(rowptr.size() == basis.size() + 1);

  auto it = rowptr.begin();
  for (const auto &[state, norm] : basis) {
    const auto &new_states = op(state);
    const auto &matrix_elements = basis.get_matrix_elements(norm, new_states);
    *it = static_cast<indptr_value_type>(matrix_elements.size());
    it++;
  }

  indptr_value_type sum = 0;
  for (auto &val : rowptr) {
    const indptr_value_type temp = val;
    val = sum;
    sum += temp;
  }
}

template<typename Operation, typename grp_t, typename bitset_t, typename norm_t,
         typename indptr_t, typename cindices_t, typename data_t>
void calc_mat_element(
    Operation &&op, const symmetric_subspace<grp_t, bitset_t, norm_t> &basis,
    quspin::detail::qmatrix<indptr_t, cindices_t, data_t> &qmat) {
  for (const auto &[state, norm] : basis) {
    const auto &new_states = op(state);
    const auto &matrix_elements = basis.get_matrix_elements(norm, new_states);

    std::sort(matrix_elements.begin(), matrix_elements.end(),
              [](const auto &a, const auto &b) {
                return std::get<2>(a) < std::get<2>(b) ||
                       (std::get<2>(a) == std::get<2>(b) &&
                        std::get<1>(a) < std::get<1>(b));
              });

    const std::size_t row = std::distance(basis.begin(), state);
    auto row_begin = qmat.row_begin(row);
    auto row_end = qmat.row_end(row);

    assert(std::distance(row_begin, row_end) == matrix_elements.size());
    auto matrix_iter = matrix_elements.begin();
    for (auto qmat_iter = qmat.row_begin(row); qmat_iter != qmat.row_end(row);
         qmat_iter++, matrix_iter++) {
      *qmat_iter = *matrix_iter;
    }
  }
}

}  // namespace quspin::detail::basis

int test_basis() {
  using namespace quspin::detail::basis;
  using namespace boost::container;

  using bitset_t = uint64_t;
  using norm_t = int8_t;
  using lattice_grp_t = grp_element<perm_dit_locations<bitset_t>, bitset_t>;
  using local_grp_t = grp_element<perm_dit_mask<bitset_t>, bitset_t>;

  const int L = 26;
  const int q = 0;
  const int z = 0;

  std::vector<std::complex<double>> chars;
  std::vector<std::vector<int>> perms;

  const int lhss = 2;
  const bitset_t mask = (~bitset_t()) >> (bit_info<bitset_t>::bits - L);

  std::vector<lattice_grp_t> lattice_elements;
  std::vector<local_grp_t> local_elements;

  local_elements.emplace_back(std::complex<double>(z), mask);

  for (int i = 0; i < L; i++) {
    std::vector<int> perm;
    for (int j = 0; j < L; j++) {
      perm.push_back((i + j) % L);
    }
    const auto &grp_char =
        std::exp(std::complex<double>(0.0, (2.0 * M_PI * i * q) / L));
    lattice_elements.emplace_back(grp_char, lhss, perm);
  }

  grp<lattice_grp_t, local_grp_t, grp_result<bitset_t>> my_grp(lattice_elements,
                                                               local_elements);

  constexpr std::size_t ns_est = binom(L, L / 2);
  subspace<bitset_t> my_subspace(ns_est);

  std::array<double, 16> mat = {1.0, 0.0, 0.0,  0.0, 0.0, -1.0, 0.5, 0.0,
                                0.0, 0.5, -1.0, 0.0, 0.0, 0.0,  0.0, 1.0};

  auto calc = [L, mat](const auto &state) {
    using bitset_t = std::decay_t<decltype(state)>;
    using input_t = std::tuple<double, bitset_t, uint8_t>;
    small_vector<input_t, 256> new_states;
    dit_manip<2> manip;

    for (std::size_t i = 0; i < L; i++) {
      const std::size_t j = (i + 1) % L;
      const std::size_t loc_index = manip.get_sub_bitstring(state, {i, j});

      auto begin = mat.begin() + 4 * loc_index;
      const auto end = begin + 4;

      for (; begin != end; begin++) {
        const double data = *begin;
        const bitset_t new_state =
            manip.set_sub_bitstring(state, loc_index, {i, j});

        if (std::abs(data) >= std::numeric_limits<double>::epsilon()) {
          new_states.emplace_back(data, new_state, static_cast<uint8_t>(i));
        }
      }
    }
    return new_states;
  };

  dit_manip<2> manip;

  bitset_t s0 = 0;
  for (int i = 0; i < L / 2; i++) {
    s0 = manip.set_sub_bitstring(s0, 1, i);
  }

  my_subspace.build(calc, s0);
  std::cout << my_subspace.size() << std::endl;
  // for(const auto&state : my_subspace) {
  //   std::cout << manip.to_string(state, L) << std::endl;
  // }
  // quspin::detail::qmatrix<double, int32_t, uint8_t> op(calc, my_subspace);

  return 0;
}

int main() {
  test_basis();

  return 0;
}
