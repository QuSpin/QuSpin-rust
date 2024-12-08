// Copyright 2024 Phillip Weinberg
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <queue>
#include <quspin/basis/detail/space.hpp>
#include <quspin/basis/detail/symmetry/grp.hpp>
#include <quspin/operator.hpp>
#include <quspin/utils/functions.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quspin::detail::basis {

template<typename space_t, typename symmetry_t>
class basis {
  public:

    symmetry_t symmetry;  // OK to copy
    std::shared_ptr<space_t> space;

    symmetric_basis(symmetry_t& _symmetry, std::shared_ptr<space_t> _space)
        : symmetry(_symmetry), space(_space) {}

    ~symmetric_basis() {}

    template<typename Container, typename Map, typename J>
    void ref_states_conj(const J i, const Container& col_states,
                         Map& columns) const {
      const typename space_t::bitset_t init_state = space->get_state(i);

      for (const auto& [state, raw_mat_ele] : col_states) {
        if (state == init_state) {
          columns[i] =
              (columns.count(i) != 0 ? columns[i] + raw_mat_ele : raw_mat_ele);
        } else {
          const auto& [ref_state, charater] = symmetry.get_refstate(state);
          const J j = space->get_index(ref_state);
          const auto norm_j = space->get_norm(j);
          const auto norm_i = space->get_norm(i);
          const auto mat_ele =
              raw_mat_ele * conj(charater) * std::sqrt(double(norm_j) / norm_i);
          columns[j] = (columns.count(j) != 0 ? columns[j] + mat_ele : mat_ele);
        }
      }
    }

    template<typename Container, typename Map, typename J>
    void ref_states(const J i, const Container& row_states, Map& rows) const {
      const typename space_t::bitset_t init_state = space->get_state(i);

      for (const auto& [state, raw_mat_ele] : row_states) {
        if (state == init_state) {
          rows[i] = (rows.count(i) != 0 ? rows[i] + raw_mat_ele : raw_mat_ele);
        } else {
          const auto& [ref_state, charater] = symmetry.get_refstate(state);
          J j = space->get_index(ref_state);
          const auto norm_j = space->get_norm(j);
          const auto norm_i = space->get_norm(i);
          const auto mat_ele =
              raw_mat_ele * charater * std::sqrt(double(norm_j) / norm_i);
          rows[j] = (rows.count(j) != 0 ? rows[j] + mat_ele : mat_ele);
        }
      }
    }

    template<typename Term, typename J>
    void calc_rowptr(const Term* terms, const int nterms, J rowptr[]) const {
      const J n_row = space->size();

      using value_type = typename Term::value_type;

      std::vector<std::pair<typename space_t::bitset_t, value_type>> col_states;
      std::unordered_map<J, value_type> columns;

      col_states.reserve(nterms);

      rowptr[0] = 0;
      for (J row = 0; row < n_row; ++row) {
        col_states.clear();
        columns.clear();
        auto state = space->get_state(row);
        // generate action on states
        for (int i = 0; i < nterms; ++i) {
          const auto& term = terms[i];
          term.op_dagger(state, col_states);
        }
        // calculate location of states in basis
        this->ref_states_conj(row, col_states, columns);
        // insert number of non-zeros elements for this row
        rowptr[row] = columns.size();
      }

      J nnz = 0;
      for (J row = 0; row < n_row; ++row) {
        J tmp = rowptr[row];
        rowptr[row] = nnz;
        nnz += tmp;
      }
      rowptr[n_row + 1] = nnz;
    }

    template<typename Term, typename J>
    void calc_matrix(const Term* terms, const int nterms,
                     typename Term::value_type values[], J rowptr[],
                     J indices[]) const {
      using value_type = typename Term::value_type;

      std::vector<std::pair<typename space_t::bitset_t, value_type>> col_states;
      std::vector<std::pair<J, value_type>> sorted_columns;
      std::unordered_map<J, value_type> columns;

      col_states.reserve(nterms);
      sorted_columns.reserve(nterms);

      const J n_row = space->size();

      rowptr[0] = 0;
      for (J row = 0; row < n_row; ++row) {
        col_states.clear();
        columns.clear();
        sorted_columns.clear();

        auto state = space->get_state(row);
        // generate action on states
        for (int i = 0; i < nterms; ++i) {
          const auto& term = terms[i];
          term.op_dagger(state, col_states);
        }
        // calculate location of states in basis
        this->ref_states_conj(row, col_states, columns);

        // sort columns
        sorted_columns.insert(sorted_columns.end(), columns.begin(),
                              columns.end());
        std::sort(sorted_columns.begin(), sorted_columns.end(),
                  [](std::pair<J, typename Term::value_type> lhs,
                     std::pair<J, typename Term::value_type> rhs) -> bool {
                    return lhs.first < rhs.first;
                  });

        // insert data
        J i = rowptr[row];
        for (const auto& [col, nzval] : sorted_columns) {
          indices[i] = col;
          values[i++] = nzval;
        }
      }
    }

    template<typename Term, typename X, typename Y>
    void on_the_fly(const Term* terms, const int nterms, const Y a, const X* x,
                    const Y b, Y* y) const {
      if (b == Y(0.0)) {
        std::fill(y, y + space->size(), 0);
      } else {
        if (b != Y(1))
          std::transform(y, y + space->size(), y,
                         [b](const Y value) -> Y { return b * value; });
      }

      using index_t = typename space_t::index_t;
      using bitset_t = typename space_t::bitset_t;
      using value_type = typename Term::value_type;

      std::vector<std::pair<bitset_t, value_type>> row_states;
      std::unordered_map<index_t, value_type> matrix_ele;

      row_states.reserve(nterms);

      for (typename space_t::index_t row = 0; row < space->size(); ++row) {
        row_states.clear();
        matrix_ele.clear();

        auto state = space->get_state(row);
        // generate action on states
        for (int i = 0; i < nterms; ++i) {
          const auto& term = terms[i];
          term.op_dagger(state, row_states);
        }
        // calculate location of states in basis
        this->ref_states_conj(row, row_states, matrix_ele);

        Y total = 0;
        for (const auto& [col, nzval] : matrix_ele) {
          total += nzval * x[col];
        }
        y[row] += a * total;
      }
    }

    template<typename Iterator>
    void build_subspace(Iterator begin, Iterator end) {
      for (auto it = begin; it != end; it++) {
        const auto norm = symmetry.check_refstate(*it);
        if (!std::isnan(norm)) {
          space->append(*it, static_cast<typename space_t::norm_t>(norm));
        }
      }
    }

    template<typename Term>
    void build_subspace(const Term* terms, const int nterms,
                        const std::vector<int>& seed_vec, const int lhss) {
      using value_type = typename Term::value_type;

      std::vector<std::pair<typename space_t::bitset_t, value_type>> row_states;
      std::queue<typename space_t::bitset_t> stack;

      {
        typename space_t::bitset_t seed_state(seed_vec, lhss);
        const auto& [new_state, n] = symmetry.calc_norm(seed_state);
        const typename space_t::norm_t norm = n;
        if (norm > 0 && !space->contains(new_state)) {
          space->append(new_state.content, norm);
          stack.push(new_state);
        }
      }

      while (!stack.empty()) {
        const auto input_state = stack.front();
        stack.pop();

        row_states.clear();
        for (int i = 0; i < nterms; ++i) {
          const auto& term = terms[i];
          term.op(input_state, row_states);
        }

        for (const auto& [output_state, _] : row_states) {
          const auto& [new_state, n] = symmetry.calc_norm(output_state);
          const typename space_t::norm_t norm = n;
          if (norm > 0 && !space->contains(new_state)) {
            space->append(new_state.content, norm);
            stack.push(new_state);
          }
        }
      }
    }
};

template<typename space_t>
class basis {
  public:

    std::shared_ptr<space_t> space;

    basis(std::shared_ptr<space_t> _space) : space(_space) {}

    ~basis() {}

    template<typename Container, typename Map, typename J>
    void ref_states_conj(const J i, const Container& col_states,
                         Map& columns) const {
      for (const auto& [state, mat_ele] : col_states) {
        const J state_index = space->get_index(state);
        columns[state_index] = (columns.count(state_index) != 0
                                    ? columns[state_index] + conj(mat_ele)
                                    : conj(mat_ele));
      }
    }

    template<typename Container, typename Map, typename J>
    void ref_states(const J i, const Container& row_states, Map& rows) const {
      for (const auto& [state, mat_ele] : row_states) {
        const J state_index = space->get_index(state);
        rows[state_index] =
            (rows.count(state_index) != 0 ? rows[state_index] + mat_ele
                                          : mat_ele);
      }
    }

    template<typename Term, typename J>
    void calc_rowptr(const Term* terms, const int nterms, J rowptr[]) const {
      J n_row = space->size();

      using value_type = typename Term::value_type;

      std::deque<std::pair<typename space_t::bitset_t, value_type>> col_states;
      std::unordered_map<J, value_type> columns;

      rowptr[0] = 0;
      for (J row = 0; row < n_row; ++row) {
        col_states.clear();
        columns.clear();
        auto state = space->get_state(row);
        // generate action on states
        for (int i = 0; i < nterms; ++i) {
          const auto& term = terms[i];
          term.op_dagger(state, col_states);
        }
        // calculate location of states in basis
        this->ref_states_conj(row, col_states, columns);
        // insert number of non-zeros elements for this row
        rowptr[row] = columns.size();
      }

      J nnz = 0;
      for (J row = 0; row < n_row; ++row) {
        J tmp = rowptr[row];
        rowptr[row] = nnz;
        nnz += tmp;
      }
      rowptr[n_row + 1] = nnz;
    }

    template<typename Term, typename J>
    void calc_matrix(const Term* terms, const int nterms,
                     typename Term::value_type values[], J rowptr[],
                     J indices[]) const {
      using value_type = typename Term::value_type;

      std::vector<std::pair<typename space_t::bitset_t, value_type>> col_states;
      std::vector<std::pair<J, value_type>> sorted_columns;
      std::unordered_map<J, value_type> columns;

      col_states.reserve(nterms);
      sorted_columns.reserve(nterms);

      J n_row = space->size();

      rowptr[0] = 0;
      for (J row = 0; row < n_row; ++row) {
        col_states.clear();
        columns.clear();
        sorted_columns.clear();

        auto state = space->get_state(row);
        // generate action on states
        for (int i = 0; i < nterms; ++i) {
          const auto& term = terms[i];
          term.op_dagger(state, col_states);
        }
        // calculate location of states in basis
        this->ref_states_conj(row, col_states, columns);

        // sort columns
        sorted_columns.insert(sorted_columns.end(), columns.begin(),
                              columns.end());
        std::sort(sorted_columns.begin(), sorted_columns.end(),
                  [](std::pair<J, typename Term::value_type> lhs,
                     std::pair<J, typename Term::value_type> rhs) -> bool {
                    return lhs.first < rhs.first;
                  });

        // insert data
        J i = rowptr[row];
        for (const auto& [col, nzval] : sorted_columns) {
          indices[i] = col;
          values[i++] = nzval;
        }
      }
    }

    template<typename Term, typename X, typename Y>
    void on_the_fly(const Term* terms, const int nterms, const Y a, const X* x,
                    const Y b, Y* y) const {
      if (b == Y(0.0)) {
        std::fill(y, y + space->size(), 0);
      } else {
        if (b != Y(1))
          std::transform(y, y + space->size(), y,
                         [b](const Y value) -> Y { return b * value; });
      }

      using value_type = typename Term::value_type;

      std::vector<std::pair<typename space_t::bitset_t, value_type>> row_states;
      std::unordered_map<typename space_t::index_t, value_type> matrix_ele;

      row_states.reserve(nterms);

      for (typename space_t::index_t row = 0; row < space->size(); ++row) {
        row_states.clear();
        matrix_ele.clear();

        auto state = space->get_state(row);
        // generate action on states
        for (int i = 0; i < nterms; ++i) {
          const auto& term = terms[i];
          term.op_dagger(state, row_states);
        }
        // calculate location of states in basis
        this->ref_states_conj(row, row_states, matrix_ele);

        Y total = 0;
        for (const auto& [col, nzval] : matrix_ele) {
          total += nzval * x[col];
        }
        y[row] += a * total;

        matrix_ele.clear();
      }
    }

    template<typename Iterator>
    void build_subspace(Iterator begin, Iterator end) {
      // generate basis states by looping over iterator
      for (auto it = begin; it != end; it++) {
        space->append(*it, static_cast<typename space_t::norm_t>(1));
      }
    }

    template<typename Term>
    void build_subspace(const Term* terms, const int nterms,
                        const std::vector<int>& seed_state, const int lhss) {
      // use list of operators to generate all the possible basis states
      using value_type = typename Term::value_type;

      std::queue<typename space_t::bitset_t> stack;
      stack.push(typename space_t::bitset_t(seed_state, lhss));

      while (!stack.empty()) {
        const auto input_state = stack.front();
        stack.pop();

        std::vector<std::pair<typename space_t::bitset_t, value_type>>
            row_states;
        for (int i = 0; i < nterms; ++i) {
          const auto& term = terms[i];
          term.op(input_state, row_states);
        }

        for (const auto& [new_state, _] : row_states) {
          if (!space->contains(new_state)) {
            space->append(new_state.content, 1);
            stack.push(new_state);
          }
        }
      }
    }
};

}  // namespace quspin::detail::basis

#ifdef QUSPIN_UNIT_TESTS

namespace quspin::detail::basis {

typedef uint8_t I;
typedef int J;
typedef int K;
typedef double T;

typedef bit_set<I> bs;
typedef dit_set<I> ds;
typedef symmetry<bit_perm<I>, perm_bit<I>, bs, T> bit_symm;
typedef symmetry<dit_perm<I>, perm_dit<I>, ds, T> dit_symm;

typedef bit_subspace<I, J, K> bit_space_t;
typedef dit_subspace<I, J, K> dit_space_t;

typedef operator_string<T> Term;

typedef std::vector<std::pair<bs, T>> Container_bits;
typedef std::unordered_map<J, T> Map_bits;

typedef std::vector<std::pair<ds, T>> Container_dits;
typedef std::unordered_map<J, T> Map_dits;

template class symmetric_basis<bit_space_t, bit_symm>;
template void symmetric_basis<bit_space_t, bit_symm>::ref_states_conj<
    Container_bits, Map_bits, J>(const J, const Container_bits&,
                                 Map_bits&) const;
template void
symmetric_basis<bit_space_t, bit_symm>::ref_states<Container_bits, Map_bits, J>(
    const J, const Container_bits&, Map_bits&) const;
template void symmetric_basis<bit_space_t, bit_symm>::calc_rowptr<Term, J>(
    const Term*, const int, J[]) const;
template void symmetric_basis<bit_space_t, bit_symm>::calc_matrix<Term, J>(
    const Term*, const int, T[], J[], J[]) const;
template void symmetric_basis<bit_space_t, bit_symm>::on_the_fly<Term, T, T>(
    const Term* terms, const int nterms, const T, const T[], const T,
    T[]) const;

template class basis<bit_space_t>;
template void basis<bit_space_t>::ref_states_conj<Container_bits, Map_bits, J>(
    const J, const Container_bits&, Map_bits&) const;
template void basis<bit_space_t>::ref_states<Container_bits, Map_bits, J>(
    const J, const Container_bits&, Map_bits&) const;
template void basis<bit_space_t>::calc_rowptr<Term, J>(const Term*, const int,
                                                       J[]) const;
template void basis<bit_space_t>::calc_matrix<Term, J>(const Term*, const int,
                                                       T[], J[], J[]) const;
template void basis<bit_space_t>::on_the_fly<Term, T, T>(const Term* terms,
                                                         const int nterms,
                                                         const T, const T[],
                                                         const T, T[]) const;

template class symmetric_basis<dit_space_t, dit_symm>;
template void symmetric_basis<dit_space_t, dit_symm>::ref_states_conj<
    Container_dits, Map_dits, J>(const J, const Container_dits&,
                                 Map_dits&) const;
template void
symmetric_basis<dit_space_t, dit_symm>::ref_states<Container_dits, Map_dits, J>(
    const J, const Container_dits&, Map_dits&) const;
template void symmetric_basis<dit_space_t, dit_symm>::calc_rowptr<Term, J>(
    const Term*, const int, J[]) const;
template void symmetric_basis<dit_space_t, dit_symm>::calc_matrix<Term, J>(
    const Term*, const int, T[], J[], J[]) const;
template void symmetric_basis<dit_space_t, dit_symm>::on_the_fly<Term, T, T>(
    const Term* terms, const int nterms, const T, const T[], const T,
    T[]) const;

template class basis<dit_space_t>;
template void basis<dit_space_t>::ref_states_conj<Container_dits, Map_dits, J>(
    const J, const Container_dits&, Map_dits&) const;
template void basis<dit_space_t>::ref_states<Container_dits, Map_dits, J>(
    const J, const Container_dits&, Map_dits&) const;
template void basis<dit_space_t>::calc_rowptr<Term, J>(const Term*, const int,
                                                       J[]) const;
template void basis<dit_space_t>::calc_matrix<Term, J>(const Term*, const int,
                                                       T[], J[], J[]) const;
template void basis<dit_space_t>::on_the_fly<Term, T, T>(const Term* terms,
                                                         const int nterms,
                                                         const T, const T[],
                                                         const T, T[]) const;

}  // namespace quspin::detail::basis

TEST_SUITE("quspin/basis/basis.h") {}

#endif
