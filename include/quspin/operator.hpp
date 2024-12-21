// Copyright 2024 Phillip Weinberg
#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <functional>
#include <iostream>
#include <iterator>
#include <quspin/basis/detail/bitbasis/cast.hpp>
#include <quspin/basis/grp/hardcore.hpp>
#include <quspin/detail/cast.hpp>
#include <quspin/detail/default_containers.hpp>
#include <sstream>
#include <unordered_set>

namespace quspin {

using cdouble = std::complex<double>;

class pauli {
  public:

    enum class OperatorType { X, Y, Z, P, M, N };

    static OperatorType convert(const char c) {
      switch (c) {
        case 'x':
        case 'X':
          return OperatorType::X;
        case 'y':
        case 'Y':
          return OperatorType::Y;
        case 'z':
        case 'Z':
          return OperatorType::Z;
        case '+':
          return OperatorType::P;
        case '-':
          return OperatorType::M;
        case 'n':
        case 'N':
          return OperatorType::N;
        default:
          throw std::invalid_argument("Invalid operator type");
      }
    }

    template<typename bitset_t, typename cindex_t>
    static std::pair<bitset_t, cdouble> apply_op(const bitset_t &state,
                                                 const OperatorType op,
                                                 const cindex_t loc) noexcept {
      const bool n = quspin::detail::basis::integer_cast<uint32_t>(
                         (state >> loc) & bitset_t(1)) &
                     uint32_t(1);
      const double s = 1.0 - 2.0 * n;
      const bool is_X = OperatorType::X == op;
      const bool is_Y = OperatorType::Y == op;
      const bool is_Z = OperatorType::Z == op;
      const bool is_P = OperatorType::P == op;
      const bool is_M = OperatorType::M == op;
      const bool is_N = OperatorType::N == op;
      const bitset_t new_state = state ^ bitset_t(is_X || is_Y || is_P || is_M)
                                             << loc;
      const cdouble result = cdouble(
          is_Z * s + (is_X | ((is_M | is_N) & n) | (is_P & (!n))), s * is_Y);
      return std::make_pair(new_state, result);
    }
};

template<typename cindex_t>
class pauli_operator_string {
  private:

    std::vector<pauli::OperatorType> ops_;
    std::vector<cindex_t> locs_;
    cdouble coeff;

  public:

    pauli_operator_string() = default;
    pauli_operator_string(pauli_operator_string &&) = default;
    pauli_operator_string(const pauli_operator_string &) = default;
    pauli_operator_string &operator=(const pauli_operator_string &) = default;
    pauli_operator_string &operator=(pauli_operator_string &&) = default;

    pauli_operator_string(const std::string &ops,
                          const std::vector<cindex_t> &locs,
                          const cdouble &coeff = 1.0)
        : coeff(coeff) {
      if (ops.size() != locs.size()) {
        throw std::invalid_argument("Invalid operator string");
      }

      this->ops_.resize(ops.size());
      this->locs_.resize(locs.size());

      std::copy(locs.begin(), locs.end(), this->locs_.begin());
      std::transform(ops.begin(), ops.end(), this->ops_.begin(),
                     [](const auto c) { return pauli::convert(c); });
    }

    template<typename bitset_t>
    std::pair<cdouble, bitset_t> operator()(
        const bitset_t &state) const noexcept {
      cdouble result = 1.0;
      bitset_t new_state = state;
      auto op_it = ops_.rbegin();
      auto loc_it = locs_.rbegin();
      for (; op_it != ops_.rend(); ++op_it, ++loc_it) {
        const auto [new_state_, result_] =
            pauli::apply_op(new_state, *op_it, *loc_it);
        new_state = new_state_;
        result *= result_;
      }

      return std::make_pair(result, new_state);
    }
};

template<typename cindex_t, std::size_t num_operators>
class fixed_pauli_operator_string {
  private:

    std::array<pauli::OperatorType, num_operators> ops_;
    std::array<cindex_t, num_operators> locs_;
    cdouble coeff;

  public:

    fixed_pauli_operator_string() = default;
    fixed_pauli_operator_string(const fixed_pauli_operator_string &) = default;
    fixed_pauli_operator_string(fixed_pauli_operator_string &&) = default;
    fixed_pauli_operator_string &operator=(
        const fixed_pauli_operator_string &) = default;
    fixed_pauli_operator_string &operator=(fixed_pauli_operator_string &&) =
        default;

    fixed_pauli_operator_string(const std::string &ops,
                                const std::vector<cindex_t> &locs,
                                const cdouble &coeff = 1.0)
        : coeff(coeff) {
      if (ops.size() != locs.size()) {
        throw std::invalid_argument("Invalid operator string");
      }

      if (locs.size() != num_operators) {
        throw std::invalid_argument("Invalid operator string");
      }

      std::copy(locs.begin(), locs.end(), this->locs_.begin());
      std::transform(ops.begin(), ops.end(), this->ops_.begin(),
                     [](const auto c) { return pauli::convert(c); });
    }

    template<typename bitset_t>
    std::pair<cdouble, bitset_t> operator()(
        const bitset_t &state) const noexcept {
      cdouble result = 1.0;
      bitset_t new_state = state;
      auto op_it = ops_.rbegin();
      auto loc_it = locs_.rbegin();
      for (; op_it != ops_.rend(); ++op_it, ++loc_it) {
        const auto [new_state_, result_] =
            pauli::apply_op(new_state, *op_it, *loc_it);
        new_state = new_state_;
        result *= result_;
      }

      return std::make_pair(result, new_state);
    }
};

template<typename cindex_t>
class pauli_hamiltonian {
  private:

    svector<std::pair<cindex_t, fixed_pauli_operator_string<cindex_t, 1>>, 64>
        ops_1;
    svector<std::pair<cindex_t, fixed_pauli_operator_string<cindex_t, 2>>, 64>
        ops_2;
    svector<std::pair<cindex_t, fixed_pauli_operator_string<cindex_t, 3>>, 64>
        ops_3;
    svector<std::pair<cindex_t, fixed_pauli_operator_string<cindex_t, 4>>, 64>
        ops_4;
    std::vector<std::pair<cindex_t, pauli_operator_string<cindex_t>>> ops_other;

    std::size_t size_ = 0;

  public:

    using cindex_type = cindex_t;

    pauli_hamiltonian() = default;
    pauli_hamiltonian(const pauli_hamiltonian &) = default;
    pauli_hamiltonian(pauli_hamiltonian &&) = default;
    pauli_hamiltonian &operator=(const pauli_hamiltonian &) = default;
    pauli_hamiltonian &operator=(pauli_hamiltonian &&) = default;

    template<typename... Args>
    pauli_hamiltonian(const std::vector<std::tuple<cindex_t, Args...>> &args) {
      for (const auto &[cindex, str, locs, coeff] : args) {
        if (cindex > std::numeric_limits<cindex_t>::max()) {
          throw std::invalid_argument("Invalid cindex");
        }

        auto max_loc = std::max_element(locs.begin(), locs.end());
        if (max_loc != locs.end()) {
          size_ = std::max(size_, static_cast<std::size_t>(*max_loc + 1));
        }

        switch (locs.size()) {
          case 1:
            ops_1.emplace_back(
                cindex, std::move(fixed_pauli_operator_string<cindex_t, 1>(
                            str, locs, coeff)));
            break;
          case 2:
            ops_2.emplace_back(
                cindex, std::move(fixed_pauli_operator_string<cindex_t, 2>(
                            str, locs, coeff)));
            break;
          case 3:
            ops_3.emplace_back(
                cindex, std::move(fixed_pauli_operator_string<cindex_t, 3>(
                            str, locs, coeff)));
            break;
          case 4:
            ops_4.emplace_back(
                cindex, std::move(fixed_pauli_operator_string<cindex_t, 4>(
                            str, locs, coeff)));
            break;
          default:
            ops_other.emplace_back(
                cindex,
                std::move(pauli_operator_string<cindex_t>(str, locs, coeff)));
        }
      }

      std::sort(ops_1.begin(), ops_1.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      std::sort(ops_2.begin(), ops_2.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      std::sort(ops_3.begin(), ops_3.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      std::sort(ops_4.begin(), ops_4.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      std::sort(ops_other.begin(), ops_other.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
    }

    std::size_t size() const noexcept { return size_; }

    std::size_t num_ops() const noexcept {
      return ops_1.size() + ops_2.size() + ops_3.size() + ops_4.size() +
             ops_other.size();
    }

    template<typename bitset_t>
    svector<std::tuple<cdouble, bitset_t, cindex_t>, 256> operator()(
        const bitset_t &state) const noexcept {
      svector<std::tuple<cdouble, bitset_t, cindex_t>, 256> result;
      result.reserve(num_ops());

      for (const auto &[cindex, op_1] : ops_1) {
        const auto [coeff, new_state] = op_1(state);
        if (coeff != 0.0) {
          result.emplace_back(coeff, new_state, cindex);
        }
      }

      for (const auto &[cindex, op_2] : ops_2) {
        const auto [coeff, new_state] = op_2(state);
        if (coeff != 0.0) {
          result.emplace_back(coeff, new_state, cindex);
        }
      }

      for (const auto &[cindex, op_3] : ops_3) {
        const auto [coeff, new_state] = op_3(state);
        if (coeff != 0.0) {
          result.emplace_back(coeff, new_state, cindex);
        }
      }

      for (const auto &[cindex, op_4] : ops_4) {
        const auto [coeff, new_state] = op_4(state);
        if (coeff != 0.0) {
          result.emplace_back(coeff, new_state, cindex);
        }
      }

      for (const auto &[cindex, op] : ops_other) {
        const auto [coeff, new_state] = op(state);
        if (coeff != 0.0) {
          result.emplace_back(coeff, new_state, cindex);
        }
      }

      return result;
    }
};

class PauliHamiltonian {
  private:

    using pauli_hamiltonian_t =
        std::variant<pauli_hamiltonian<uint8_t>, pauli_hamiltonian<uint16_t>>;

    pauli_hamiltonian_t internals_;

    template<typename cindex_t>
    static pauli_hamiltonian<cindex_t> create_typed_internals(
        const std::vector<
            std::tuple<std::size_t, std::string, std::vector<std::size_t>,
                       std::complex<double>>> &ham_list) {
      std::vector<std::tuple<cindex_t, std::string, std::vector<cindex_t>,
                             std::complex<double>>>
          ham_list_typed;
      for (const auto &[cindex, str, locs, coeff] : ham_list) {
        std::vector<cindex_t> locs_typed;
        locs_typed.reserve(locs.size());
        std::transform(
            locs.begin(), locs.end(), std::back_inserter(locs_typed),
            [](const auto loc) { return static_cast<cindex_t>(loc); });
        ham_list_typed.emplace_back(static_cast<cindex_t>(cindex), str,
                                    locs_typed, coeff);
      }

      return pauli_hamiltonian<cindex_t>(ham_list_typed);
    }

    static pauli_hamiltonian_t create_internals(
        const std::vector<
            std::tuple<std::size_t, std::string, std::vector<std::size_t>,
                       std::complex<double>>> &ham_list) {
      std::size_t max_cindex_value = 0;

      std::for_each(ham_list.begin(), ham_list.end(),
                    [&max_cindex_value](const auto &ele) {
                      std::size_t loc_max = *std::max_element(
                          std::get<2>(ele).begin(), std::get<2>(ele).end());
                      max_cindex_value =
                          std::max(max_cindex_value, std::get<0>(ele));
                      max_cindex_value = std::max(max_cindex_value, loc_max);
                    });

      if (max_cindex_value > std::numeric_limits<uint8_t>::max()) {
        return create_typed_internals<uint8_t>(ham_list);

      } else if (max_cindex_value <= std::numeric_limits<uint16_t>::max()) {
        return create_typed_internals<uint16_t>(ham_list);

      } else {
        throw std::invalid_argument("Invalid cindex");
      }
    }

  public:

    PauliHamiltonian(
        const std::vector<
            std::tuple<std::size_t, std::string, std::vector<std::size_t>,
                       std::complex<double>>> &ham_list)
        : internals_(create_internals(ham_list)) {}

    std::size_t size() const {
      return std::visit([](const auto &ham) { return ham.size(); }, internals_);
    }

    pauli_hamiltonian_t internals() const { return internals_; }
};

}  // namespace quspin
