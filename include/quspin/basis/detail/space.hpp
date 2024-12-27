// Copyright 2024 Phillip Weinberg
#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <quspin/basis/detail/bitbasis/dit_manip.hpp>
#include <quspin/basis/detail/types.hpp>
#include <quspin/detail/default_containers.hpp>
#include <quspin/detail/omp.hpp>
#include <ranges>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quspin::detail::basis {

template<typename bitset_t>
  requires std::unsigned_integral<bitset_t>
class space {
  private:

    std::size_t Ns;
    std::ranges::iota_view<bitset_t> basis_states;

  public:

    using bitset_type = bitset_t;

    space(std::size_t Ns) : Ns(Ns), basis_states(Ns) {}

    space(const space&) = default;
    space& operator=(const space&) = default;

    space(space&&) = default;
    space& operator=(space&&) = default;

    std::size_t size() const { return Ns; }

    bitset_t state_at(const std::size_t index) const {
      return bitset_t(Ns - index - 1);
    }

    std::size_t index(const bitset_t& bits) const {
      assert(bits < Ns);
      return static_cast<std::size_t>(Ns - bits - 1);
    }

    const bitset_t operator[](std::size_t index) const {
      return static_cast<bitset_t>(Ns - index - 1);
    }

    decltype(auto) begin() const {
      return basis_states.rbegin();  // reverse iterator
    }

    decltype(auto) end() const { return basis_states.rend(); }
};

template<typename bitset_t,
         typename map_t = default_map_t<bitset_t, std::size_t>>
class subspace {
  private:

    std::vector<bitset_t> basis_states;
    map_t basis_map;

  public:

    using map_iterator = typename map_t::iterator;
    using const_map_iterator = typename map_t::const_iterator;
    using iterator = typename std::vector<bitset_t>::iterator;
    using const_iterator = typename std::vector<bitset_t>::const_iterator;
    using bitset_type = bitset_t;

    subspace() = default;

    // delete copy constructor and assignment operator
    subspace(const subspace& other) = delete;
    subspace& operator=(const subspace& other) = delete;

    // default move constructor and assignment operator
    subspace(subspace&& other) = default;
    subspace& operator=(subspace&& other) = default;

    std::size_t size() const { return basis_states.size(); }

    bitset_t state_at(const std::size_t index) const {
      return basis_states[index];
    }

    std::pair<const_map_iterator, const_map_iterator> find(
        const bitset_t& bits) const {
      return std::make_pair(basis_map.find(bits), basis_map.end());
    }

    const bitset_t& operator[](std::size_t index) const {
      return basis_states[index];
    }

    bitset_t& operator[](std::size_t index) { return basis_states[index]; }

    void sort() {
      basis_map.clear();
      std::sort(basis_states.begin(), basis_states.end());
      for (std::size_t i = 0; i < basis_states.size(); i++) {
        basis_map[basis_states[i]] = i;
      }
    }

    template<typename Operation>
    void build(Operation&& op, const bitset_t& state0) {
      std::stack<bitset_t, std::vector<bitset_t>> stack;

      if (!basis_map.contains(state0)) {
        basis_map[state0] = basis_states.size();
        basis_states.push_back(state0);
        stack.push(state0);
      }

      while (!stack.empty()) {
        const bitset_t& state = stack.top();
        // stack.pop_back();
        stack.pop();
        const auto& matrix_elements = op(state);
        for (const auto& matrix_element : matrix_elements) {
          const bitset_t& next_state = std::get<bitset_t>(matrix_element);
          if (state != next_state && !basis_map.contains(next_state)) {
            basis_map[next_state] = basis_states.size();
            basis_states.push_back(next_state);
            // stack.emplace_back(next_state);
            stack.push(next_state);
          }
        }
      }

      basis_states.shrink_to_fit();
      sort();
    }

    const_iterator begin() const { return basis_states.begin(); }

    iterator begin() { return basis_states.begin(); }

    const_iterator end() const { return basis_states.end(); }

    iterator end() { return basis_states.end(); }
};

template<typename grp_t, typename bitset_t, typename norm_t,
         typename map_t = default_map_t<bitset_t, std::size_t>>
class symmetric_subspace {
  private:

    grp_t symm_grp;
    std::vector<std::pair<bitset_t, norm_t>> basis_states;
    map_t basis_map;

  public:

    using map_iterator = typename map_t::iterator;
    using const_map_iterator = typename map_t::const_iterator;
    using iterator =
        typename std::vector<std::pair<bitset_t, norm_t>>::iterator;
    using const_iterator =
        typename std::vector<std::pair<bitset_t, norm_t>>::const_iterator;
    using norm_type = norm_t;
    using bitset_type = bitset_t;

    symmetric_subspace(const grp_t& symm_grp) : symm_grp(symm_grp) {}

    // delete copy constructor and assignment operator
    symmetric_subspace(const symmetric_subspace& other) = delete;
    symmetric_subspace& operator=(const symmetric_subspace& other) = delete;

    symmetric_subspace(symmetric_subspace&& other) = default;
    symmetric_subspace& operator=(symmetric_subspace&& other) = default;

    std::size_t size() const { return basis_states.size(); }

    bitset_t state_at(const std::size_t index) const {
      return std::get<bitset_t>(basis_states[index]);
    }

    std::pair<bitset_t, std::complex<double>> get_refstate(
        const bitset_t& bits) const {
      return symm_grp.check_refstate(bits);
    }

    std::pair<const_map_iterator, const_map_iterator> find(
        const bitset_t& bits) const {
      return std::make_pair(basis_map.find(bits), basis_map.end());
    }

    const std::pair<bitset_t, norm_t>& operator[](std::size_t index) const {
      return basis_states[index];
    }

    std::pair<bitset_t, norm_t>& operator[](std::size_t index) {
      return basis_states[index];
    }

    void sort() {
      auto sort_kernel = [](const auto& a, const auto& b) {
        return std::get<bitset_t>(a) < std::get<bitset_t>(b);
      };
      const bool is_sorted =
          std::is_sorted(basis_states.begin(), basis_states.end(), sort_kernel);
      if (is_sorted) {
        return;
      }
      basis_map.clear();
      std::sort(basis_states.begin(), basis_states.end(), sort_kernel);
      for (std::size_t i = 0; i < basis_states.size(); i++) {
        basis_map[std::get<bitset_t>(basis_states[i])] = i;
      }
    }

    template<typename Operation>
    void build(Operation&& op, const bitset_t& state0) {
      std::vector<bitset_t> stack;

      const auto& [ref_state, coeff] = symm_grp.get_refstate(state0);
      const auto& [_, norm] = symm_grp.check_refstate(ref_state);

      if (norm != 0.0 && !basis_map.contains(ref_state)) {
        basis_map[ref_state] = basis_states.size();
        basis_states.emplace_back(ref_state, norm);
        stack.emplace_back(ref_state);
      } else {
        throw std::runtime_error("Invalid initial state");
      }

      using container_t = svector_t<std::pair<bitset_t, norm_t>, 128>;
      svector_t<container_t, 128> new_basis_states_thread(
          omp_get_max_threads());

#pragma omp parallel
#pragma omp single
      {
        while (!stack.empty()) {
          const bitset_t state = stack.back();
          stack.pop_back();

#pragma omp task depend(in : state) firstprivate(state) \
    shared(new_basis_states_thread)
          {
            const auto& matrix_elements = op(state);
            const int tid = omp_get_thread_num();
            auto& new_basis_states = new_basis_states_thread[tid];
            for (const auto& matrix_element : matrix_elements) {
              const bitset_t& next_state = std::get<bitset_t>(matrix_element);
              if (next_state == state) continue;

              const auto& [next_ref_state, next_norm] =
                  symm_grp.check_refstate(next_state);
              if (next_state == next_ref_state && next_norm != 0.0 &&
                  !basis_map.contains(next_state)) {
                new_basis_states.emplace_back(next_state, next_norm);
              }
            }
          }

          if (stack.size() > 0) continue;

#pragma omp taskwait

          for (auto& new_basis_states : new_basis_states_thread) {
            for (const auto& new_basis_state : new_basis_states) {
              if (basis_map.contains(std::get<bitset_t>(new_basis_state))) {
                continue;
              }

              basis_map[std::get<bitset_t>(new_basis_state)] =
                  basis_states.size();
              basis_states.push_back(new_basis_state);
              stack.emplace_back(std::get<bitset_t>(new_basis_state));
            }
            new_basis_states.clear();
          }
        }
      }

      basis_states.shrink_to_fit();
      sort();
    }

    const_iterator begin() const { return basis_states.begin(); }

    const_iterator end() const { return basis_states.end(); }
};

}  // namespace quspin::detail::basis
