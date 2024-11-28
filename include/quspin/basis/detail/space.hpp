// Copyright 2024 Phillip Weinberg
#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
// quspin includes
#include <quspin/basis/detail/bitbasis/bits.hpp>
#include <quspin/basis/detail/bitbasis/dits.hpp>

namespace quspin::details::basis {

// J = {int32, int64}
// I = {uint32, uint64, uint1024, uint16384_t}
// K = {uint8, uint16}

template<typename I, typename J>
class dit_fullspace  // sps > 2
{
  private:

    const int lhss;            // local hilbert space sice
    const J Ns;                // number of states
    const I mask;              // mask for local hilbert space storage.
    const dit_integer_t bits;  // number of bits to store lhss

  public:

    typedef dit_set<I> bitset_t;
    typedef J index_t;
    typedef int norm_t;

    dit_fullspace(const int _lhss, const J _Ns)
        : lhss(_lhss),
          Ns(_Ns),
          mask(constants::mask[_lhss]),
          bits(constants::bits[_lhss]) {}

    dit_fullspace(const dit_fullspace<I, J>& other) = default;

    ~dit_fullspace() {}

    inline J size() const { return Ns; }

    inline J get_Ns() const { return Ns; }

    inline bitset_t get_state(const J index) const {
      return bitset_t(I(Ns - index - 1), lhss, mask, bits);
    }

    inline J get_index(const bitset_t& state) const {
      return Ns - integer<J, I>::cast(state.content) - 1;
    }

    J get_index(const std::vector<dit_integer_t>& state_vector) const {
      bitset_t state(state_vector, lhss);
      return get_index(state);
    }

    static int get_norm(const bitset_t& state) { return 1; }

    static int get_norm(const J index) { return 1; }
};

template<typename I, typename J, typename K>
class dit_subspace  // sps > 2
{
  private:

    const int lhss;            // local hilbert space sice
    const I mask;              // mask for bits
    const dit_integer_t bits;  // number of bits to store lhss

    std::vector<std::pair<I, K>> states;
    std::unordered_map<I, J> index_map;

  public:

    typedef dit_set<I> bitset_t;
    typedef J index_t;
    typedef K norm_t;

    dit_subspace(const int _lhss, const J Ns_est)
        : lhss(_lhss),
          mask(constants::mask[_lhss]),
          bits(constants::bits[_lhss]) {
      states.reserve(Ns_est);
      index_map.reserve(Ns_est * 2);
    }

    ~dit_subspace() {}

    inline J size() const { return states.size(); }

    inline J get_Ns() const { return states.size(); }

    inline size_t nbytes() const {
      return states.size() * sizeof(std::pair<I, K>);
    }

    inline bitset_t get_state(const size_t index) const {
      return bitset_t(std::get<0>(states[index]), lhss, mask, bits);
    }

    J get_index(const std::vector<dit_integer_t>& state_vector) const {
      bitset_t state(state_vector, lhss);
      return (contains(state) ? get_index(state) : -1);
    }

    inline J get_index(const bitset_t& state) const {
      return index_map.at(state.content);
    }

    inline K get_norm(const bitset_t& state) const {
      return get_norm(get_index(state));
    }

    inline K get_norm(const J index) const {
      return std::get<1>(states[index]);
    }

    inline bool contains(const bitset_t& state) const {
      return index_map.contains(state.content);
    }

    void append(const I new_state, const K new_norm) {
      if (index_map.contains(new_state)) {
        states.push_back(std::make_pair(new_state, new_norm));
        index_map[new_state] = states.size();
      }
    }

    void append(const dit_subspace& other) {
      for (const auto& [new_state, new_norm] : other.states) {
        this->append(new_state, new_norm);
      }
    }

    void sort_states() {
      const bool is_sorted = std::is_sorted(
          states.begin(), states.end(),
          [](const std::pair<I, K>& lhs, const std::pair<I, K>& rhs) -> bool {
            return lhs.first < rhs.first;
          });

      if (!is_sorted) {
        std::sort(
            states.begin(), states.end(),
            [](const std::pair<I, K>& lhs, const std::pair<I, K>& rhs) -> bool {
              return lhs.first < rhs.first;
            });

        index_map.clear();
        J index = 0;
        for (const auto& [state, norm] : states) {
          index_map[state] = index++;
        }
      }
    }

    void serialize_states(char* output) const {
      char* states_ptr = (char*)states.data();
      std::copy(states_ptr, states_ptr + nbytes(), output);
    }

    void deserialize_states(char* input, const size_t nbytes) {
      const size_t n_elements = nbytes / sizeof(std::pair<I, K>);

      states.clear();  // clears current data
      index_map.clear();

      states.resize(n_elements);

      std::copy(input, input + nbytes, (char*)states.data());

      J index = 0;
      for (const auto& [state, norm] : states) {
        index_map[state] = index++;
      }
    }
};

template<typename I, typename J>
class bit_fullspace  // sps = 2
{
  private:

    const J Ns;  // total number of states

  public:

    typedef bit_set<I> bitset_t;
    typedef J index_t;
    typedef int norm_t;

    bit_fullspace(const J _Ns, const int _lhss = 2) : Ns(_Ns) {}

    ~bit_fullspace() {}

    inline J size() const { return Ns; }

    inline size_t get_Ns() const { return Ns; }

    inline bitset_t get_state(const J index) const {
      return bitset_t(I(Ns - index - 1));
    }

    inline J get_index(const bitset_t& state) const {
      return Ns - integer<J, I>::cast(state.content) - 1;
    }

    J get_index(const std::vector<dit_integer_t>& state_vector) const {
      bitset_t state(state_vector);
      return get_index(state);
    }

    static int get_norm(const bitset_t& state) { return 1; }

    static int get_norm(const J index) { return 1; }
};

template<typename I, typename J,
         typename K>  // tem[plate argument to turn off hash mapping]
class bit_subspace    // sps = 2
{
  private:

    std::vector<std::pair<I, K>> states;
    // TODO: create ABI to switch on and off the index_map;
    std::unordered_map<I, J> index_map;

  public:

    typedef bit_set<I> bitset_t;
    typedef J index_t;
    typedef K norm_t;

    bit_subspace(const J Ns_est, const int _lhss = 2) {
      states.reserve(Ns_est);
      index_map.reserve(Ns_est * 2);
    }

    ~bit_subspace() {}

    inline J size() const { return states.size(); }

    inline J get_Ns() const { return states.size(); }

    inline size_t nbytes() const {
      return states.size() * sizeof(std::pair<I, K>);
    }

    inline bitset_t get_state(const size_t index) const {
      return bitset_t(std::get<0>(states[index]));
    }

    J get_index(const std::vector<dit_integer_t>& state_vector) const {
      bitset_t state(state_vector);
      return (contains(state) ? get_index(state) : -1);
    }

    inline J get_index(const bitset_t& state) const {
      return index_map.at(state.content);
    }

    inline K get_norm(const bitset_t& state) const {
      return get_norm(get_index(state));
    }

    inline K get_norm(const J index) const {
      return std::get<1>(states[index]);
    }

    inline bool contains(const bitset_t& state) const {
      return index_map.count(state.content) != 0;
    }

    void append(const I new_state, const K new_norm) {
      if (index_map.count(new_state) == 0) {
        states.push_back(std::make_pair(new_state, new_norm));
        index_map[new_state] = states.size();
      }
    }

    void append(const bit_subspace& other) {
      for (const auto& [new_state, new_norm] : other.states) {
        this->append(new_state, new_norm);
      }
    }

    void sort_states() {
      const bool is_sorted = std::is_sorted(
          states.begin(), states.end(),
          [](const std::pair<I, K>& lhs, const std::pair<I, K>& rhs) -> bool {
            return lhs.first < rhs.first;
          });

      if (!is_sorted) {
        std::sort(
            states.begin(), states.end(),
            [](const std::pair<I, K>& lhs, const std::pair<I, K>& rhs) -> bool {
              return lhs.first < rhs.first;
            });

        index_map.clear();
        J index = 0;
        for (const auto& [state, norm] : states) {
          index_map[state] = index++;
        }
      }
    }

    void serialize_states(char* output) const {
      char* states_ptr = (char*)states.data();
      std::copy(states_ptr, states_ptr + nbytes(), output);
    }

    void deserialize_states(char* input, const size_t nbytes) {
      const size_t n_elements = nbytes / sizeof(std::pair<I, K>);

      states.clear();  // clears current data
      index_map.clear();

      states.resize(n_elements);

      std::copy(input, input + nbytes, (char*)states.data());

      J index = 0;
      for (const auto& [state, norm] : states) {
        index_map[state] = index++;
      }
    }
};

using spaces = std::variant<
    dit_fullspace<uint32_t, int32_t>, dit_fullspace<uint64_t, int32_t>,
    dit_subspace<uint32_t, int32_t, int8_t>,
    dit_subspace<uint64_t, int32_t, int8_t>,
    dit_subspace<uint128_t, int32_t, int8_t>,
    dit_subspace<uint256_t, int32_t, int16_t>,
    dit_subspace<uint16384_t, int32_t, int16_t>,
    bit_fullspace<uint32_t, int32_t>, bit_fullspace<uint64_t, int32_t>,
    bit_subspace<uint32_t, int32_t, int8_t>,
    bit_subspace<uint64_t, int32_t, int8_t>,
    bit_subspace<uint128_t, int32_t, int8_t>,
    bit_subspace<uint256_t, int32_t, int16_t>,
    bit_subspace<uint16384_t, int32_t, int16_t>>;

}  // namespace quspin::details::basis
