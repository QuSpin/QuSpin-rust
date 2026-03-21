// Copyright 2024 Phillip Weinberg
#pragma once

#include <algorithm>
#include <array>
#include <cinttypes>
#include <concepts>
#include <iostream>
#include <stdexcept>

namespace quspin::detail::basis {

template<std::size_t bits, typename storage_t = unsigned long long>
  requires std::unsigned_integral<storage_t>
class bitset {
  private:

    static constexpr std::size_t word_size = sizeof(storage_t) * 8;
    static constexpr std::size_t num_words =
        bits / word_size + (bits % word_size != 0);

    template<typename F>
    bitset& binary_operator(const bitset& other, F&& op) {
      is_hashed_ = false;
      auto other_begin = other.data_.begin();

      for (auto& word : data_) {
        word = op(word, *other_begin++);
      }

      return *this;
    }

  public:

    using storage_type = storage_t;
    using storage_array_type = std::array<storage_t, num_words>;

  private:

    storage_array_type data_;
    mutable bool is_hashed_ = false;
    mutable std::size_t hash_value_ = 0;

  public:

    template<typename integer_t = std::size_t>
      requires std::integral<integer_t>
    bitset(integer_t value = 0) {
      const storage_t mask = ~storage_t(0);
      const std::size_t input_word_size = sizeof(integer_t) * 8;
      data_.fill(0);
      if constexpr (num_words > 0) {
        if (input_word_size <= word_size) {
          data_.at(0) = static_cast<storage_t>(value);
        } else {
          for (auto& word : data_) {
            word = static_cast<storage_t>(value) & mask;
            value >>= word_size;
          }
        }
      }
    }

    template<typename integer_t = std::size_t>
      requires std::integral<integer_t>
    bitset& operator=(const integer_t value) {
      *this = bitset(value);
      return *this;
    }

    bitset(const bitset& other) = default;
    bitset(bitset&& other) = default;
    bitset& operator=(const bitset& other) = default;
    bitset& operator=(bitset&& other) = default;

    constexpr bool operator==(const bitset& other) const {
      return data_ == other.data_;
    }

    decltype(auto) operator<=>(const bitset& other) const {
      return data_ <=> other.data_;
    }

    bitset& operator>>=(const std::size_t shift) {
      is_hashed_ = false;
      return *this = *this >> shift;
    }

    bitset& operator<<=(const std::size_t shift) {
      is_hashed_ = false;
      return *this = *this << shift;
    }

    bitset& operator&=(const bitset& other) {
      return binary_operator(other, std::bit_and<storage_t>());
      return *this;
    }

    bitset& operator|=(const bitset& other) {
      return binary_operator(other, std::bit_or<storage_t>());
      return *this;
    }

    bitset& operator^=(const bitset& other) {
      return binary_operator(other, std::bit_xor<storage_t>());
      return *this;
    }

    bitset operator~() const {
      bitset out = *this;
      for (auto& word : out.data()) {
        word = ~word;
      }
      return out;
    }

    bitset operator>>(const std::size_t shift) const {
      bitset out;
      const std::size_t word_shift = shift / word_size;
      const std::size_t shift_mod = shift % word_size;
      const std::size_t shift_mod_comp = word_size - shift_mod;

      for (std::size_t idx = word_shift; idx < num_words - 1; idx++) {
        const std::size_t out_idx = idx - word_shift;
        out.data_.at(out_idx) = (data_.at(idx) >> shift_mod) |
                                (data_.at(idx + 1) << shift_mod_comp);
      }
      if (num_words >= word_shift) {
        out.data_.at(num_words - word_shift - 1) =
            (data_.at(num_words - 1) >> shift_mod);
      }

      return out;
    }

    bitset operator<<(const std::size_t shift) const {
      bitset out;

      const std::size_t word_shift = shift / word_size;
      const std::size_t shift_mod = shift % word_size;
      const std::size_t shift_mod_comp = word_size - shift_mod;

      if (word_shift < num_words) {
        out.data_.at(word_shift) = data_.at(0) << shift_mod;
      }

      for (std::size_t out_idx = word_shift + 1; out_idx < num_words;
           out_idx++) {
        const std::size_t idx = out_idx - word_shift;
        out.data_.at(out_idx) = (data_.at(idx - 1) >> shift_mod_comp) |
                                (data_.at(idx) << shift_mod);
      }

      return out;
    }

    bitset operator&(const bitset& other) const {
      bitset out = *this;
      out &= other;
      return out;
    }

    bitset operator|(const bitset& other) const {
      bitset out = *this;
      out |= other;
      return out;
    }

    bitset operator^(const bitset& other) const {
      bitset out = *this;
      out ^= other;
      return out;
    }

    unsigned long long to_ullong() const {
      auto is_zero = [](const auto& word) { return word == 0; };
      if constexpr (num_words > 0) {
        const std::size_t num_output_words =
            word_size / (sizeof(unsigned long long) * 8);
        if (std::all_of(data_.begin(), data_.begin() + num_output_words,
                        is_zero)) {
          unsigned long long out = 0;
          for (std::size_t idx = 0; idx < num_output_words; idx++) {
            out |= static_cast<unsigned long long>(data_.at(idx))
                   << (idx * word_size);
          }
          return out;
        } else {
          throw std::overflow_error("bitset::to_ullong overflow");
        }
      } else {
        return static_cast<storage_t>(0);
      }
    }

    storage_array_type& data() noexcept { return data_; }

    const storage_array_type& data() const noexcept { return data_; }

    std::size_t calc_hash() const {
      if (is_hashed_) {
        return hash_value_;
      } else {
        hash_value_ = 0;
        for (const auto& word : data_) {
          hash_value_ ^= std::hash<storage_t>{}(word);
        }
        is_hashed_ = true;
        return hash_value_;
      }
    }
};

using dit_integer_t = int;

using bits32_t = std::uint32_t;
using bits64_t = std::uint64_t;
using bits128_t = bitset<128>;
using bits256_t = bitset<256>;
using bits512_t = bitset<512>;
using bits1024_t = bitset<1024>;
using bits2048_t = bitset<2048>;
using bits4096_t = bitset<4096>;
using bits8192_t = bitset<8192>;
using bits16384_t = bitset<16384>;

template<typename I>
concept BasisPrimativeTypes =
    std::same_as<I, bits32_t> || std::same_as<I, bits64_t> ||
    std::same_as<I, bits128_t> || std::same_as<I, bits256_t> ||
    std::same_as<I, bits512_t> || std::same_as<I, bits1024_t> ||
    std::same_as<I, bits2048_t> || std::same_as<I, bits4096_t> ||
    std::same_as<I, bits4096_t> || std::same_as<I, bits8192_t> ||
    std::same_as<I, bits16384_t>;

}  // namespace quspin::detail::basis

template<std::size_t bits>
struct std::hash<quspin::detail::basis::bitset<bits>> {
    using bitset = quspin::detail::basis::bitset<bits>;

    std::size_t operator()(const bitset& b) const { return b.calc_hash(); }
};
