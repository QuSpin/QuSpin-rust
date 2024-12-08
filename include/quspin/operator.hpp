// Copyright 2024 Phillip Weinberg
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <complex>
#include <limits>
#include <list>
#include <memory>
#include <quspin/basis/detail/bitbasis/bits.hpp>
#include <quspin/basis/detail/bitbasis/dits.hpp>
#include <quspin/utils/functions.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quspin {

template<class T, size_t N>
class N_body_dit_op {
    //
  private:

    const basis::dit_integer_t lhss;  // size of local
    const int dim;
    std::array<int, N> locs;
    std::vector<T> data;
    std::vector<bool> nonzero;

  public:

    static const int length = N;
    typedef T value_type;

    static int get_power(const int _lhss) {
      int _dim = 1;
      for (size_t i = 0; i < N; i++) {
        _dim *= _lhss;
      }
      return _dim;
    }

    N_body_dit_op(const basis::dit_integer_t _lhss,
                  const std::vector<int>& _locs, const std::vector<T>& _data)
        : lhss(_lhss), dim(get_power(_lhss)) {
      assert(_data.size() == dim * dim);
      assert(_locs.size() == N);
      // copy to contiguous pointers
      data.insert(data.begin(), _data.begin(), _data.end());
      std::copy(_locs.begin(), _locs.end(), locs.begin());

      nonzero.resize(data.size());
      std::transform(data.begin(), data.end(), nonzero.begin(),
                     [](const T value) -> bool { return value != T(0); });
    }

    N_body_dit_op(const basis::dit_integer_t _lhss, int* _locs, T* _data)
        : lhss(_lhss), dim(get_power(_lhss)) {
      // copy to contiguous pointers
      data.insert(data.begin(), _data, _data + dim * dim);
      std::copy(_locs, _locs + N, locs.begin());

      nonzero.resize(data.size());
      std::transform(data.begin(), data.end(), nonzero.begin(),
                     [](const T value) -> bool { return value != T(0); });
    }

    ~N_body_dit_op() {}

    template<typename bitset_t, typename container_t>
    void op(const bitset_t& s, container_t& output) const {
      const int a = basis::get_sub_bitstring(s, locs);
      for (int b = 0; b < dim; ++b) {  // loop over columns
        const int i = dim * a + b;
        if (nonzero[i]) {
          const bitset_t r = basis::set_sub_bitstring(s, b, locs);
          output.push_back(std::make_pair(r, data[i]));
        }
      }
    }

    template<typename bitset_t, typename container_t>
    void op_dagger(const bitset_t& s, container_t& output) const {
      const int a = basis::get_sub_bitstring(s, locs);
      for (int b = 0; b < dim; ++b) {  // loop over rows
        const int i = dim * b + a;
        if (nonzero[i]) {
          const bitset_t r = basis::set_sub_bitstring(s, b, locs);
          output.push_back(std::make_pair(r, conj(data[i])));
        }
      }
    }
};

template<class T, size_t N>
class N_body_bit_op {
    //
  private:

    static const int dim = static_cast<size_t>(integer_pow<2, N>::value);

    std::array<int, N> locs;
    std::array<T, dim * dim> data;
    std::array<bool, dim * dim> nonzero;

  public:

    static const int length = N;
    typedef T value_type;

    N_body_bit_op(const std::vector<int>& _locs, const std::vector<T>& _data) {
      assert(_data.size() == dim * dim);
      assert(_locs.size() == N);
      // copy to contiguous pointers
      std::copy(_data.begin(), _data.end(), data.begin());
      std::copy(_locs.begin(), _locs.end(), locs.begin());

      std::transform(data.begin(), data.end(), nonzero.begin(),
                     [](const T& value) -> bool { return value != T(0); });
    }

    N_body_bit_op(int* _locs, T* _data) {
      // copy to contiguous pointers (already allocated)
      std::copy(_data, _data + dim * dim, data.begin());
      std::copy(_locs, _locs + N, locs.begin());

      std::transform(data.begin(), data.end(), nonzero.begin(),
                     [](const T& value) -> bool { return value != T(0); });
    }

    ~N_body_bit_op() {}

    template<typename bitset_t, typename container_t>
    void op(const bitset_t& s, container_t& output) const {
      const int a = basis::get_sub_bitstring(s, locs);

      for (int b = 0; b < dim; ++b) {  // loop over columns
        const int i = dim * a + b;
        if (nonzero[i]) {
          bitset_t r = basis::set_sub_bitstring(s, b, locs);
          output.push_back(std::make_pair(r, data[i]));
        }
      }
    }

    template<typename bitset_t, typename container_t>
    void op_dagger(const bitset_t& s, container_t& output) const {
      const int a = basis::get_sub_bitstring(s, locs);

      for (int b = 0; b < dim; ++b) {  // loop over rows
        const int i = dim * b + a;
        if (nonzero[i]) {
          bitset_t r = basis::set_sub_bitstring(s, b, locs);
          output.push_back(std::make_pair(r, conj(data[i])));
        }
      }
    }
};

template<typename T>
class operator_string  // generic operator
{
  private:

    const int lhss;          // local hilbert space size for each term
    const int nlocs;         // number of local operators
    std::vector<int> locs;   // number of local operators in
    std::vector<int> perms;  // non-branching operators stored as permutations
    std::vector<int>
        inv_perms;  // non-branching operators dagger stored as permutations
    std::vector<T> datas;      // matrix elements for non-branching operators.
    std::vector<T> inv_datas;  // matrix elements for dagger operator.

  public:

    static const int length = 0;
    typedef T value_type;

    operator_string(const std::vector<int>& _locs,
                    const std::vector<std::vector<int>>& _perms,
                    const std::vector<std::vector<T>>& _datas)
        : lhss(_perms.front().size()), nlocs(_locs.size()) {
      assert(_locs.size() == _perms.size());
      assert(_locs.size() == _datas.size());

      locs.insert(locs.end(), _locs.begin(), _locs.end());

      for (size_t i = 0; i < _locs.size(); i++) {
        std::vector<int> perm(_perms[i].begin(), _perms[i].end()),
            inv_perm(_perms[i].size());
        std::vector<T> data(_datas[i].begin(), _datas[i].end()),
            inv_data(_datas[i].size());

        assert(perm.size() == lhss);
        assert(data.size() == lhss);

        int j = 0;
        for (const int p : perm) {
          assert((p >= 0) && (p < lhss));
          inv_perm[p] = j;
          inv_data[p] = conj(data[j]);
          j++;
        }

        perms.insert(perms.end(), perm.begin(), perm.end());
        inv_perms.insert(inv_perms.end(), inv_perm.begin(), inv_perm.end());

        datas.insert(datas.end(), data.begin(), data.end());
        inv_datas.insert(inv_datas.end(), inv_data.begin(), inv_data.end());
      }
    }

    operator_string(const int _lhss, const int _nlocs, int* _locs, int* _perms,
                    T* _datas)
        : lhss(_lhss), nlocs(_nlocs) {
      // constructor for raw data.
      locs.insert(locs.end(), _locs, _locs + _nlocs);
      perms.insert(perms.end(), _perms, _perms + _lhss * _nlocs);
      datas.insert(datas.end(), _datas, _datas + _lhss * _nlocs);
      inv_perms.resize(perms.size());
      inv_datas.resize(perms.size());

      std::transform(datas.begin(), datas.end(), inv_datas.begin(),
                     [](const T& val) { return conj(val); });

      int* _inv_perms = inv_perms.data();
      for (int i = 0; i < _nlocs; ++i) {
        int* perm = _perms + i * _lhss;
        int* inv_perm = _inv_perms + i * _lhss;
        for (int j = 0; j < _lhss; ++j) {
          inv_perm[perm[j]] = j;
        }
      }
    }

    ~operator_string() {}

    template<typename bitset_t, typename container_t>
    void op(const bitset_t& s, container_t& output) const {
      T m = T(1.0);
      bitset_t r(s);

      bool nonzero = true;
      size_t ptr = 0;

      for (int i = 0; i < nlocs; ++i) {
        const int s_loc = quspin::detail::basis::get_sub_bitstring(r, locs[i]);
        const size_t ind = ptr + s_loc;

        m *= datas[ind];
        r = basis::set_sub_bitstring(r, perms[ind], locs[i]);

        if (m == T(0)) {
          nonzero = false;
          break;
        }

        // shift to next permutation
        ptr += lhss;
      }

      if (nonzero) output.push_back(std::make_pair(r, m));
    }

    template<typename bitset_t, typename container_t>
    void op_dagger(const bitset_t& s, container_t& output) const {
      T m = T(1.0);
      bitset_t r(s);

      bool nonzero = true;
      size_t ptr = inv_perms.size() - lhss;

      for (int i = nlocs - 1; i > -1; --i) {
        const int s_loc = basis::get_sub_bitstring(r, locs[i]);
        const size_t ind = ptr + s_loc;

        m *= inv_datas[ind];
        r = basis::set_sub_bitstring(r, inv_perms[ind], locs[i]);

        if (m == T(0)) {
          nonzero = false;
          break;
        }

        ptr -= lhss;
      }

      if (nonzero) output.push_back(std::make_pair(r, m));
    }
};

}  // namespace quspin

#ifdef QUSPIN_UNIT_TESTS

namespace quspin {  // test cases

typedef basis::bit_set<uint8_t> bs;
typedef basis::dit_set<uint8_t> ds;

// qubits

template class operator_string<double>;
template void operator_string<double>::op<bs>(
    const bs&, std::vector<std::pair<bs, double>>&) const;
template void operator_string<double>::op_dagger<bs>(
    const bs&, std::vector<std::pair<bs, double>>&) const;

template class N_body_bit_op<double, 2>;
template void N_body_bit_op<double, 2>::op<bs>(
    const bs&, std::vector<std::pair<bs, double>>&) const;
template void N_body_bit_op<double, 2>::op_dagger<bs>(
    const bs&, std::vector<std::pair<bs, double>>&) const;

// qudits

template class N_body_dit_op<double, 2>;
template void N_body_dit_op<double, 2>::op<ds>(
    const ds&, std::vector<std::pair<ds, double>>&) const;
template void N_body_dit_op<double, 2>::op_dagger<ds>(
    const ds&, std::vector<std::pair<ds, double>>&) const;

}  // namespace quspin

TEST_SUITE("quspin/operators.h") {
  using namespace quspin;

  TEST_CASE("operator_string.op") {
    operator_string<double>* H;

    // qudits
    std::vector<std::pair<ds, double>> dit_output;
    basis::dit_set<uint8_t> dit_state({0, 1, 2, 1}, 3);

    H = new operator_string<double>({0, 1}, {{0, 1, 2}, {0, 1, 2}},
                                    {{-1.0, 0.0, 1.0}, {-1.0, 0.0, 1.0}});

    H->op(dit_state, dit_output);
    CHECK(dit_output.size() == 0);

    delete H;

    H = new operator_string<double>({0, 2}, {{0, 1, 2}, {0, 1, 2}},
                                    {{-1.0, 0.0, 1.0}, {-1.0, 0.0, 1.0}});

    dit_output.clear();
    H->op(dit_state, dit_output);
    CHECK(dit_output.size() == 1);
    CHECK(dit_output[0].first.to_string() == "0 1 2 1 ");
    CHECK(dit_output[0].second == -1.0);

    delete H;

    H = new operator_string<double>(
        {0, 2, 3}, {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}},
        {{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}});

    dit_output.clear();
    H->op(dit_state, dit_output);
    CHECK(dit_output.size() == 1);
    CHECK(dit_output[0].first.to_string() == "0 1 0 0 ");
    CHECK(dit_output[0].second == 6.0);

    delete H;

    // qubits

    std::vector<std::pair<bs, double>> bit_output;
    basis::bit_set<uint8_t> bit_state({0, 1, 1, 0, 1, 0, 0, 1});

    // \sigma_z \sigma_z
    H = new operator_string<double>({0, 1}, {{0, 1}, {0, 1}},
                                    {{-1.0, 1.0}, {-1.0, 1.0}});
    H->op(bit_state, bit_output);

    CHECK(bit_output.size() == 1);
    CHECK(bit_output[0].first.to_string() == "0 1 1 0 1 0 0 1 ");
    CHECK(bit_output[0].second == -1.0);

    delete H;

    H = new operator_string<double>({0, 3}, {{0, 1}, {0, 1}},
                                    {{-1.0, 1.0}, {-1.0, 1.0}});
    H->op(bit_state, bit_output);

    CHECK(bit_output.size() == 2);
    CHECK(bit_output[1].first.to_string() == "0 1 1 0 1 0 0 1 ");
    CHECK(bit_output[1].second == 1.0);

    H = new operator_string<double>({0, 1, 4}, {{0, 1}, {0, 1}, {1, 0}},
                                    {{-1.0, 1.0}, {-1.0, 1.0}, {0.5, 1.0}});
    bit_output.clear();
    H->op(bit_state, bit_output);

    CHECK(bit_output.size() == 1);
    CHECK(bit_output[0].first.to_string() == "0 1 1 0 0 0 0 1 ");
    CHECK(bit_output[0].second == -1.0);

    delete H;

    H = new operator_string<double>({0, 1, 0}, {{0, 1}, {0, 1}, {1, 0}},
                                    {{-1.0, 1.0}, {-1.0, 1.0}, {0.5, 1.0}});
    bit_output.clear();
    H->op(bit_state, bit_output);

    CHECK(bit_output.size() == 1);
    CHECK(bit_output[0].first.to_string() == "1 1 1 0 1 0 0 1 ");
    CHECK(bit_output[0].second == -0.5);

    delete H;
  }

  TEST_CASE("operator_string.op_dagger") {
    typedef double T;
    operator_string<T>* H;
    std::vector<std::pair<ds, T>> output;
    basis::dit_set<uint8_t> state({2, 10}, 16);

    std::vector<int> perm;
    std::vector<T> data;

    for (int i = 0; i < 16; i++) {
      int ip = (i + 1) % 16;
      perm.push_back(ip);
      data.push_back(std::sqrt(ip));
    }

    H = new operator_string<T>({0, 1}, {perm, perm}, {data, data});

    H->op_dagger(state, output);
    CHECK(output.size() == 1);
    CHECK(output[0].first.to_string() == "1 9 ");
    CHECK(output[0].second == doctest::Approx(std::sqrt(2 * 10)));
  }

  TEST_CASE("N_body_bit_op<double,2>") {
    N_body_bit_op<double, 2>* H;
    std::vector<std::pair<bs, double>> output;
    basis::bit_set<uint8_t> state({0, 1, 1, 0, 1, 0, 0, 1});

    std::vector<double> H_loc = {0.25, 0.0, 0.0,   0.0, 0.0, -0.25, 0.5, 0.0,
                                 0.0,  0.5, -0.25, 0.0, 0.0, 0.0,   0.0, 0.25};

    H = new N_body_bit_op<double, 2>({0, 1}, H_loc);

    H->op(state, output);

    REQUIRE(output.size() == 2);
    CHECK(output[0].first.to_string() == "1 0 1 0 1 0 0 1 ");
    CHECK(output[0].second == 0.5);
    CHECK(output[1].first.to_string() == "0 1 1 0 1 0 0 1 ");
    CHECK(output[1].second == -0.25);

    output.clear();
    H->op_dagger(state, output);

    REQUIRE(output.size() == 2);
    CHECK(output[0].first.to_string() == "1 0 1 0 1 0 0 1 ");
    CHECK(output[0].second == 0.5);

    CHECK(output[1].first.to_string() == "0 1 1 0 1 0 0 1 ");
    CHECK(output[1].second == -0.25);

    delete H;

    H = new N_body_bit_op<double, 2>({0, 3}, H_loc);

    output.clear();
    H->op(state, output);
    H->op_dagger(state, output);
    REQUIRE(output.size() == 2);
    CHECK(output[0].first.to_string() == "0 1 1 0 1 0 0 1 ");
    CHECK(output[0].second == 0.25);
    CHECK(output[1].first.to_string() == "0 1 1 0 1 0 0 1 ");
    CHECK(output[1].second == 0.25);

    delete H;
  }

  TEST_CASE("N_body_dit_op<double,2>") {
    std::vector<double> H_loc = {
        1., 0., 0., 0.,  0., 0., 0., 0., 0., 0.,  0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., -1., 0., 1., 0., 0., 0., 0.,  0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.,  1., 0., 0., 0., 1., 0.,  0., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0.,  0., 0., 0., 1., 0., -1., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0.,  0., 0., 0., 0., 0., 0.,  0., 0., 1.};
    using namespace quspin;
    N_body_dit_op<double, 2>* H;

    // qudits
    std::vector<std::pair<ds, double>> output;
    basis::dit_set<uint8_t> state({0, 1, 2, 1}, 3);

    H = new N_body_dit_op<double, 2>(3, {0, 1}, H_loc);

    H->op(state, output);

    CHECK(output.size() == 1);
    CHECK(output[0].first.to_string() == "1 0 2 1 ");
    CHECK(output[0].second == 1.0);

    delete H;

    H = new N_body_dit_op<double, 2>(3, {0, 2}, H_loc);

    output.clear();
    H->op(state, output);

    REQUIRE(output.size() == 2);
    CHECK(output[0].first.to_string() == "1 1 1 1 ");
    CHECK(output[0].second == 1.0);
    CHECK(output[1].first.to_string() == "0 1 2 1 ");
    CHECK(output[1].second == -1.0);

    delete H;
  }
}

#endif
