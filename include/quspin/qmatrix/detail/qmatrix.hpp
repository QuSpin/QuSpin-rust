// Copyright 2024 Phillip Weinberg
#pragma once

#include <boost/container/small_vector.hpp>
#include <iterator>
#include <quspin/array/detail/array.hpp>
#include <quspin/basis/detail/space.hpp>
#include <quspin/detail/omp.hpp>
#include <quspin/detail/type_concepts.hpp>
#include <quspin/dtype/detail/dtype.hpp>
#include <type_traits>
#include <variant>
#include <vector>

namespace quspin { namespace detail {

template<typename T>
void atomic_iadd(std::complex<T> &lhs, const std::complex<T> &rhs) {
  T *lhs_ptr = reinterpret_cast<T *>(&lhs);
  const T *rhs_ptr = reinterpret_cast<const T *>(&rhs);

#pragma omp atomic
  lhs_ptr[0] += rhs_ptr[0];

#pragma omp atomic
  lhs_ptr[1] += rhs_ptr[1];
}

template<typename T>
void atomic_iadd(T &lhs, const T &rhs) {
#pragma omp atomic
  lhs += rhs;
}

template<typename forward_iterator_t>
class row_iterator_value {
  private:

    forward_iterator_t begin_;
    forward_iterator_t end_;

  public:

    row_iterator_value(forward_iterator_t begin, forward_iterator_t end)
        : begin_(begin), end_(end) {}

    forward_iterator_t begin() const { return begin_; }

    forward_iterator_t end() const { return end_; }
};

template<typename qmatrix_t>
class qmatrix_row_iterator {
  private:

    std::size_t row_;
    const qmatrix_t &qmat_;

  public:

    qmatrix_row_iterator(const std::size_t row, const qmatrix_t &qmat)
        : row_(row), qmat_(qmat) {}

    qmatrix_row_iterator(const qmatrix_row_iterator &) = default;
    qmatrix_row_iterator &operator=(const qmatrix_row_iterator &) = default;

    decltype(auto) operator*() {
      return row_iterator_value(qmat_.row_begin(row_), qmat_.row_end(row_));
    }

    std::size_t row() const { return row_; }

    qmatrix_row_iterator &operator++() {
      row_++;
      return *this;
    }

    qmatrix_row_iterator operator++(int) {
      qmatrix_row_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    bool operator==(const qmatrix_row_iterator &other) const {
      return row_ == other.row_;
    }

    bool operator!=(const qmatrix_row_iterator &other) const {
      return !(*this == other);
    }
};

template<typename T, typename I, typename J>
  requires QMatrixTypes<T, I, J>
class qmatrix : public typed_object<T> {
  private:

    std::size_t dim_;
    std::size_t num_coeff_;
    std::vector<I> indptr_;
    std::vector<std::tuple<T, I, J>> data_;

    template<typename Operation, typename grp_t, typename bitset_t,
             typename norm_t, typename map_t>
    static decltype(auto) calculate_row(
        Operation &&op, const std::size_t row_index,
        const basis::symmetric_subspace<grp_t, bitset_t, norm_t, map_t>
            &basis) {
      boost::container::small_vector<std::tuple<T, I, J>, 256> new_data;

      const auto &[state, norm] = basis[row_index];
      const auto &new_matrix_elements = op(state);
      for (const auto &[data, new_state, cindex] : new_matrix_elements) {
        const auto &[ref_state, grp_char] = basis.get_refstate(new_state);

        const auto &[map_it, end_map_it] = basis.find(ref_state);

        if (map_it == end_map_it) continue;

        const I index = std::get<1>(*map_it);

        const double new_norm = std::get<norm_t>(basis[index]);
        const T matrix_element = quspin::detail::cast<T>(
            data * grp_char * std::sqrt(new_norm / norm));

        auto element_finder = [index, cindex](const auto &elem) {
          return std::get<I>(elem) == index && std::get<J>(elem) == cindex;
        };
        auto find_it =
            std::find_if(new_data.begin(), new_data.end(), element_finder);

        if (find_it != new_data.end()) {
          std::get<T>(*find_it) += matrix_element;
        } else {
          new_data.emplace_back(matrix_element, index, cindex);
        }
      }

      return new_data;
    }

    template<typename Operation, typename bitset_t, typename map_t>
    static decltype(auto) calculate_row(
        Operation &&op, const std::size_t row_index,
        const basis::subspace<bitset_t, map_t> &basis) {
      boost::container::small_vector<std::tuple<T, I, J>, 256> new_data;

      const auto &state = basis[row_index];
      const auto &new_matrix_elements = op(state);

      new_data.reserve(new_matrix_elements.size());
      for (const auto &[data, new_state, cindex] : new_matrix_elements) {
        const auto &[map_it, end_map_it] = basis.find(new_state);

        if (map_it == end_map_it) continue;
        const I index = std::get<1>(*map_it);
        const T matrix_element = quspin::detail::cast<T>(data);

        auto element_finder = [index, cindex](const auto &elem) {
          return std::get<I>(elem) == index && std::get<J>(elem) == cindex;
        };
        auto find_it =
            std::find_if(new_data.begin(), new_data.end(), element_finder);

        if (find_it != new_data.end()) {
          std::get<T>(*find_it) += matrix_element;
        } else {
          new_data.emplace_back(matrix_element, index, cindex);
        }
      }
      return new_data;
    }

    template<typename Operation, typename bitset_t>
    static decltype(auto) calculate_row(Operation &&op,
                                        const std::size_t row_index,
                                        const basis::space<bitset_t> &basis) {
      boost::container::small_vector<std::tuple<T, I, J>, 256> new_data;

      const auto &[state, norm] = basis[row_index];
      const auto &new_matrix_elements = op(state);

      new_data.reserve(new_matrix_elements.size());
      for (const auto &[data, new_state, cindex] : new_matrix_elements) {
        const I index = basis.index(new_state);
        const T matrix_element = quspin::detail::cast<T>(data);

        auto element_finder = [index, cindex](const auto &elem) {
          return std::get<I>(elem) == index && std::get<J>(elem) == cindex;
        };
        auto find_it =
            std::find_if(new_data.begin(), new_data.end(), element_finder);

        if (find_it != new_data.end()) {
          std::get<T>(*find_it) += matrix_element;
        } else {
          new_data.emplace_back(matrix_element, index, cindex);
        }
      }
      return new_data;
    }

  public:

    qmatrix() = default;

    qmatrix(std::vector<I> &&indptr, std::vector<std::tuple<T, I, J>> &&data)
        : indptr_(std::move(indptr)), data_(std::move(data)) {
      dim_ = indptr_.size() - 1;
      num_coeff_ = 0;
      for (const auto &ele : data_) {
        num_coeff_ = std::max(num_coeff_, std::get<J>(ele));
      }
      if (!has_sorted_indices()) {
        sort_indices();
      }
    }

    template<typename Operation, typename basis_t>
    qmatrix(Operation &&op, const basis_t &basis) {
      using bitset_t = typename basis_t::bitset_type;
      using invoke_t = std::invoke_result_t<Operation, bitset_t>;
      using element_t =
          std::decay_t<decltype(*std::declval<invoke_t>().begin())>;
      static_assert(std::is_same_v<element_t, std::tuple<T, bitset_t, J>>,
                    "Operation must return a tuple of type (T, I, J)");

      indptr_.resize(basis.size() + 1);
      dim_ = basis.size();

      if (omp_get_max_threads() > 1) {
#pragma omp parallel shared(op, num_coeff_, data_, indptr_, basis) \
    firstprivate(dim_) default(none)
        {
#pragma omp for
          for (std::size_t row_index = 0; row_index < dim_; ++row_index) {
            const auto &new_data = calculate_row(op, row_index, basis);
            indptr_[row_index] = new_data.size();
          }
#pragma omp barrier

#pragma omp single nowait
          {
            std::size_t sum = 0;
            for (std::size_t i = 0; i <= dim_; ++i) {
              const std::size_t temp = indptr_[i];
              indptr_[i] = sum;
              sum += temp;
            }
            data_.resize(indptr_.back());
          }
#pragma omp barrier

#pragma omp for
          for (std::size_t row_index = 0; row_index < dim_; ++row_index) {
            auto new_data = calculate_row(op, row_index, basis);

            std::sort(new_data.begin(), new_data.end(),
                      [](const auto &a, const auto &b) {
                        return std::get<I>(a) < std::get<I>(b) ||
                               (std::get<I>(a) == std::get<I>(b) &&
                                std::get<J>(a) < std::get<J>(b));
                      });

            auto data_start = data_.begin() + indptr_[row_index];
            std::move(new_data.begin(), new_data.end(), data_start);
          }
#pragma omp barrier

#pragma omp for reduction(max : num_coeff_)
          for (const auto &ele : data_) {
            const std::size_t coeff_index = std::get<J>(ele);
            num_coeff_ = std::max(num_coeff_, coeff_index);
          }
        }
      } else {
        num_coeff_ = 0;
        indptr_.push_back(data_.size());
        for (std::size_t row_index = 0; row_index < dim_; ++row_index) {
          auto new_data = calculate_row(op, row_index, basis);
          std::sort(new_data.begin(), new_data.end(),
                    [](const auto &a, const auto &b) {
                      return std::get<I>(a) < std::get<I>(b) ||
                             (std::get<I>(a) == std::get<I>(b) &&
                              std::get<J>(a) < std::get<J>(b));
                    });

          for (const auto &ele : new_data) {
            auto begin_it = data_.begin() + indptr_.back();
            auto find_it =
                std::find_if(begin_it, data_.end(), [ele](const auto &data) {
                  return std::get<I>(data) == std::get<I>(ele) &&
                         std::get<J>(data) == std::get<J>(ele);
                });

            if (find_it == data_.end()) {
              data_.push_back(std::move(ele));
            } else {
              std::get<T>(*find_it) += std::get<T>(ele);
            }

            const std::size_t cindex = std::get<J>(ele);
            num_coeff_ = std::max(num_coeff_, cindex);
          }
          indptr_.push_back(data_.size());
        }
      }
    }

    qmatrix(const qmatrix &) = delete;
    qmatrix &operator=(const qmatrix &) = delete;

    qmatrix(qmatrix &&) = default;
    qmatrix &operator=(qmatrix &&) = default;

    void sort_indices() {
      auto comp = [](const auto &lhs, const auto &rhs) {
        return std::get<2>(lhs) < std::get<2>(rhs) ||
               (std::get<2>(lhs) == std::get<2>(rhs) &&
                std::get<1>(lhs) < std::get<1>(rhs));
      };
      for (auto &row_iter : *this) {
        std::sort(row_iter.begin(), row_iter.end(), comp);
      }
    }

    bool has_sorted_indices() const {
      auto comp = [](const auto &lhs, const auto &rhs) {
        return std::get<1>(lhs) < std::get<1>(rhs) ||
               (std::get<1>(lhs) == std::get<1>(rhs) &&
                std::get<2>(lhs) < std::get<2>(rhs));
      };
      for (auto &row_iter : *this) {
        const bool is_row_sorted =
            std::is_sorted(row_iter.begin(), row_iter.end(), comp);
        if (!is_row_sorted) {
          return false;
        }
      }
      return true;
    }

    template<typename Container, typename K>
    void _dot_1d(const bool overwrite_out, const Container &coeff,
                 const array<K> input, array<K> output) {
#pragma omp parallel for firstprivate(coeff)
      for (auto row_it : *this) {
        Container sum(coeff.size());

        for (const auto &[data, index, cindex] : row_it) {
          std::array<size_t, 1> col_index = {index};
          sum[cindex] += data * input[col_index];
        }

        std::array<size_t, 1> row_index = {row_it.row()};
        output[row_index] =
            std::inner_product(sum.begin(), sum.end(), coeff.begin(),
                               (overwrite_out ? K(0) : output[row_index]));
      }
    }

    template<typename Container, typename K>
    void _dot_2d(const bool overwrite_out, const Container &coeff,
                 const array<K> input, array<K> output) {
      if (output.strides(0) > output.strices(1)) {
        if (overwrite_out) {
#pragma omp parallel for
          for (std::size_t row_index = 0; row_index < dim(); ++row_index) {
            for (std::size_t vec = 0; vec < input.shape(1); ++vec) {
              std::array<std::size_t, 2> index = {row_index, vec};
              output[index] = K();
            }
          }
        }
#pragma omp parallel for firstprivate(coeff)
        for (auto row_it : *this) {
          for (const auto &[data, index, cindex] : row_it) {
            for (std::size_t vec = 0; vec < input.shape(1); ++vec) {
              const K coeff_data = coeff[static_cast<std::size_t>(cindex)];
              std::array<size_t, 2> in_index = {index, vec};
              std::array<size_t, 2> out_index = {row_it.row(), vec};
              output[out_index] += data * input[in_index] * coeff_data;
            }
          }
        }
      } else {
        if (overwrite_out) {
#pragma omp parallel for
          for (std::size_t vec = 0; vec < input.shape(1); ++vec) {
            for (std::size_t row_index = 0; row_index < dim(); ++row_index) {
              std::array<std::size_t, 2> index = {row_index, vec};
              output[index] = K();
            }
          }
        }
#pragma omp parallel for firstprivate(coeff)
        for (auto row_it : *this) {
          for (std::size_t vec = 0; vec < input.shape(1); ++vec) {
            for (const auto &[data, index, cindex] : row_it) {
              const K coeff_data = coeff[static_cast<std::size_t>(cindex)];
              std::array<size_t, 2> in_index = {index, vec};
              std::array<size_t, 2> out_index = {row_it.row(), vec};
              output[out_index] += data * input[in_index] * coeff_data;
            }
          }
        }
      }
    }

    template<typename Container, PrimativeTypes K>
      requires PrimativeTypes<K> && std::convertible_to<upcast_t<T, K>, K>
    void dot(const bool overwrite_out, const Container &coeff,
             const array<K> input, array<K> output) {
      if (coeff.size() != num_coeff()) {
        throw std::runtime_error("Invalid number of coefficients");
      }

      if (dim() != output.shape(0)) {
        throw std::runtime_error("Invalid output shape");
      }

      if (output.shape() != input.shape()) {
        throw std::runtime_error("Invalid input shape");
      }

      if (output.ndim() == 1) {
        _dot_1d(overwrite_out, coeff, input, output);
      } else if (output.ndim() == 2) {
        _dot_2d(overwrite_out, coeff, input, output);
      } else {
        throw std::runtime_error("Input must be 1D or 2D");
      }
    }

    template<typename Container, typename K>
    void _dot_transpose_1d(const K overwrite_out, const Container &coeff,
                           const array<K> input, array<K> output) {
      if (overwrite_out) {
#pragma omp parallel for
        for (std::size_t row_index = 0; row_index < dim(); ++row_index) {
          std::array<std::size_t, 1> index = {row_index};
          output[index] = overwrite_out ? K(0) : output[index];
        }
      }

#pragma omp parallel for firstprivate(coeff)
      for (const auto &row_it : *this) {
        for (const auto &[data, index, cindex] : row_it) {
          std::array<size_t, 1> col_index = {index};
          std::array<size_t, 1> row_index = {row_it.row()};
          atomic_iadd(output[row_index],
                      data * input[col_index] * coeff[cindex]);
        }
      }
    }

    template<typename Container, typename K>
    void _dot_transpose_2d(const bool overwrite_out, const Container &coeff,
                           const array<K> input, array<K> output) {
      if (output.strides(0) > output.strices(1)) {
        if (overwrite_out) {
#pragma omp parallel for
          for (std::size_t row_index = 0; row_index < dim(); ++row_index) {
            for (std::size_t vec = 0; vec < input.shape(1); ++vec) {
              std::array<std::size_t, 2> index = {row_index, vec};
              output[index] = K();
            }
          }
        }
#pragma omp parallel for firstprivate(coeff)
        for (auto row_it : *this) {
          for (const auto &[data, index, cindex] : row_it) {
            for (std::size_t vec = 0; vec < input.shape(1); ++vec) {
              const K coeff_data = coeff[static_cast<std::size_t>(cindex)];
              std::array<std::size_t, 2> in_index = {row_it.row(), vec};
              std::array<std::size_t, 2> out_index = {index, vec};
              atomic_iadd(output[out_index],
                          data * input[in_index] * coeff_data);
            }
          }
        }
      } else {
        if (overwrite_out) {
#pragma omp parallel for
          for (std::size_t vec = 0; vec < input.shape(1); ++vec) {
            for (std::size_t row_index = 0; row_index < dim(); ++row_index) {
              std::array<std::size_t, 2> index = {row_index, vec};
              output[index] = K();
            }
          }
        }
#pragma omp parallel for firstprivate(coeff)
        for (auto row_it : *this) {
          for (std::size_t vec = 0; vec < input.shape(1); ++vec) {
            for (const auto &[data, index, cindex] : row_it) {
              const K coeff_data = coeff[static_cast<std::size_t>(cindex)];
              std::array<std::size_t, 2> in_index = {row_it.row(), vec};
              std::array<std::size_t, 2> out_index = {index, vec};
              atomic_iadd(output[out_index],
                          data * input[in_index] * coeff_data);
            }
          }
        }
      }
    }

    template<typename Container, PrimativeTypes K>
      requires PrimativeTypes<K> && std::convertible_to<upcast_t<T, K>, K>
    void dot_transpose(const bool overwrite_out, const Container &coeff,
                       const array<K> input, array<K> output) {
      if (coeff.size() != num_coeff()) {
        throw std::runtime_error("Invalid number of coefficients");
      }

      if (dim() != output.shape(0)) {
        throw std::runtime_error("Invalid output shape");
      }

      if (output.shape() != input.shape()) {
        throw std::runtime_error("Invalid input shape");
      }

      if (output.ndim() == 1) {
        _dot_transpose_1d(overwrite_out, coeff, input, output);
      } else if (output.ndim() == 2) {
        _dot_transpose_2d(overwrite_out, coeff, input, output);
      } else {
        throw std::runtime_error("Output must be 1D or 2D");
      }
    }

    qmatrix operator+(const qmatrix &rhs) const;
    qmatrix operator-(const qmatrix &rhs) const;

    std::size_t dim() const { return dim_; }

    std::size_t num_coeff() const { return num_coeff_; }

    decltype(auto) begin() const { return qmatrix_row_iterator(0, *this); }

    decltype(auto) end() const { return qmatrix_row_iterator(dim_, *this); }

    decltype(auto) row_bounds(const std::size_t row) const {
      assert(row < dim_);
      auto begin = data_.begin() + indptr_[row];
      auto end = data_.begin() + indptr_[row + 1];
      return std::make_pair(begin, end);
    }
};

template<typename Op, typename T, typename I, typename J>
  requires QMatrixTypes<T, I, J>
qmatrix<T, I, J> binary_op(Op &&op, const qmatrix<T, I, J> &lhs,
                           const qmatrix<T, I, J> &rhs) {
  static_assert(std::is_invocable_v<Op, T, T>,
                "Incompatible Operator with input types");
  static_assert(std::is_same_v<T, std::decay_t<std::invoke_result_t<Op, T, T>>>,
                "Incompatible output type");

  assert(lhs.dim() == rhs.dim());

  std::vector<I> indptr;
  std::vector<std::tuple<T, I, J>> out_data;

  auto lt = [](const auto &lhs, const auto &rhs) {
    return std::get<I>(lhs) < std::get<I>(rhs) ||
           (std::get<I>(lhs) == std::get<I>(rhs) &&
            std::get<J>(lhs) < std::get<J>(rhs));
  };

  const int num_threads = omp_get_max_threads();

  if (num_threads > 1) {
    indptr.resize(lhs.dim() + 1);
    indptr[0] = 0;
#pragma omp parallel firstprivate(op, lt) shared(indptr, lhs, rhs) default(none)
    {
#pragma omp for
      for (std::size_t row_index = 0; row_index < lhs.dim(); ++row_index) {
        I sum_size = 0;

        auto [lhs_row_begin, lhs_row_end] = lhs.row_bounds(row_index);
        auto [rhs_row_begin, rhs_row_end] = rhs.row_bounds(row_index);

        while (lhs_row_begin != lhs_row_end && rhs_row_begin != rhs_row_end) {
          if (lt(*lhs_row_begin, *rhs_row_begin)) {
            const T result = op(std::get<T>(*lhs_row_begin++), T());
            sum_size += cast<I>(result != T());
          } else if (lt(*rhs_row_begin, *lhs_row_begin)) {
            const T result = op(T(), std::get<T>(*rhs_row_begin++));
            sum_size += cast<I>(result != T());
          } else {
            const T result = op(std::get<T>(*lhs_row_begin++),
                                std::get<T>(*rhs_row_begin++));
            sum_size += cast<I>(result != T());
          }
        }

        while (lhs_row_begin != lhs_row_end) {
          const T result = op(std::get<T>(*lhs_row_begin++), T());
          sum_size += cast<I>(result != T());
        }

        while (rhs_row_begin != rhs_row_end) {
          const T result = op(T(), std::get<T>(*rhs_row_begin++));
          sum_size += cast<I>(result != T());
        }
      }

#pragma omp barrier

#pragma omp single nowait
      {
        // cumulaive sum
        for (std::size_t row_index = 1; row_index <= lhs.dim(); ++row_index) {
          indptr[row_index] += indptr[row_index - 1];
        }
        out_data.resize(indptr.back());
      }

#pragma omp barrier

#pragma omp for
      for (std::size_t row_index = 0; row_index < lhs.dim(); ++row_index) {
        I sum_size = 0;

        auto out_row_begin = out_data.begin() + indptr[row_index];
        auto out_row_end = out_data.begin() + indptr[row_index + 1];

        auto [lhs_row_begin, lhs_row_end] = lhs.row_bounds(row_index);
        auto [rhs_row_begin, rhs_row_end] = rhs.row_bounds(row_index);

        while (lhs_row_begin != lhs_row_end && rhs_row_begin != rhs_row_end) {
          if (lt(*lhs_row_begin, *rhs_row_begin)) {
            const T result = op(std::get<T>(*lhs_row_begin), T());
            if (result != T()) {
              *out_row_begin++ =
                  std::forward_as_tuple(result, std::get<I>(*lhs_row_begin),
                                        std::get<J>(*lhs_row_begin));
            }
            ++lhs_row_begin;
          } else if (lt(*rhs_row_begin > *lhs_row_begin)) {
            const T result = op(T(), std::get<T>(*rhs_row_begin));
            if (result != T()) {
              *out_row_begin++ =
                  std::forward_as_tuple(result, std::get<I>(*rhs_row_begin),
                                        std::get<J>(*rhs_row_begin));
            }
            ++rhs_row_begin;
          } else {
            const T result =
                op(std::get<T>(*lhs_row_begin), std::get<T>(*rhs_row_begin));
            if (result != T()) {
              *out_row_begin++ =
                  std::forward_as_tuple(result, std::get<I>(*lhs_row_begin),
                                        std::get<J>(*lhs_row_begin));
            }
            ++lhs_row_begin;
            ++rhs_row_begin;
          }
        }
      }
    }
  } else {
    indptr.push_back(out_data.size());
    for (std::size_t row_index = 0; row_index < lhs.dim(); ++row_index) {
      auto [lhs_row_begin, lhs_row_end] = lhs.row_bounds(row_index);
      auto [rhs_row_begin, rhs_row_end] = rhs.row_bounds(row_index);

      while (lhs_row_begin != lhs_row_end && rhs_row_begin != rhs_row_end) {
        if (lt(*lhs_row_begin, *rhs_row_begin)) {
          const T result = op(std::get<T>(*lhs_row_begin), T());
          if (result != T()) {
            out_data.emplace_back(
                std::forward_as_tuple(result, std::get<I>(*lhs_row_begin),
                                      std::get<J>(*lhs_row_begin)));
          }
          ++lhs_row_begin;
        } else if (lt(*rhs_row_begin, *lhs_row_begin)) {
          const T result = op(T(), std::get<T>(*rhs_row_begin));
          if (result != T()) {
            out_data.emplace_back(
                std::forward_as_tuple(result, std::get<I>(*rhs_row_begin),
                                      std::get<J>(*rhs_row_begin)));
          }
          ++rhs_row_begin;
        } else {
          const T result =
              op(std::get<T>(*lhs_row_begin), std::get<T>(*rhs_row_begin));
          if (result != T()) {
            out_data.emplace_back(
                std::forward_as_tuple(result, std::get<I>(*lhs_row_begin),
                                      std::get<J>(*lhs_row_begin)));
          }
          ++lhs_row_begin;
          ++rhs_row_begin;
        }
      }

      while (lhs_row_begin != lhs_row_end) {
        out_data.push_back(*lhs_row_begin++);
      }

      while (rhs_row_begin != rhs_row_end) {
        out_data.push_back(*rhs_row_begin++);
      }

      indptr.push_back(out_data.size());
    }
  }

  return qmatrix<T, I, J>(indptr, out_data);
}

template<typename T, typename I, typename J>
  requires QMatrixTypes<T, I, J>
qmatrix<T, I, J> qmatrix<T, I, J>::operator+(const qmatrix &rhs) const {
  return binary_op(std::plus<T>(), *this, rhs);
}

template<typename T, typename I, typename J>
  requires QMatrixTypes<T, I, J>
qmatrix<T, I, J> qmatrix<T, I, J>::operator-(const qmatrix &rhs) const {
  return binary_op(std::minus<T>(), *this, rhs);
}

using qmatrices = std::variant<
    qmatrix<int8_t, int32_t, uint8_t>, qmatrix<int16_t, int32_t, uint8_t>,
    qmatrix<float, int32_t, uint8_t>, qmatrix<double, int32_t, uint8_t>,
    qmatrix<cfloat, int32_t, uint8_t>, qmatrix<cdouble, int32_t, uint8_t>,
    qmatrix<int8_t, int64_t, uint8_t>, qmatrix<int16_t, int64_t, uint8_t>,
    qmatrix<float, int64_t, uint8_t>, qmatrix<double, int64_t, uint8_t>,
    qmatrix<cfloat, int64_t, uint8_t>, qmatrix<cdouble, int64_t, uint8_t>,
    qmatrix<int8_t, int32_t, uint16_t>, qmatrix<int16_t, int32_t, uint16_t>,
    qmatrix<float, int32_t, uint16_t>, qmatrix<double, int32_t, uint16_t>,
    qmatrix<cfloat, int32_t, uint16_t>, qmatrix<cdouble, int32_t, uint16_t>,
    qmatrix<int8_t, int64_t, uint16_t>, qmatrix<int16_t, int64_t, uint16_t>,
    qmatrix<float, int64_t, uint16_t>, qmatrix<double, int64_t, uint16_t>,
    qmatrix<cfloat, int64_t, uint16_t>, qmatrix<cdouble, int64_t, uint16_t>>;

}}  // namespace quspin::detail
