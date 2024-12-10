// Copyright 2024 Phillip Weinberg
#pragma once

#include <numeric>
#include <quspin/array/detail/iterator.hpp>
#include <quspin/detail/pointer.hpp>
#include <quspin/detail/type_concepts.hpp>
#include <quspin/dtype/detail/dtype.hpp>
#include <vector>

namespace quspin { namespace detail {

template<PrimativeTypes T>
struct array : public typed_object<T> {
  private:

    reference_counted_ptr<T> data_;
    std::vector<std::size_t> stride_;
    std::vector<std::size_t> shape_;
    std::size_t size_;
    std::size_t ndim_;
    bool is_contiguous_;

    static std::size_t get_ndim(const std::vector<std::size_t> &shape);
    static std::size_t get_size(const std::vector<std::size_t> &shape);
    static std::vector<std::size_t> get_strides(
        const std::vector<std::size_t> &shape);

    static bool check_contiguous(const std::vector<std::size_t> &stride,
                                 const std::vector<std::size_t> &shape);

  public:

    static constexpr std::size_t MAX_DIM = 64;

    array();
    array(const std::vector<std::size_t> &shape);
    array(const std::vector<std::size_t> &shape,
          const std::vector<std::size_t> &stride, T *data);
    array(std::initializer_list<T> values);

    template<typename Iterator>
    std::size_t get_offset(const Iterator &input_begin,
                           const Iterator &input_end) const;

    template<typename Iterator>
    void check_offset(const Iterator &input_begin,
                      const Iterator &input_end) const;

    const T *data() const;
    T *mut_data();

    array_iterator<T> begin() const;
    array_iterator<T> end() const;
    array_iterator<T> begin();
    array_iterator<T> end();

    bool is_contiguous() const;

    std::size_t strides(const std::size_t &i) const;
    std::size_t shape(const std::size_t &i) const;
    std::vector<std::size_t> shape() const;
    std::vector<std::size_t> strides() const;

    std::size_t ndim() const;
    std::size_t size() const;
    std::size_t itemsize() const;
    std::size_t use_count() const;

    template<typename Container>
    const T &at(const Container &indices) const;
    template<std::integral IndexType>
    const T &at(std::initializer_list<IndexType> indices) const;

    template<typename Container>
    T &at(const Container &indices);
    template<std::integral IndexType>
    T &at(std::initializer_list<IndexType> indices);

    template<typename Container>
    const T &operator[](const Container &indices) const;
    template<std::integral IndexType>
    const T &operator[](std::initializer_list<IndexType> indices) const;

    template<typename Container>
    T &operator[](const Container &indices);
    template<std::integral IndexType>
    T &operator[](std::initializer_list<IndexType> indices);

    array<T> copy() const;
};

template<PrimativeTypes T>
std::size_t array<T>::get_ndim(const std::vector<std::size_t> &shape) {
  return shape.size();
}

template<PrimativeTypes T>
std::size_t array<T>::get_size(const std::vector<std::size_t> &shape) {
  return (shape.size() > 0
              ? std::reduce(shape.begin(), shape.end(), std::size_t(1),
                            std::multiplies<std::size_t>())
              : 0);
}

template<PrimativeTypes T>
std::vector<std::size_t> array<T>::get_strides(
    const std::vector<std::size_t> &shape) {
  std::vector<std::size_t> &&strides = std::vector<std::size_t>(shape.size());
  if (shape.size() == 0) {
    return std::move(strides);
  }

  strides[shape.size() - 1] = sizeof(T);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return std::move(strides);
}

template<PrimativeTypes T>
bool array<T>::check_contiguous(const std::vector<std::size_t> &stride,
                                const std::vector<std::size_t> &shape) {
  if (stride.size() != shape.size()) {
    throw std::invalid_argument("Stride and shape must have the same size");
  }

  if (stride.size() == 0) {  // empty array
    return true;
  }

  const std::size_t ndim = stride.size();
  std::size_t last_stride = stride[ndim - 1];

  if (last_stride != sizeof(T)) {
    return false;
  }

  if (ndim == 1) {
    return true;
  }
  for (std::size_t i = ndim - 1; i > 0; i--) {
    if (stride[i - 1] != last_stride * shape[i]) {
      return false;
    }
    last_stride = stride[i];
  }
  return true;
}

template<PrimativeTypes T>
template<typename Iterator>
void array<T>::check_offset(const Iterator &indices_begin,
                            const Iterator &indices_end) const {
  if (std::distance(indices_begin, indices_end) !=
      static_cast<std::ptrdiff_t>(ndim())) {
    throw std::runtime_error("Invalid number of indices");
  }

  auto shape_begin = shape_.begin();
  auto ndim = shape_.size();

  std::for_each(
      indices_begin, indices_end, [&shape_begin, &ndim](const std::size_t &i) {
        if (i >= *shape_begin) {
          throw std::runtime_error("Index " + std::to_string(i) +
                                   " out of bounds: " + std::to_string(i) +
                                   " >= " + std::to_string(ndim));
        }
        ++shape_begin;
      });
}

template<PrimativeTypes T>
template<typename Iterator>
std::size_t array<T>::get_offset(const Iterator &input_begin,
                                 const Iterator &input_end) const {
  return std::inner_product(input_begin, input_end, stride_.begin(),
                            std::size_t()) /
         sizeof(T);
}

template<PrimativeTypes T>
array<T>::array()
    : data_(reference_counted_ptr<T>()),
      stride_({}),
      shape_({}),
      size_(0),
      ndim_(0),
      is_contiguous_(true) {}

template<PrimativeTypes T>
array<T>::array(const std::vector<std::size_t> &shape)
    : data_(reference_counted_ptr<T>(get_size(shape))),
      stride_(get_strides(shape)),
      shape_(shape),
      size_(get_size(shape)),
      ndim_(get_ndim(shape)),
      is_contiguous_(true) {}

template<PrimativeTypes T>
array<T>::array(const std::vector<std::size_t> &shape,
                const std::vector<std::size_t> &stride, T *data)
    : data_(reference_counted_ptr<T>(data)),
      stride_(stride),
      shape_(shape),
      size_(get_size(shape)),
      ndim_(get_ndim(shape)),
      is_contiguous_(check_contiguous(stride, shape)) {}

template<PrimativeTypes T>
array<T>::array(std::initializer_list<T> values)
    : data_(reference_counted_ptr<T>(values.size())),
      stride_({sizeof(T)}),
      shape_({values.size()}),
      size_(values.size()),
      ndim_(1),
      is_contiguous_(true) {
  std::copy(values.begin(), values.end(), begin());
}

template<PrimativeTypes T>
const T *array<T>::data() const {
  return data_.get();
}

template<PrimativeTypes T>
T *array<T>::mut_data() {
  return data_.get();
}

template<PrimativeTypes T>
array_iterator<T> array<T>::begin() const {
  array_iterator<T> iter(const_cast<T *>(data()), shape_, stride_, 0);
  return iter;
}

template<PrimativeTypes T>
array_iterator<T> array<T>::end() const {
  array_iterator<T> iter(const_cast<T *>(data()), shape_, stride_, size());
  return iter;
}

template<PrimativeTypes T>
array_iterator<T> array<T>::begin() {
  array_iterator<T> iter(mut_data(), shape_, stride_, 0);
  return iter;
}

template<PrimativeTypes T>
array_iterator<T> array<T>::end() {
  array_iterator<T> iter(mut_data(), shape_, stride_, size());
  return iter;
}

template<PrimativeTypes T>
bool array<T>::is_contiguous() const {
  return is_contiguous_;
}

template<PrimativeTypes T>
std::size_t array<T>::strides(const std::size_t &i) const {
  if (i >= ndim_) {
    return 0;
  }
  return stride_.at(i);
}

template<PrimativeTypes T>
std::size_t array<T>::shape(const std::size_t &i) const {
  if (i >= ndim_) {
    return 0;
  }
  return shape_.at(i);
}

template<PrimativeTypes T>
std::vector<std::size_t> array<T>::shape() const {
  return shape_;
}

template<PrimativeTypes T>
std::vector<std::size_t> array<T>::strides() const {
  return stride_;
}

template<PrimativeTypes T>
std::size_t array<T>::ndim() const {
  return ndim_;
}

template<PrimativeTypes T>
std::size_t array<T>::size() const {
  return size_;
}

template<PrimativeTypes T>
std::size_t array<T>::itemsize() const {
  return sizeof(T);
}

template<PrimativeTypes T>
std::size_t array<T>::use_count() const {
  return data_.use_count();
}

template<PrimativeTypes T>
template<typename Container>
inline const T &array<T>::operator[](const Container &indices) const {
  const std::size_t offset = get_offset(indices.begin(), indices.end());
  return data()[offset];
}

template<PrimativeTypes T>
template<std::integral IndexType>
inline const T &array<T>::operator[](
    std::initializer_list<IndexType> indices) const {
  const std::size_t offset = get_offset(indices.begin(), indices.end());
  return data()[offset];
}

template<PrimativeTypes T>
template<typename Container>
inline T &array<T>::operator[](const Container &indices) {
  const std::size_t offset = get_offset(indices.begin(), indices.end());
  return mut_data()[offset];
}

template<PrimativeTypes T>
template<std::integral IndexType>
inline T &array<T>::operator[](std::initializer_list<IndexType> indices) {
  const std::size_t offset = get_offset(indices.begin(), indices.end());
  return mut_data()[offset];
}

template<PrimativeTypes T>
template<typename Container>
const T &array<T>::at(const Container &indices) const {
  check_offset(indices.begin(), indices.end());
  const std::size_t offset = get_offset(indices.begin(), indices.end());
  return data()[offset];
}

template<PrimativeTypes T>
template<std::integral IndexType>
const T &array<T>::at(std::initializer_list<IndexType> indices) const {
  check_offset(indices.begin(), indices.end());
  const std::size_t offset = get_offset(indices.begin(), indices.end());
  return data()[offset];
}

template<PrimativeTypes T>
template<typename Container>
T &array<T>::at(const Container &indices) {
  check_offset(indices.begin(), indices.end());
  const std::size_t offset = get_offset(indices.begin(), indices.end());
  return mut_data()[offset];
}

template<PrimativeTypes T>
template<std::integral IndexType>
T &array<T>::at(std::initializer_list<IndexType> indices) {
  check_offset(indices.begin(), indices.end());
  const std::size_t offset = get_offset(indices.begin(), indices.end());
  return mut_data()[offset];
}

template<PrimativeTypes T>
array<T> array<T>::copy() const {
  array<T> out(shape_);
  std::copy(begin(), end(), out.begin());
  return out;
}

template<typename T>
struct value_type<array<T>> {
    using type = T;
};

template<typename T>
constexpr bool is_integer_array_v =
    std::is_integral_v<typename array<T>::value_type>;

template<typename T>
constexpr bool is_floating_array_v =
    std::is_floating_point_v<typename array<T>::value_type>;

template<typename T>
constexpr bool is_complex_array_v =
    std::conjunction_v<std::is_same<typename array<T>::value_type, cfloat>,
                       std::is_same<typename array<T>::value_type, cdouble>,
                       std::is_same<typename array<T>::value_type, cldouble>>;

template<typename U, typename T>
constexpr bool is_array_type_v =
    std::is_same_v<U, typename array<T>::value_type>;

using arrays = std::variant<array<int8_t>, array<uint8_t>, array<int16_t>,
                            array<uint16_t>, array<uint32_t>, array<int32_t>,
                            array<uint64_t>, array<int64_t>, array<float>,
                            array<double>, array<cfloat>, array<cdouble>>;

using index_arrays = std::variant<array<int32_t>, array<int64_t>>;
using matrix_arrays = std::variant<array<int8_t>, array<float>, array<double>,
                                   array<cfloat>, array<cdouble>>;
using state_arrays = std::variant<array<double>, array<cdouble>>;

template<typename T>
constexpr bool is_index_type_v =
    std::conjunction_v<std::is_same<T, int32_t>, std::is_same<T, int64_t>>;

template<typename T>
constexpr bool is_matrix_type_v =
    std::conjunction_v<std::is_same<T, int8_t>, std::is_same<T, float>,
                       std::is_same<T, double>, std::is_same<T, cfloat>,
                       std::is_same<T, cdouble>>;

template<typename T>
constexpr bool is_state_type_v =
    std::conjunction_v<std::is_same<T, double>, std::is_same<T, cdouble>>;

}}  // namespace quspin::detail
