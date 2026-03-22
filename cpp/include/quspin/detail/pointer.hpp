// Copyright 2024 Phillip Weinberg
#pragma once

#include <quspin/detail/type_concepts.hpp>
#include <quspin/dtype/detail/dtype.hpp>

namespace quspin { namespace detail {

template<PrimativeTypes T>
struct reference_counted_ptr : public typed_object<T> {
    T *ptr;
    std::size_t *ref_count_;
    bool owns_pointer;  // if true, the pointer can be deleted when ref_count_
                        // == 1

    void inc();
    void dec();

  public:

    reference_counted_ptr();
    explicit reference_counted_ptr(T *ptr);
    explicit reference_counted_ptr(std::size_t size);
    explicit reference_counted_ptr(const reference_counted_ptr &other);
    ~reference_counted_ptr();

    reference_counted_ptr<T> &operator=(const reference_counted_ptr<T> &);
    std::size_t use_count() const;

    T *get();
    const T *get() const;
    void *data() const;
};

template<PrimativeTypes T>
void reference_counted_ptr<T>::inc() {
  (*ref_count_)++;
}

template<PrimativeTypes T>
void reference_counted_ptr<T>::dec() {
  if (*ref_count_ == 1) {
    delete ref_count_;
    if (owns_pointer) {
      delete[] ptr;
    }
  } else {
    (*ref_count_)--;
  }
}

template<PrimativeTypes T>
reference_counted_ptr<T>::reference_counted_ptr()
    : ptr(nullptr),
      ref_count_(new std::size_t(1)),
      owns_pointer(false)  // cannot delete nullptr
{}

template<PrimativeTypes T>
reference_counted_ptr<T>::reference_counted_ptr(T *ptr)
    : ptr(ptr),
      ref_count_(new std::size_t(1)),
      owns_pointer(false)  // objects all own the memory, can delete
{}

template<PrimativeTypes T>
reference_counted_ptr<T>::reference_counted_ptr(std::size_t size)
    : ptr(new T[size]),
      ref_count_(new std::size_t(1)),
      owns_pointer(true)  // objects all own the memory, can delete
{
  std::fill(ptr, ptr + size, T());
}

template<PrimativeTypes T>
reference_counted_ptr<T>::reference_counted_ptr(
    const reference_counted_ptr &other)
    : ptr(other.ptr),
      ref_count_(other.ref_count_),
      owns_pointer(other.owns_pointer)  // inherit ownership of memory
{
  inc();
}

template<PrimativeTypes T>
reference_counted_ptr<T>::~reference_counted_ptr() {
  dec();
}

template<PrimativeTypes T>
reference_counted_ptr<T> &reference_counted_ptr<T>::operator=(
    const reference_counted_ptr<T> &other) {
  dec();
  ptr = other.ptr;
  ref_count_ = other.ref_count_;
  owns_pointer = other.owns_pointer;
  inc();

  return *this;
}

template<PrimativeTypes T>
std::size_t reference_counted_ptr<T>::use_count() const {
  return *ref_count_;
}

template<PrimativeTypes T>
T *reference_counted_ptr<T>::get() {
  return ptr;
}

template<PrimativeTypes T>
const T *reference_counted_ptr<T>::get() const {
  return static_cast<const T *>(ptr);
}

template<PrimativeTypes T>
void *reference_counted_ptr<T>::data() const {
  return reinterpret_cast<void *>(ptr);
}

}}  // namespace quspin::detail
