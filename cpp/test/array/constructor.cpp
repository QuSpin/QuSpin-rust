// Copyright 2024 Phillip Weinberg
#include <cassert>
#include <exception>
#include <quspin/array/array.hpp>
#include <quspin/array/kernels.hpp>
#include <quspin/dtype/dtype.hpp>
#include <string>

int main() {
  using namespace quspin;

  Array a({1, 2, 3}, Float);
  assert(a.ndim() == 3);
  assert(a.size() == 6);
  std::vector<std::size_t> shape = a.shape();
  assert(shape == std::vector<std::size_t>({1, 2, 3}));

  for (std::size_t i = 0; i < a.shape(0); i++) {
    for (std::size_t j = 0; j < a.shape(1); j++) {
      for (std::size_t k = 0; k < a.shape(2); k++) {
        a[{i, j, k}] = static_cast<float>(i + j + k);
      }
    }
  }

  for (std::size_t i = 0; i < a.shape(0); i++) {
    for (std::size_t j = 0; j < a.shape(1); j++) {
      for (std::size_t k = 0; k < a.shape(2); k++) {
        assert(static_cast<float>(a[{i, j, k}]) ==
               static_cast<float>(i + j + k));
      }
    }
  }
}
