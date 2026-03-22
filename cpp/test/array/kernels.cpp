// Copyright 2024 Phillip Weinberg
#include <cassert>
#include <exception>
#include <quspin/array/array.hpp>
#include <quspin/array/kernels.hpp>
#include <quspin/dtype/dtype.hpp>
#include <string>

int main() {
  using namespace quspin;

  Array A({100}, Int64);
  Array B({100}, Float);

  for (std::size_t i = 0; i < A.shape(0); i++) {
    A[{i}] = i;
    B[{i}] = 2 * i;
  }

  Array D = add(A, B);

  assert(D.dtype() == Float);

  for (std::size_t i = 0; i < A.shape(0); i++) {
    assert(static_cast<float>(D[{i}]) == static_cast<float>(3 * i));
  }

  Array a({10}, Int64);
  Array b({9}, Float);
  Array c({10}, Int64);
  Array out({10}, Int8);

  try {
    add(a, b);
  } catch (const std::invalid_argument &e) {
    assert(std::string(e.what()) == std::string("Incompatible shapes"));
  }

  try {
    add(a, c, out);
  } catch (const std::invalid_argument &e) {
    assert(std::string(e.what()) == std::string("Incompatible out type"));
  }

  assert(allclose(A, A));
}
