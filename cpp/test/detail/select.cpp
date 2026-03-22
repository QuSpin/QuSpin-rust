// Copyright 2024 Phillip Weinberg

#include <iostream>
#include <quspin/array/array.hpp>
#include <quspin/array/detail/array.hpp>
#include <quspin/detail/select.hpp>
#include <quspin/dtype/dtype.hpp>

int main() {
  using namespace quspin;

  Array a({10}, Int8);

  auto b = select<detail::array<int8_t>>(a);

  try {
    auto c = select<detail::array<int16_t>>(a);
  } catch (const std::invalid_argument &e) {
    std::cout << e.what() << std::endl;
  }

  return 0;
}
