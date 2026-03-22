// Copyright 2024 Phillip Weinberg

#include <chrono>
#include <iostream>
#include <quspin/array/detail/array.hpp>
#include <tuple>
#include <vector>

using namespace quspin::detail;

template<typename Func, typename... Args>
decltype(auto) time_func(Func func, Args... args) {
  auto start = std::chrono::high_resolution_clock::now();
  auto res = func(args...);
  auto stop = std::chrono::high_resolution_clock::now();
  return std::make_tuple(res, stop - start);
}

std::size_t test(array<float> &arr) {
  std::size_t i = 0;
  for (auto it = arr.begin(); it != arr.end(); ++it) {
    i += 1;
  }
  return i;
}

int main() {
  std::vector<std::size_t> shape0 = {100, 100, 100};
  array<float> arr0(shape0);

  auto [res0, time0] = time_func(test, arr0);

  std::cout << "Time: " << time0.count() << "ns" << std::endl;

  std::vector<std::size_t> shape1 = {100, 10000};
  array<float> arr1(shape1);

  auto [res1, time1] = time_func(test, arr1);

  std::cout << "Time: " << time1.count() << "ns" << std::endl;

  std::vector<std::size_t> shape2 = {1000000};
  array<float> arr2(shape2);

  auto [res2, time2] = time_func(test, arr2);

  std::cout << "Time: " << time2.count() << "ns" << std::endl;

  return 0;
}
