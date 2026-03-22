

#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <quspin/basis/hardcore.hpp>
#include <quspin/dtype/dtype.hpp>
#include <quspin/operator.hpp>
#include <quspin/qmatrix/qmatrix.hpp>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

template<typename Function, typename... Args>
decltype(auto) time_function(Function &&f, Args &&...args) {
  auto t1 = std::chrono::high_resolution_clock::now();
  auto result = std::forward<Function>(f)(std::forward<Args>(args)...);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout
      << "time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << std::endl;
  return result;
}

int test_basis() {
  const std::size_t L = 20;
  const int q = 0;
  const int z = 0;

  using term_t = std::tuple<std::size_t, std::string, std::vector<std::size_t>,
                            std::complex<double>>;
  std::vector<term_t> ham_list;

  for (std::size_t i = 0; i < L; i++) {
    std::vector<std::size_t> locs = {i};
    const std::size_t cindex = 0;
    ham_list.emplace_back(cindex, "z", locs, 1.0);
  }

  for (std::size_t i = 0; i < L - 1; i++) {
    std::vector<std::size_t> locs = {i, i + 1};
    const std::size_t cindex = 1;
    ham_list.emplace_back(cindex, "xx", locs, 1.0);
  }

  quspin::PauliHamiltonian ham(ham_list);
  quspin::HardcoreBasis my_subspace(32, true);
  std::vector<std::vector<std::size_t>> seeds = {{1}};

  my_subspace.construct_from(ham, seeds);

  std::cout << "size: " << my_subspace.size() << std::endl;

  quspin::QMatrix(ham, my_subspace, quspin::Double);

  return 0;
}

int main() {
  test_basis();

  return 0;
}
