
#include <iostream>
#include <quspin/basis/detail/space.hpp>
#include <quspin/basis/grp/hardcore.hpp>
#include <ranges>
#include <vector>

int main() {
  Operator op(4);
  std::vector<std::vector<std::size_t>> seeds;
  seeds.emplace_back(std::vector<std::size_t>{0, 1, 0, 1});

  quspin::HardcoreBasis basis(op, seeds);

  for (std::size_t i = 0; i < basis.size(); i++) {
    std::cout << basis.fock_state(i) << std::endl;
  }

  return 0;
}
