
#include <quspin/basis/detail/bitbasis/dit_manip.hpp>
#include <quspin/basis/detail/bitbasis/dit_perm.hpp>
#include <quspin/basis/detail/types.hpp>
#include <valarray>
#include <vector>

using namespace quspin::details::basis;

int main() {
  std::array<uint32_t, 8> s_val = {0b0000, 0b0001, 0b0010, 0b0011,
                                   0b0100, 0b0101, 0b0110, 0b0111};

  dit_manip<2> d;

  std::vector<grp_result<uint32_t>> s_list;

  for (auto s : s_val) {
    s_list.emplace_back(s, std::complex<double>{1.0, 0.0});
  }

  std::vector<int> perm = {1, 0};
  std::complex<double> grp_char = {-1.0, 0.0};

  using perm_operation_t = perm_dit_locations<uint32_t>;
  using lattice_grp_element_t = grp_element<uint32_t, perm_operation_t>;
  lattice_grp_element_t grp_single(grp_char, 2, perm);

  return 0;
}
