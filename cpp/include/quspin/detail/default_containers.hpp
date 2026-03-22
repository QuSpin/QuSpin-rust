#include <unordered_map>
#include <vector>

namespace quspin::detail {

template<typename T, std::size_t N>
using svector_t = std::vector<T>;

template<typename Key, typename Value>
using default_map_t = std::unordered_map<Key, Value>;

}  // namespace quspin::detail
