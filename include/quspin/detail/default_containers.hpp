#ifdef USE_BOOST
#include <boost/container/small_vector.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

namespace quspin::detail {
template<typename T, std::size_t N>
using svector_t = boost::container::small_vector<T, N>;

template<typename Key, typename Value>
using default_map_t = boost::unordered_flat_map<Key, Value>;

}  // namespace quspin::detail
#else

#include <unordered_map>
#include <vector>

namespace quspin::detail {
template<typename T, std::size_t N>
using svector = std::vector<T>;

template<typename Key, typename Value>
using default_map_t = std::unordered_map<Key, Value>;
}  // namespace quspin::detail
#endif
