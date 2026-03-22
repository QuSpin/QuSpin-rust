
#include <iostream>
#include <memory>
#include <variant>
#include <vector>

template<typename T>
class A {
  public:

    std::vector<T> value;

    A() = default;

    A(std::vector<T> &&value) : value(std::move(value)) {}

    A(const A &a) = delete;
    A(A &&a) = default;
    A &operator=(const A &a) = delete;
    A &operator=(A &&a) = default;
};

using ATypes = std::variant<A<int>, A<double>>;

int main() {
  ATypes a(std::in_place_type<A<int>>, std::vector<int>{1, 2, 3, 4, 5});

  return 0;
}
