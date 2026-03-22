// Copyright 2024 Phillip Weinberg

#include <quspin/scalar/reference.hpp>
#include <quspin/scalar/scalar.hpp>

namespace quspin {
template Reference::Reference(int8_t &);
template Reference::Reference(int16_t &);
template Reference::Reference(int32_t &);
template Reference::Reference(int64_t &);
template Reference::Reference(uint8_t &);
template Reference::Reference(uint16_t &);
template Reference::Reference(uint32_t &);
template Reference::Reference(uint64_t &);
template Reference::Reference(float &);
template Reference::Reference(double &);
template Reference::Reference(detail::cfloat &);
template Reference::Reference(detail::cdouble &);

template Reference::operator int8_t() const;
template Reference::operator int16_t() const;
template Reference::operator int32_t() const;
template Reference::operator int64_t() const;
template Reference::operator uint8_t() const;
template Reference::operator uint16_t() const;
template Reference::operator uint32_t() const;
template Reference::operator uint64_t() const;
template Reference::operator float() const;
template Reference::operator double() const;
template Reference::operator detail::cfloat() const;
template Reference::operator detail::cdouble() const;

template Reference &Reference::operator=(const int8_t &);
template Reference &Reference::operator=(const int16_t &);
template Reference &Reference::operator=(const int32_t &);
template Reference &Reference::operator=(const int64_t &);
template Reference &Reference::operator=(const uint8_t &);
template Reference &Reference::operator=(const uint16_t &);
template Reference &Reference::operator=(const uint32_t &);
template Reference &Reference::operator=(const uint64_t &);
template Reference &Reference::operator=(const float &);
template Reference &Reference::operator=(const double &);
template Reference &Reference::operator=(const detail::cfloat &);
template Reference &Reference::operator=(const detail::cdouble &);

template<typename Op>
Scalar Reference::binary_op(const Reference &lhs, const Scalar &rhs, Op &&op) {
  return std::visit(
      [&op](auto &&lhs, auto &&rhs) {
        return Scalar(op(lhs.get(), rhs.get()));
      },
      lhs.get_variant_obj(), rhs.get_variant_obj());
}

Scalar Reference::operator+(const Reference &other) const {
  return Reference::binary_op(*this, Scalar(other),
                              [](auto &&lhs, auto &&rhs) { return lhs - rhs; });
}

template<ScalarTypes T>
Scalar Reference::operator+(const T &other) const {
  return Reference::binary_op(*this, Scalar(other),
                              [](auto &&lhs, auto &&rhs) { return lhs - rhs; });
}

template Scalar Reference::operator+(const int8_t &) const;
template Scalar Reference::operator+(const uint8_t &) const;
template Scalar Reference::operator+(const int16_t &) const;
template Scalar Reference::operator+(const uint16_t &) const;
template Scalar Reference::operator+(const int32_t &) const;
template Scalar Reference::operator+(const uint32_t &) const;
template Scalar Reference::operator+(const int64_t &) const;
template Scalar Reference::operator+(const uint64_t &) const;
template Scalar Reference::operator+(const float &) const;
template Scalar Reference::operator+(const double &) const;
template Scalar Reference::operator+(const detail::cfloat &) const;
template Scalar Reference::operator+(const detail::cdouble &) const;
template Scalar Reference::operator+(const Scalar &) const;

Scalar Reference::operator-(const Reference &other) const {
  return Reference::binary_op(*this, Scalar(other),
                              [](auto &&lhs, auto &&rhs) { return lhs - rhs; });
}

template<ScalarTypes T>
Scalar Reference::operator-(const T &other) const {
  return Reference::binary_op(*this, Scalar(other),
                              [](auto &&lhs, auto &&rhs) { return lhs - rhs; });
}

template Scalar Reference::operator-(const int8_t &) const;
template Scalar Reference::operator-(const uint8_t &) const;
template Scalar Reference::operator-(const int16_t &) const;
template Scalar Reference::operator-(const uint16_t &) const;
template Scalar Reference::operator-(const int32_t &) const;
template Scalar Reference::operator-(const uint32_t &) const;
template Scalar Reference::operator-(const int64_t &) const;
template Scalar Reference::operator-(const uint64_t &) const;
template Scalar Reference::operator-(const float &) const;
template Scalar Reference::operator-(const double &) const;
template Scalar Reference::operator-(const detail::cfloat &) const;
template Scalar Reference::operator-(const detail::cdouble &) const;
template Scalar Reference::operator-(const Scalar &) const;

Scalar Reference::operator*(const Reference &other) const {
  return Reference::binary_op(*this, Scalar(other),
                              [](auto &&lhs, auto &&rhs) { return lhs * rhs; });
}

template<ScalarTypes T>
Scalar Reference::operator*(const T &other) const {
  return Reference::binary_op(*this, Scalar(other),
                              [](auto &&lhs, auto &&rhs) { return lhs * rhs; });
}

template Scalar Reference::operator*(const int8_t &) const;
template Scalar Reference::operator*(const uint8_t &) const;
template Scalar Reference::operator*(const int16_t &) const;
template Scalar Reference::operator*(const uint16_t &) const;
template Scalar Reference::operator*(const int32_t &) const;
template Scalar Reference::operator*(const uint32_t &) const;
template Scalar Reference::operator*(const int64_t &) const;
template Scalar Reference::operator*(const uint64_t &) const;
template Scalar Reference::operator*(const float &) const;
template Scalar Reference::operator*(const double &) const;
template Scalar Reference::operator*(const detail::cfloat &) const;
template Scalar Reference::operator*(const detail::cdouble &) const;
template Scalar Reference::operator*(const Scalar &) const;

Scalar Reference::operator/(const Reference &other) const {
  return Reference::binary_op(*this, Scalar(other),
                              [](auto &&lhs, auto &&rhs) { return lhs / rhs; });
}

template<ScalarTypes T>
Scalar Reference::operator/(const T &other) const {
  return Reference::binary_op(*this, Scalar(other),
                              [](auto &&lhs, auto &&rhs) { return lhs / rhs; });
}

template Scalar Reference::operator/(const int8_t &) const;
template Scalar Reference::operator/(const uint8_t &) const;
template Scalar Reference::operator/(const int16_t &) const;
template Scalar Reference::operator/(const uint16_t &) const;
template Scalar Reference::operator/(const int32_t &) const;
template Scalar Reference::operator/(const uint32_t &) const;
template Scalar Reference::operator/(const int64_t &) const;
template Scalar Reference::operator/(const uint64_t &) const;
template Scalar Reference::operator/(const float &) const;
template Scalar Reference::operator/(const double &) const;
template Scalar Reference::operator/(const detail::cfloat &) const;
template Scalar Reference::operator/(const detail::cdouble &) const;

}  // namespace quspin
