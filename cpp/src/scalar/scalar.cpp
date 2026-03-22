// Copyright 2024 Phillip Weinberg

#include <quspin/detail/type_concepts.hpp>
#include <quspin/scalar/scalar.hpp>

namespace quspin {

template Scalar::Scalar(const int8_t &);
template Scalar::Scalar(const uint8_t &);
template Scalar::Scalar(const int16_t &);
template Scalar::Scalar(const uint16_t &);
template Scalar::Scalar(const int32_t &);
template Scalar::Scalar(const uint32_t &);
template Scalar::Scalar(const int64_t &);
template Scalar::Scalar(const uint64_t &);
template Scalar::Scalar(const float &);
template Scalar::Scalar(const double &);
template Scalar::Scalar(const detail::cfloat &);
template Scalar::Scalar(const detail::cdouble &);

template<PrimativeTypes T>
Scalar::operator T() const {
  return std::visit(
      [](const auto &internals) { return detail::cast<T>(internals.get()); },
      internals_);
}

template Scalar::operator int8_t() const;
template Scalar::operator uint8_t() const;
template Scalar::operator int16_t() const;
template Scalar::operator uint16_t() const;
template Scalar::operator int32_t() const;
template Scalar::operator uint32_t() const;
template Scalar::operator int64_t() const;
template Scalar::operator uint64_t() const;
template Scalar::operator float() const;
template Scalar::operator double() const;
template Scalar::operator detail::cfloat() const;
template Scalar::operator detail::cdouble() const;

template<PrimativeTypes T>
Scalar &Scalar::operator=(const T &value) {
  internals_ = detail::scalars(detail::scalar<T>(value));
  return *this;
}

template Scalar &Scalar::operator=(const int8_t &);
template Scalar &Scalar::operator=(const uint8_t &);
template Scalar &Scalar::operator=(const int16_t &);
template Scalar &Scalar::operator=(const uint16_t &);
template Scalar &Scalar::operator=(const int32_t &);
template Scalar &Scalar::operator=(const uint32_t &);
template Scalar &Scalar::operator=(const int64_t &);
template Scalar &Scalar::operator=(const uint64_t &);
template Scalar &Scalar::operator=(const float &);
template Scalar &Scalar::operator=(const double &);
template Scalar &Scalar::operator=(const detail::cfloat &);
template Scalar &Scalar::operator=(const detail::cdouble &);

template<typename Op>
Scalar Scalar::binary_op(const Scalar &lhs, const Scalar &rhs, Op &&op) {
  return std::visit(
      [&op](auto &&lhs, auto &&rhs) {
        return Scalar(op(lhs.get(), rhs.get()));
      },
      lhs.get_variant_obj(), rhs.get_variant_obj());
}

Scalar Scalar::operator+(const Scalar &other) const {
  return Scalar::binary_op(*this, other,
                           [](auto &&lhs, auto &&rhs) { return lhs + rhs; });
}

template<PrimativeTypes T>
Scalar Scalar::operator+(const T &other) const {
  return Scalar::binary_op(*this, Scalar(other),
                           [](auto &&lhs, auto &&rhs) { return lhs + rhs; });
}

template Scalar Scalar::operator+(const int8_t &) const;
template Scalar Scalar::operator+(const uint8_t &) const;
template Scalar Scalar::operator+(const int16_t &) const;
template Scalar Scalar::operator+(const uint16_t &) const;
template Scalar Scalar::operator+(const int32_t &) const;
template Scalar Scalar::operator+(const uint32_t &) const;
template Scalar Scalar::operator+(const int64_t &) const;
template Scalar Scalar::operator+(const uint64_t &) const;
template Scalar Scalar::operator+(const float &) const;
template Scalar Scalar::operator+(const double &) const;
template Scalar Scalar::operator+(const detail::cfloat &) const;
template Scalar Scalar::operator+(const detail::cdouble &) const;

Scalar Scalar::operator-(const Scalar &other) const {
  return Scalar::binary_op(*this, other,
                           [](auto &&lhs, auto &&rhs) { return lhs - rhs; });
}

template<PrimativeTypes T>
Scalar Scalar::operator-(const T &other) const {
  return Scalar::binary_op(*this, Scalar(other),
                           [](auto &&lhs, auto &&rhs) { return lhs - rhs; });
}

template Scalar Scalar::operator-(const int8_t &) const;
template Scalar Scalar::operator-(const uint8_t &) const;
template Scalar Scalar::operator-(const int16_t &) const;
template Scalar Scalar::operator-(const uint16_t &) const;
template Scalar Scalar::operator-(const int32_t &) const;
template Scalar Scalar::operator-(const uint32_t &) const;
template Scalar Scalar::operator-(const int64_t &) const;
template Scalar Scalar::operator-(const uint64_t &) const;
template Scalar Scalar::operator-(const float &) const;
template Scalar Scalar::operator-(const double &) const;
template Scalar Scalar::operator-(const detail::cfloat &) const;
template Scalar Scalar::operator-(const detail::cdouble &) const;

Scalar Scalar::operator*(const Scalar &other) const {
  return Scalar::binary_op(*this, other,
                           [](auto &&lhs, auto &&rhs) { return lhs * rhs; });
}

template<PrimativeTypes T>
Scalar Scalar::operator*(const T &other) const {
  return Scalar::binary_op(*this, Scalar(other),
                           [](auto &&lhs, auto &&rhs) { return lhs * rhs; });
}

template Scalar Scalar::operator*(const int8_t &) const;
template Scalar Scalar::operator*(const uint8_t &) const;
template Scalar Scalar::operator*(const int16_t &) const;
template Scalar Scalar::operator*(const uint16_t &) const;
template Scalar Scalar::operator*(const int32_t &) const;
template Scalar Scalar::operator*(const uint32_t &) const;
template Scalar Scalar::operator*(const int64_t &) const;
template Scalar Scalar::operator*(const uint64_t &) const;
template Scalar Scalar::operator*(const float &) const;
template Scalar Scalar::operator*(const double &) const;
template Scalar Scalar::operator*(const detail::cfloat &) const;
template Scalar Scalar::operator*(const detail::cdouble &) const;

Scalar Scalar::operator/(const Scalar &other) const {
  return Scalar::binary_op(*this, other,
                           [](auto &&lhs, auto &&rhs) { return lhs / rhs; });
}

template<PrimativeTypes T>
Scalar Scalar::operator/(const T &other) const {
  return Scalar::binary_op(*this, Scalar(other),
                           [](auto &&lhs, auto &&rhs) { return lhs / rhs; });
}

template Scalar Scalar::operator/(const int8_t &) const;
template Scalar Scalar::operator/(const uint8_t &) const;
template Scalar Scalar::operator/(const int16_t &) const;
template Scalar Scalar::operator/(const uint16_t &) const;
template Scalar Scalar::operator/(const int32_t &) const;
template Scalar Scalar::operator/(const uint32_t &) const;
template Scalar Scalar::operator/(const int64_t &) const;
template Scalar Scalar::operator/(const uint64_t &) const;
template Scalar Scalar::operator/(const float &) const;
template Scalar Scalar::operator/(const double &) const;
template Scalar Scalar::operator/(const detail::cfloat &) const;
template Scalar Scalar::operator/(const detail::cdouble &) const;
}  // namespace quspin
