// Copyright 2024 Phillip Weinberg
#pragma once

#include <memory>
#include <quspin/detail/operators.hpp>
#include <quspin/detail/variant_container.hpp>
#include <quspin/dtype/detail/dtype.hpp>
#include <string>
#include <vector>

namespace quspin {

class DType : public detail::VariantContainer<detail::dtypes> {
    using detail::VariantContainer<detail::dtypes>::internals_;

    static constexpr detail::dtypes default_value() {
      return detail::dtypes(detail::dtype<double>());
    }

  public:

    DType() : detail::VariantContainer<detail::dtypes>(default_value()) {};

    template<typename T>
    DType(const detail::dtype<T> &dtype)
        : detail::VariantContainer<detail::dtypes>(dtype) {}

    std::string name() const;

    template<typename T>
    static DType of() {
      using val_t = detail::value_type_t<std::decay_t<T>>;
      return DType(detail::dtype<detail::value_type_t<val_t>>());
    }

    bool operator==(const DType &dtype) const;
};

DType result_dtype(std::vector<DType> &);
bool int_dtype(const DType &);
bool float_dtype(const DType &);
bool complex_dtype(const DType &);

static const DType Int8 = DType(detail::dtype<int8_t>());
static const DType UInt8 = DType(detail::dtype<uint8_t>());
static const DType Int16 = DType(detail::dtype<int16_t>());
static const DType UInt16 = DType(detail::dtype<uint16_t>());
static const DType Int32 = DType(detail::dtype<int32_t>());
static const DType UInt32 = DType(detail::dtype<uint32_t>());
static const DType Int64 = DType(detail::dtype<int64_t>());
static const DType UInt64 = DType(detail::dtype<uint64_t>());
static const DType Float = DType(detail::dtype<float>());
static const DType Double = DType(detail::dtype<double>());
static const DType CFloat = DType(detail::dtype<detail::cfloat>());
static const DType CDouble = DType(detail::dtype<detail::cdouble>());

template<typename Variant>
class DTypeObject : public detail::VariantContainer<Variant> {
  protected:

    using detail::VariantContainer<Variant>::internals_;

  public:

    DTypeObject(const Variant &internals)
        : detail::VariantContainer<Variant>(internals) {}

    DTypeObject(Variant &&internals)
        : detail::VariantContainer<Variant>(std::move(internals)) {}

    DType dtype() const {
      return std::visit(
          [](const auto &obj) { return DType::of<decltype(obj)>(); },
          internals_);
    }
};

}  // namespace quspin
