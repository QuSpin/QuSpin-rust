// Copyright 2024 Phillip Weinberg
#include <cassert>
#include <quspin/dtype/dtype.hpp>

int main() {
  using namespace quspin;

  assert(Int8 == DType::of<int8_t>());
  assert(Int16 == DType::of<int16_t>());
  assert(Int32 == DType::of<int32_t>());
  assert(Int64 == DType::of<int64_t>());
  assert(UInt8 == DType::of<uint8_t>());
  assert(UInt16 == DType::of<uint16_t>());
  assert(UInt32 == DType::of<uint32_t>());
  assert(UInt64 == DType::of<uint64_t>());
  assert(Float == DType::of<float>());
  assert(Double == DType::of<double>());
  assert(CFloat == DType::of<detail::cfloat>());
  assert(CDouble == DType::of<detail::cdouble>());
}
