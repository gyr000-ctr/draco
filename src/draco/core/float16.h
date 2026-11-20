// Copyright 2026 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_CORE_FLOAT16_H_
#define DRACO_CORE_FLOAT16_H_

#include <stdint.h>

#include <cstring>
#include <limits>
#include <type_traits>

namespace draco {

// Converts an IEEE 754 binary32 (float) value to the bit pattern of the
// closest IEEE 754 binary16 (half precision) value, using
// round-to-nearest-even. Values that overflow the half precision range are
// converted to the appropriately signed infinity, values too small to be
// represented are converted to the appropriately signed zero. NaN payloads
// are preserved in the most significant mantissa bits (and the result is
// guaranteed to stay NaN). Any binary16 value converted to binary32 and back
// with Float16BitsToFloat() / FloatToFloat16Bits() yields the original bit
// pattern, including NaN, +/-Inf and subnormal values.
inline uint16_t FloatToFloat16Bits(float value) {
  uint32_t float_bits;
  std::memcpy(&float_bits, &value, sizeof(float_bits));
  const uint32_t sign = (float_bits >> 16) & 0x8000;
  const uint32_t exponent = (float_bits >> 23) & 0xff;
  const uint32_t mantissa = float_bits & 0x7fffff;

  if (exponent == 0xff) {
    // Infinity or NaN.
    if (mantissa == 0) {
      return static_cast<uint16_t>(sign | 0x7c00);
    }
    // NaN: preserve the most significant payload bits and make sure the
    // result stays NaN (the payload must be non-zero).
    uint32_t half_mantissa = mantissa >> 13;
    if (half_mantissa == 0) {
      half_mantissa = 1;
    }
    return static_cast<uint16_t>(sign | 0x7c00 | half_mantissa);
  }

  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 0x1f) {
    // Overflow: convert to infinity.
    return static_cast<uint16_t>(sign | 0x7c00);
  }
  if (half_exponent <= 0) {
    // Subnormal half precision value (or zero).
    if (half_exponent < -10) {
      // The value is too small to be represented as a subnormal half. It
      // rounds to zero (the largest such value, 2^-25, is a tie that rounds
      // to the even zero).
      return static_cast<uint16_t>(sign);
    }
    // Round the mantissa (with the implicit leading bit restored) to the
    // subnormal precision using round-to-nearest-even.
    const uint32_t full_mantissa = mantissa | 0x800000;
    const int shift = 14 - half_exponent;
    uint32_t half_mantissa = full_mantissa >> shift;
    const uint32_t remainder = full_mantissa & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1))) {
      ++half_mantissa;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }
  // Normalized half precision value: round the mantissa from 23 to 10 bits
  // using round-to-nearest-even.
  uint32_t half_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fff;
  uint32_t new_exponent = static_cast<uint32_t>(half_exponent);
  if (remainder > 0x1000 || (remainder == 0x1000 && (half_mantissa & 1))) {
    ++half_mantissa;
    if (half_mantissa == 0x400) {
      // The mantissa overflowed, shift the carry into the exponent.
      half_mantissa = 0;
      ++new_exponent;
      if (new_exponent >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00);
      }
    }
  }
  return static_cast<uint16_t>(sign | (new_exponent << 10) | half_mantissa);
}

// Converts the bit pattern of an IEEE 754 binary16 (half precision) value to
// an IEEE 754 binary32 (float) value. The conversion is always exact.
inline float Float16BitsToFloat(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1f;
  const uint32_t mantissa = bits & 0x3ff;
  uint32_t float_bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      // +/- zero.
      float_bits = sign;
    } else {
      // Subnormal half: value = mantissa * 2^-24. The multiplication is
      // exact because both operands are exactly representable as floats.
      const float value =
          static_cast<float>(mantissa) * 5.9604644775390625e-08f;
      std::memcpy(&float_bits, &value, sizeof(float_bits));
      float_bits |= sign;
    }
  } else if (exponent == 0x1f) {
    // Infinity or NaN. The mantissa payload is preserved.
    float_bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    // Normalized half: rebias the exponent (15 -> 127).
    float_bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float value;
  std::memcpy(&value, &float_bits, sizeof(value));
  return value;
}

// A 16-bit IEEE 754 binary16 (half precision) floating point value stored
// using 2 bytes. The class is trivially copyable so it can be directly read
// from / written to data buffers used by geometry attributes.
class Float16 {
 public:
  Float16() : bits_(0) {}
  Float16(const Float16& other) = default;
  Float16& operator=(const Float16& other) = default;

  // Implicit conversions from arithmetic types. These are needed by generic
  // code that converts attribute values between different data types.
  Float16(float value) : bits_(FloatToFloat16Bits(value)) {}  // NOLINT
  Float16(double value)                                       // NOLINT
      : bits_(FloatToFloat16Bits(static_cast<float>(value))) {}
  template <typename T,
            typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
  Float16(T value)  // NOLINT
      : bits_(FloatToFloat16Bits(static_cast<float>(value))) {}

  // Implicit conversion to float. Conversion to other arithmetic types is
  // done by chaining this conversion with a standard conversion.
  operator float() const { return Float16BitsToFloat(bits_); }  // NOLINT

  // Creates a Float16 from its raw bit representation.
  static Float16 FromBits(uint16_t bits) {
    Float16 value;
    value.bits_ = bits;
    return value;
  }
  // Returns the raw bit representation of the value.
  uint16_t bits() const { return bits_; }

  Float16& operator+=(Float16 rhs) {
    return *this = Float16(static_cast<float>(*this) + static_cast<float>(rhs));
  }
  Float16& operator-=(Float16 rhs) {
    return *this = Float16(static_cast<float>(*this) - static_cast<float>(rhs));
  }
  Float16& operator*=(Float16 rhs) {
    return *this = Float16(static_cast<float>(*this) * static_cast<float>(rhs));
  }
  Float16& operator/=(Float16 rhs) {
    return *this = Float16(static_cast<float>(*this) / static_cast<float>(rhs));
  }

 private:
  uint16_t bits_;
};

static_assert(sizeof(Float16) == 2, "Float16 must be 2 bytes.");
static_assert(std::is_trivially_copyable<Float16>::value,
              "Float16 must be trivially copyable.");

}  // namespace draco

namespace std {

// Numeric limits for draco::Float16 mirroring the binary16 format.
template <>
class numeric_limits<draco::Float16> {
 public:
  static constexpr bool is_specialized = true;
  static draco::Float16 min() noexcept {
    return draco::Float16::FromBits(0x0400);  // 2^-14.
  }
  static draco::Float16 max() noexcept {
    return draco::Float16::FromBits(0x7bff);  // 65504.
  }
  static draco::Float16 lowest() noexcept {
    return draco::Float16::FromBits(0xfbff);  // -65504.
  }
  static draco::Float16 epsilon() noexcept {
    return draco::Float16::FromBits(0x1400);  // 2^-10.
  }
  static draco::Float16 infinity() noexcept {
    return draco::Float16::FromBits(0x7c00);
  }
  static draco::Float16 quiet_NaN() noexcept {
    return draco::Float16::FromBits(0x7e00);
  }
  static draco::Float16 denorm_min() noexcept {
    return draco::Float16::FromBits(0x0001);  // 2^-24.
  }
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int max_exponent = 16;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr bool has_denorm = true;
  static constexpr bool has_denorm_loss = true;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr bool traps = false;
  static constexpr bool tinyness_before = false;
  static constexpr float_round_style round_style = round_to_nearest;
};

}  // namespace std

#endif  // DRACO_CORE_FLOAT16_H_
