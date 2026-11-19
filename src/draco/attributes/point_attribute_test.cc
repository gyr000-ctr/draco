// Copyright 2017 The Draco Authors.
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
#include "draco/attributes/point_attribute.h"

#include <array>
#include <limits>

#include "draco/core/draco_test_base.h"

namespace {

class PointAttributeTest : public ::testing::Test {
 protected:
  PointAttributeTest() = default;
};

TEST_F(PointAttributeTest, TestCopy) {
  // This test verifies that PointAttribute can copy data from another point
  // attribute.
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::POSITION, 1, draco::DT_INT32, false, 10);

  for (int32_t i = 0; i < 10; ++i) {
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &i);
  }

  pa.set_unique_id(12);

  draco::PointAttribute other_pa;
  other_pa.CopyFrom(pa);

  draco::PointAttributeHasher hasher;
  ASSERT_EQ(hasher(pa), hasher(other_pa));
  ASSERT_EQ(pa.unique_id(), other_pa.unique_id());

  // The hash function does not actually compute the hash from attribute values,
  // so ensure the data got copied correctly as well.
  for (int32_t i = 0; i < 10; ++i) {
    int32_t data;
    other_pa.GetValue(draco::AttributeValueIndex(i), &data);
    ASSERT_EQ(data, i);
  }
}

TEST_F(PointAttributeTest, TestGetValueFloat) {
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32, false, 5);
  float points[3];
  for (int32_t i = 0; i < 5; ++i) {
    points[0] = i * 3.0;
    points[1] = (i * 3.0) + 1.0;
    points[2] = (i * 3.0) + 2.0;
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &points);
  }

  for (int32_t i = 0; i < 5; ++i) {
    pa.GetValue(draco::AttributeValueIndex(i), &points);
    ASSERT_FLOAT_EQ(points[0], i * 3.0);
    ASSERT_FLOAT_EQ(points[1], (i * 3.0) + 1.0);
    ASSERT_FLOAT_EQ(points[2], (i * 3.0) + 2.0);
  }
}

TEST_F(PointAttributeTest, TestGetArray) {
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32, false, 5);
  float points[3];
  for (int32_t i = 0; i < 5; ++i) {
    points[0] = i * 3.0;
    points[1] = (i * 3.0) + 1.0;
    points[2] = (i * 3.0) + 2.0;
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &points);
  }

  for (int32_t i = 0; i < 5; ++i) {
    std::array<float, 3> att_value;
    att_value = pa.GetValue<float, 3>(draco::AttributeValueIndex(i));
    ASSERT_FLOAT_EQ(att_value[0], i * 3.0);
    ASSERT_FLOAT_EQ(att_value[1], (i * 3.0) + 1.0);
    ASSERT_FLOAT_EQ(att_value[2], (i * 3.0) + 2.0);
  }
  for (int32_t i = 0; i < 5; ++i) {
    std::array<float, 3> att_value;
    EXPECT_TRUE(
        (pa.GetValue<float, 3>(draco::AttributeValueIndex(i), &att_value)));
    ASSERT_FLOAT_EQ(att_value[0], i * 3.0);
    ASSERT_FLOAT_EQ(att_value[1], (i * 3.0) + 1.0);
    ASSERT_FLOAT_EQ(att_value[2], (i * 3.0) + 2.0);
  }
}

TEST_F(PointAttributeTest, TestArrayReadError) {
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32, false, 5);
  float points[3];
  for (int32_t i = 0; i < 5; ++i) {
    points[0] = i * 3.0;
    points[1] = (i * 3.0) + 1.0;
    points[2] = (i * 3.0) + 2.0;
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &points);
  }

  std::array<float, 3> att_value;
  EXPECT_FALSE(
      (pa.GetValue<float, 3>(draco::AttributeValueIndex(5), &att_value)));
}

TEST_F(PointAttributeTest, TestResize) {
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::POSITION, 3, draco::DT_FLOAT32, false, 5);
  ASSERT_EQ(pa.size(), 5);
  ASSERT_EQ(pa.buffer()->data_size(), 4 * 3 * 5);

  pa.Resize(10);
  ASSERT_EQ(pa.size(), 10);
  ASSERT_EQ(pa.buffer()->data_size(), 4 * 3 * 10);
}

TEST_F(PointAttributeTest, NormalizedUint16ToUint8Rescaling) {
  // Converting a normalized uint16 attribute to uint8 must re-scale the values
  // through the shared 0-1 semantics rather than truncating them. Endpoints
  // must map exactly: 0 -> 0 and 65535 -> 255.
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::COLOR, 1, draco::DT_UINT16, true, 4);
  const uint16_t values[] = {0, 32768, 32767, 65535};
  for (int i = 0; i < 4; ++i) {
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &values[i]);
  }

  std::array<uint8_t, 1> out;
  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(0), out.data())));
  EXPECT_EQ(out[0], 0);

  // 32768 / 65535 * 255 = 127.500..., rounds to 128 (the midpoint).
  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(1), out.data())));
  EXPECT_EQ(out[0], 128);

  // 32767 / 65535 * 255 = 127.496..., rounds to 127.
  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(2), out.data())));
  EXPECT_EQ(out[0], 127);

  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(3), out.data())));
  EXPECT_EQ(out[0], 255);
}

TEST_F(PointAttributeTest, NormalizedInt16ToInt8Rescaling) {
  // Signed normalized integers must re-scale through their 0-1 semantics as
  // well, using the symmetric max-value mapping that is consistent with the
  // integer-to-float normalized conversion. The positive maximum 32767 must
  // map to 127 and 0 must stay 0.
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::GENERIC, 1, draco::DT_INT16, true, 3);
  const int16_t values[] = {0, 32767, -32768};
  for (int i = 0; i < 3; ++i) {
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &values[i]);
  }

  std::array<int8_t, 1> out;
  ASSERT_TRUE((pa.ConvertValue<int8_t, 1>(
      draco::AttributeValueIndex(0), out.data())));
  EXPECT_EQ(out[0], 0);

  ASSERT_TRUE((pa.ConvertValue<int8_t, 1>(
      draco::AttributeValueIndex(1), out.data())));
  EXPECT_EQ(out[0], 127);

  // -32768 / 32767 * 127 = -127.004..., rounds to -127.
  ASSERT_TRUE((pa.ConvertValue<int8_t, 1>(
      draco::AttributeValueIndex(2), out.data())));
  EXPECT_EQ(out[0], -127);
}

TEST_F(PointAttributeTest, NormalizedUint8ToUint16Rescaling) {
  // Widening a normalized attribute must map the uint8 endpoints to the uint16
  // endpoints exactly.
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::COLOR, 1, draco::DT_UINT8, true, 3);
  const uint8_t values[] = {0, 128, 255};
  for (int i = 0; i < 3; ++i) {
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &values[i]);
  }

  std::array<uint16_t, 1> out;
  ASSERT_TRUE((pa.ConvertValue<uint16_t, 1>(
      draco::AttributeValueIndex(0), out.data())));
  EXPECT_EQ(out[0], 0);

  // 128 / 255 * 65535 = 32896.47..., rounds to 32896.
  ASSERT_TRUE((pa.ConvertValue<uint16_t, 1>(
      draco::AttributeValueIndex(1), out.data())));
  EXPECT_EQ(out[0], 32896);

  ASSERT_TRUE((pa.ConvertValue<uint16_t, 1>(
      draco::AttributeValueIndex(2), out.data())));
  EXPECT_EQ(out[0], 65535);
}

TEST_F(PointAttributeTest, NormalizedIntegerConversionSaturates) {
  // On the normalized path every integer value represents a valid 0-1 sample,
  // so conversions must saturate to the output endpoints instead of failing.
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::GENERIC, 1, draco::DT_INT16, true, 2);
  const int16_t values[] = {-32768, 32767};
  for (int i = 0; i < 2; ++i) {
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &values[i]);
  }

  std::array<uint8_t, 1> out;
  // -32768 / 32767 is negative, which cannot be represented by uint8 and must
  // be saturated to 0 rather than rejected.
  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(0), out.data())));
  EXPECT_EQ(out[0], 0);

  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(1), out.data())));
  EXPECT_EQ(out[0], 255);
}

TEST_F(PointAttributeTest, NormalizedFloatNaNToIntegerIsRejected) {
  // A NaN (or out-of-range) normalized float must be rejected rather than
  // cast into an integer, which would be undefined behavior.
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::COLOR, 1, draco::DT_FLOAT32, true, 3);
  const float values[] = {std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f};
  for (int i = 0; i < 3; ++i) {
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &values[i]);
  }

  std::array<uint8_t, 1> out;
  EXPECT_FALSE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(0), out.data())));
  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(1), out.data())));
  EXPECT_EQ(out[0], 255);
  EXPECT_FALSE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(2), out.data())));
}

TEST_F(PointAttributeTest, NormalizedSameWidthIsIdentity) {
  // Re-scaling between integer types of the same width must leave every value
  // unchanged.
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::COLOR, 1, draco::DT_UINT16, true, 4);
  const uint16_t values[] = {0, 123, 32768, 65535};
  for (int i = 0; i < 4; ++i) {
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &values[i]);
  }

  for (int i = 0; i < 4; ++i) {
    std::array<uint16_t, 1> out;
    ASSERT_TRUE((pa.ConvertValue<uint16_t, 1>(
        draco::AttributeValueIndex(i), out.data())));
    EXPECT_EQ(out[0], values[i]);
  }
}

TEST_F(PointAttributeTest, NonNormalizedIntegerConversionIsUnchanged) {
  // Non-normalized integer attributes must keep the per-sample identity
  // semantics: values that fit are converted verbatim and values that do not
  // fit into the output range must be rejected instead of being re-scaled.
  draco::PointAttribute pa;
  pa.Init(draco::GeometryAttribute::GENERIC, 1, draco::DT_UINT16, false, 3);
  const uint16_t values[] = {0, 255, 65535};
  for (int i = 0; i < 3; ++i) {
    pa.SetAttributeValue(draco::AttributeValueIndex(i), &values[i]);
  }

  std::array<uint8_t, 1> out;
  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(0), out.data())));
  EXPECT_EQ(out[0], 0);
  ASSERT_TRUE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(1), out.data())));
  EXPECT_EQ(out[0], 255);
  // 65535 cannot be represented by uint8 on the non-normalized path.
  EXPECT_FALSE((pa.ConvertValue<uint8_t, 1>(
      draco::AttributeValueIndex(2), out.data())));
}

}  // namespace
