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
#include "draco/core/float16.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "draco/compression/config/compression_shared.h"
#include "draco/compression/decode.h"
#include "draco/compression/encode.h"
#include "draco/core/draco_test_base.h"
#include "draco/core/draco_types.h"
#include "draco/io/ply_decoder.h"
#include "draco/io/ply_encoder.h"
#include "draco/mesh/triangle_soup_mesh_builder.h"
#include "draco/point_cloud/point_cloud.h"

namespace draco {

class HalfFloatTest : public ::testing::Test {};

// Verifies that the new DT_FLOAT16 type is appended after all pre-existing
// types so that their bit-stream ordinals remain unchanged.
TEST_F(HalfFloatTest, TypeOrdinalsUnchanged) {
  EXPECT_EQ(static_cast<int>(DT_INVALID), 0);
  EXPECT_EQ(static_cast<int>(DT_INT8), 1);
  EXPECT_EQ(static_cast<int>(DT_UINT8), 2);
  EXPECT_EQ(static_cast<int>(DT_INT16), 3);
  EXPECT_EQ(static_cast<int>(DT_UINT16), 4);
  EXPECT_EQ(static_cast<int>(DT_INT32), 5);
  EXPECT_EQ(static_cast<int>(DT_UINT32), 6);
  EXPECT_EQ(static_cast<int>(DT_INT64), 7);
  EXPECT_EQ(static_cast<int>(DT_UINT64), 8);
  EXPECT_EQ(static_cast<int>(DT_FLOAT32), 9);
  EXPECT_EQ(static_cast<int>(DT_FLOAT64), 10);
  EXPECT_EQ(static_cast<int>(DT_BOOL), 11);
  EXPECT_EQ(static_cast<int>(DT_FLOAT16), 12);
  EXPECT_EQ(static_cast<int>(DT_TYPES_COUNT), 13);
}

TEST_F(HalfFloatTest, DataTypeProperties) {
  EXPECT_EQ(DataTypeLength(DT_FLOAT16), 2);
  EXPECT_FALSE(IsDataTypeIntegral(DT_FLOAT16));
  EXPECT_EQ(sizeof(Float16), 2u);
}

// All 65536 half precision bit patterns (including NaN, +/-Inf and
// subnormals) must survive a round trip through float and back.
TEST_F(HalfFloatTest, AllHalfBitPatternsRoundTrip) {
  for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
    const uint16_t half_bits = static_cast<uint16_t>(bits);
    const float value = Float16BitsToFloat(half_bits);
    EXPECT_EQ(FloatToFloat16Bits(value), half_bits)
        << "bit pattern 0x" << std::hex << bits;
    // Same check through the Float16 class interface.
    const Float16 half_value = Float16::FromBits(half_bits);
    const float converted = static_cast<float>(half_value);
    EXPECT_EQ(Float16(converted).bits(), half_bits);
  }
}

TEST_F(HalfFloatTest, HalfToFloatKnownValues) {
  EXPECT_EQ(Float16BitsToFloat(0x0000), 0.0f);
  const float neg_zero = Float16BitsToFloat(0x8000);
  EXPECT_EQ(neg_zero, 0.0f);
  EXPECT_TRUE(std::signbit(neg_zero));
  EXPECT_EQ(Float16BitsToFloat(0x3c00), 1.0f);
  EXPECT_EQ(Float16BitsToFloat(0xc000), -2.0f);
  EXPECT_EQ(Float16BitsToFloat(0x7bff), 65504.0f);
  EXPECT_EQ(Float16BitsToFloat(0xfbff), -65504.0f);
  // Smallest subnormal.
  EXPECT_EQ(Float16BitsToFloat(0x0001), 5.9604644775390625e-08f);
  // Largest subnormal.
  EXPECT_EQ(Float16BitsToFloat(0x03ff), 6.0975551605224609e-05f);
  // Smallest normalized.
  EXPECT_EQ(Float16BitsToFloat(0x0400), 6.103515625e-05f);
  EXPECT_TRUE(std::isinf(Float16BitsToFloat(0x7c00)));
  EXPECT_GT(Float16BitsToFloat(0x7c00), 0.0f);
  EXPECT_TRUE(std::isinf(Float16BitsToFloat(0xfc00)));
  EXPECT_LT(Float16BitsToFloat(0xfc00), 0.0f);
  EXPECT_TRUE(std::isnan(Float16BitsToFloat(0x7e00)));
  EXPECT_TRUE(std::isnan(Float16BitsToFloat(0x7c01)));
  EXPECT_TRUE(std::isnan(Float16BitsToFloat(0xfe00)));
}

TEST_F(HalfFloatTest, FloatToHalfConversions) {
  EXPECT_EQ(FloatToFloat16Bits(0.0f), 0x0000);
  EXPECT_EQ(FloatToFloat16Bits(-0.0f), 0x8000);
  EXPECT_EQ(FloatToFloat16Bits(1.0f), 0x3c00);
  EXPECT_EQ(FloatToFloat16Bits(-2.0f), 0xc000);
  EXPECT_EQ(FloatToFloat16Bits(0.1f), 0x2e66);
  EXPECT_EQ(FloatToFloat16Bits(65504.0f), 0x7bff);
  // Overflow converts to infinity.
  EXPECT_EQ(FloatToFloat16Bits(65520.0f), 0x7c00);
  EXPECT_EQ(FloatToFloat16Bits(70000.0f), 0x7c00);
  EXPECT_EQ(FloatToFloat16Bits(-70000.0f), 0xfc00);
  EXPECT_EQ(FloatToFloat16Bits(std::numeric_limits<float>::infinity()),
            0x7c00);
  EXPECT_EQ(FloatToFloat16Bits(-std::numeric_limits<float>::infinity()),
            0xfc00);
  // Subnormals.
  EXPECT_EQ(FloatToFloat16Bits(5.9604644775390625e-08f), 0x0001);
  EXPECT_EQ(FloatToFloat16Bits(-5.9604644775390625e-08f), 0x8001);
  // Values below 2^-25 round to zero (2^-25 itself is a tie that rounds to
  // the even zero).
  EXPECT_EQ(FloatToFloat16Bits(2.98023223876953125e-08f), 0x0000);
  EXPECT_EQ(FloatToFloat16Bits(1.0e-10f), 0x0000);
  // Round-to-nearest-even on normalized values: 1 + 2^-11 is a tie between
  // 1.0 and 1 + 2^-10 and rounds to the even 1.0.
  EXPECT_EQ(FloatToFloat16Bits(1.00048828125f), 0x3c00);
  // 1 + 3 * 2^-11 is a tie between 0x3c01 and 0x3c02 and rounds to the even
  // 0x3c02.
  EXPECT_EQ(FloatToFloat16Bits(1.00146484375f), 0x3c02);
  // NaN stays NaN and preserves the most significant payload bits.
  EXPECT_TRUE((FloatToFloat16Bits(std::numeric_limits<float>::quiet_NaN()) &
               0x7c00) == 0x7c00);
  EXPECT_TRUE((FloatToFloat16Bits(std::numeric_limits<float>::quiet_NaN()) &
               0x03ff) != 0);
  uint32_t signaling_nan_bits = 0x7f800001;
  float signaling_nan;
  std::memcpy(&signaling_nan, &signaling_nan_bits, sizeof(signaling_nan));
  const uint16_t converted_nan = FloatToFloat16Bits(signaling_nan);
  EXPECT_TRUE((converted_nan & 0x7c00) == 0x7c00);
  EXPECT_TRUE((converted_nan & 0x03ff) != 0);
}

TEST_F(HalfFloatTest, Float16ClassConversions) {
  Float16 value(1.5f);
  EXPECT_EQ(value.bits(), 0x3e00);
  EXPECT_EQ(static_cast<float>(value), 1.5f);
  value = Float16::FromBits(0x7c00);
  EXPECT_TRUE(std::isinf(static_cast<float>(value)));
  // Construction from integral types.
  EXPECT_EQ(Float16(2).bits(), 0x4000);
  EXPECT_EQ(Float16(static_cast<uint8_t>(3)).bits(), 0x4200);
  // Compound assignment used by generic attribute conversion code.
  Float16 normalized(1.0f);
  normalized /= Float16(255);
  EXPECT_NEAR(static_cast<float>(normalized), 1.0f / 255.0f, 1e-6f);
}

// Tests conversion between Float16 and other types through the
// GeometryAttribute interface.
TEST_F(HalfFloatTest, GeometryAttributeValueConversion) {
  GeometryAttribute ga;
  ga.Init(GeometryAttribute::GENERIC, nullptr, 1, DT_FLOAT16, false,
          DataTypeLength(DT_FLOAT16), 0);
  PointAttribute att(ga);
  ASSERT_TRUE(att.Reset(3));

  const Float16 in_values[3] = {Float16(2.0f), Float16::FromBits(0x7c00),
                                Float16::FromBits(0x7e00)};
  att.SetAttributeValue(AttributeValueIndex(0), &in_values[0]);
  att.SetAttributeValue(AttributeValueIndex(1), &in_values[1]);
  att.SetAttributeValue(AttributeValueIndex(2), &in_values[2]);

  // Convert to float.
  float out_float = 0.0f;
  ASSERT_TRUE(att.ConvertValue<float>(AttributeValueIndex(0), &out_float));
  EXPECT_EQ(out_float, 2.0f);
  ASSERT_TRUE(att.ConvertValue<float>(AttributeValueIndex(1), &out_float));
  EXPECT_TRUE(std::isinf(out_float));
  ASSERT_TRUE(att.ConvertValue<float>(AttributeValueIndex(2), &out_float));
  EXPECT_TRUE(std::isnan(out_float));

  // Convert to integer. Finite values convert, Inf and NaN must be rejected.
  int32_t out_int = 0;
  ASSERT_TRUE(att.ConvertValue<int32_t>(AttributeValueIndex(0), &out_int));
  EXPECT_EQ(out_int, 2);
  EXPECT_FALSE(att.ConvertValue<int32_t>(AttributeValueIndex(1), &out_int));
  EXPECT_FALSE(att.ConvertValue<int32_t>(AttributeValueIndex(2), &out_int));

  // Convert float to Float16 through the attribute interface.
  GeometryAttribute float_ga;
  float_ga.Init(GeometryAttribute::GENERIC, nullptr, 1, DT_FLOAT32, false,
                DataTypeLength(DT_FLOAT32), 0);
  PointAttribute float_att(float_ga);
  ASSERT_TRUE(float_att.Reset(1));
  const float float_value = 0.5f;
  float_att.SetAttributeValue(AttributeValueIndex(0), &float_value);
  Float16 out_half;
  ASSERT_TRUE(float_att.ConvertValue<Float16>(AttributeValueIndex(0),
                                              &out_half));
  EXPECT_EQ(out_half.bits(), 0x3800);
}

class HalfAttributeEncodingTest : public ::testing::Test {
 protected:
  // Creates a point cloud with a single 3-component Float16 position
  // attribute initialized with the provided bit patterns.
  std::unique_ptr<PointCloud> CreateHalfPointCloud(
      const std::vector<uint16_t>& position_bits) const {
    std::unique_ptr<PointCloud> pc(new PointCloud());
    const int num_points = static_cast<int>(position_bits.size() / 3);
    GeometryAttribute va;
    va.Init(GeometryAttribute::POSITION, nullptr, 3, DT_FLOAT16, false,
            DataTypeLength(DT_FLOAT16) * 3, 0);
    const int att_id = pc->AddAttribute(va, true, num_points);
    PointAttribute* const att = pc->attribute(att_id);
    for (int i = 0; i < num_points; ++i) {
      const Float16 values[3] = {
          Float16::FromBits(position_bits[3 * i]),
          Float16::FromBits(position_bits[3 * i + 1]),
          Float16::FromBits(position_bits[3 * i + 2])};
      att->SetAttributeValue(AttributeValueIndex(i), values);
    }
    pc->set_num_points(num_points);
    return pc;
  }

  // Encodes and decodes the point cloud with the given encoder and returns
  // the decoded point cloud.
  std::unique_ptr<PointCloud> EncodeAndDecode(const PointCloud& pc,
                                              Encoder* encoder) const {
    EncoderBuffer buffer;
    if (!encoder->EncodePointCloudToBuffer(pc, &buffer).ok()) {
      return nullptr;
    }
    Decoder decoder;
    DecoderBuffer decoder_buffer;
    decoder_buffer.Init(buffer.data(), buffer.size());
    auto decoded = decoder.DecodePointCloudFromBuffer(&decoder_buffer);
    if (!decoded.ok()) {
      return nullptr;
    }
    return std::move(decoded).value();
  }

  // Verifies that the position attribute of |decoded| contains exactly the
  // half precision bit patterns of |position_bits|.
  void ExpectHalfPositions(const PointCloud& decoded,
                           const std::vector<uint16_t>& position_bits) const {
    const PointAttribute* const att =
        decoded.GetNamedAttribute(GeometryAttribute::POSITION);
    ASSERT_NE(att, nullptr);
    ASSERT_EQ(att->data_type(), DT_FLOAT16);
    ASSERT_EQ(att->num_components(), 3);
    const int num_points = static_cast<int>(position_bits.size() / 3);
    ASSERT_EQ(att->size(), num_points);
    for (int i = 0; i < num_points; ++i) {
      const auto value = att->GetValue<Float16, 3>(AttributeValueIndex(i));
      for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(value[c].bits(), position_bits[3 * i + c])
            << "point " << i << " component " << c;
      }
    }
  }
};

// Lossless encoding must preserve all bit patterns exactly, including NaN,
// +/-Inf, subnormals and -0.0.
TEST_F(HalfAttributeEncodingTest, LosslessPointCloudRoundTrip) {
  const std::vector<uint16_t> bits = {
      0x0000, 0x8000, 0x3c00,  // +0.0, -0.0, 1.0
      0x7c00, 0xfc00, 0x7bff,  // +Inf, -Inf, 65504
      0x0001, 0x8001, 0x03ff,  // subnormals
      0x7e00, 0x7c01, 0xfe00,  // NaNs
      0x3555, 0xc000, 0x2e66,  // regular values
  };
  const std::unique_ptr<PointCloud> pc = CreateHalfPointCloud(bits);
  Encoder encoder;
  encoder.SetEncodingMethod(POINT_CLOUD_SEQUENTIAL_ENCODING);
  const std::unique_ptr<PointCloud> decoded = EncodeAndDecode(*pc, &encoder);
  ASSERT_NE(decoded, nullptr);
  ExpectHalfPositions(*decoded, bits);
}

// Lossless mesh encoding must preserve half precision bit patterns as well.
TEST_F(HalfAttributeEncodingTest, LosslessMeshRoundTrip) {
  TriangleSoupMeshBuilder builder;
  builder.Start(2);
  const int pos_att_id =
      builder.AddAttribute(GeometryAttribute::POSITION, 3, DT_FLOAT32);
  const int nrm_att_id =
      builder.AddAttribute(GeometryAttribute::NORMAL, 3, DT_FLOAT16);
  const float pos[4][3] = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f},
                           {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};
  // Normal bit patterns including NaN, +/-Inf and subnormals.
  const uint16_t nrm_bits[4][3] = {
      {0x3c00, 0x0000, 0x8000},
      {0x7c00, 0xfc00, 0x0001},
      {0x7e00, 0x7c01, 0x03ff},
      {0x2e66, 0xc000, 0x7bff},
  };
  Float16 nrm[4][3];
  for (int i = 0; i < 4; ++i) {
    for (int c = 0; c < 3; ++c) {
      nrm[i][c] = Float16::FromBits(nrm_bits[i][c]);
    }
  }
  builder.SetAttributeValuesForFace(pos_att_id, FaceIndex(0), pos[0], pos[1],
                                    pos[2]);
  builder.SetAttributeValuesForFace(nrm_att_id, FaceIndex(0), nrm[0], nrm[1],
                                    nrm[2]);
  builder.SetAttributeValuesForFace(pos_att_id, FaceIndex(1), pos[0], pos[2],
                                    pos[3]);
  builder.SetAttributeValuesForFace(nrm_att_id, FaceIndex(1), nrm[0], nrm[2],
                                    nrm[3]);
  std::unique_ptr<Mesh> mesh = builder.Finalize();
  ASSERT_NE(mesh, nullptr);

  Encoder encoder;
  EncoderBuffer buffer;
  ASSERT_TRUE(encoder.EncodeMeshToBuffer(*mesh, &buffer).ok());
  Decoder decoder;
  DecoderBuffer decoder_buffer;
  decoder_buffer.Init(buffer.data(), buffer.size());
  auto decoded = decoder.DecodeMeshFromBuffer(&decoder_buffer);
  ASSERT_TRUE(decoded.ok());
  const PointAttribute* const att =
      decoded.value()->GetNamedAttribute(GeometryAttribute::NORMAL);
  ASSERT_NE(att, nullptr);
  ASSERT_EQ(att->data_type(), DT_FLOAT16);
  // Collect all decoded normal bit patterns and compare them as multisets
  // (mesh decoding may reorder attribute values).
  std::vector<uint16_t> expected_bits;
  std::vector<uint16_t> decoded_bits;
  for (int i = 0; i < 4; ++i) {
    for (int c = 0; c < 3; ++c) {
      expected_bits.push_back(nrm_bits[i][c]);
    }
  }
  ASSERT_EQ(att->size(), 4);
  for (int i = 0; i < 4; ++i) {
    const auto value = att->GetValue<Float16, 3>(AttributeValueIndex(i));
    for (int c = 0; c < 3; ++c) {
      decoded_bits.push_back(value[c].bits());
    }
  }
  std::sort(expected_bits.begin(), expected_bits.end());
  std::sort(decoded_bits.begin(), decoded_bits.end());
  EXPECT_EQ(decoded_bits, expected_bits);
}

// Quantized encoding of half precision values must decode to values close to
// the originals.
TEST_F(HalfAttributeEncodingTest, QuantizedPointCloudRoundTrip) {
  std::vector<uint16_t> bits;
  const float float_values[] = {-10.f, -1.5f, -0.0f, 0.25f, 1.f, 3.5f, 10.f};
  for (int i = 0; i < 7; ++i) {
    const Float16 value(float_values[i]);
    bits.push_back(value.bits());
    bits.push_back(value.bits());
    bits.push_back(value.bits());
  }
  const std::unique_ptr<PointCloud> pc = CreateHalfPointCloud(bits);
  Encoder encoder;
  encoder.SetEncodingMethod(POINT_CLOUD_SEQUENTIAL_ENCODING);
  encoder.SetAttributeQuantization(GeometryAttribute::POSITION, 12);
  const std::unique_ptr<PointCloud> decoded = EncodeAndDecode(*pc, &encoder);
  ASSERT_NE(decoded, nullptr);
  const PointAttribute* const att =
      decoded->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(att, nullptr);
  ASSERT_EQ(att->data_type(), DT_FLOAT16);
  ASSERT_EQ(att->size(), 7);
  for (int i = 0; i < 7; ++i) {
    const auto value = att->GetValue<Float16, 3>(AttributeValueIndex(i));
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(static_cast<float>(value[c]), float_values[i], 0.02f)
          << "point " << i << " component " << c;
    }
  }
}

// The kd-tree encoder must support quantized half precision attributes. Note
// that the kd-tree decoder does not preserve the order of points (this is
// the same behavior as for float32 attributes), so the values are compared
// as sorted sequences.
TEST_F(HalfAttributeEncodingTest, QuantizedKdTreeRoundTrip) {
  std::vector<uint16_t> bits;
  std::vector<float> float_values;
  for (int i = 0; i < 32; ++i) {
    const float value = -5.0f + 0.25f * static_cast<float>(i);
    float_values.push_back(value);
    const Float16 half_value(value);
    bits.push_back(half_value.bits());
    bits.push_back(half_value.bits());
    bits.push_back(half_value.bits());
  }
  const std::unique_ptr<PointCloud> pc = CreateHalfPointCloud(bits);
  Encoder encoder;
  encoder.SetEncodingMethod(POINT_CLOUD_KD_TREE_ENCODING);
  encoder.SetAttributeQuantization(GeometryAttribute::POSITION, 12);
  const std::unique_ptr<PointCloud> decoded = EncodeAndDecode(*pc, &encoder);
  ASSERT_NE(decoded, nullptr);
  const PointAttribute* const att =
      decoded->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(att, nullptr);
  ASSERT_EQ(att->data_type(), DT_FLOAT16);
  ASSERT_EQ(att->size(), 32);
  std::vector<float> decoded_values;
  for (int i = 0; i < 32; ++i) {
    const auto value = att->GetValue<Float16, 3>(AttributeValueIndex(i));
    for (int c = 0; c < 3; ++c) {
      decoded_values.push_back(static_cast<float>(value[c]));
    }
  }
  std::vector<float> expected_values;
  for (int i = 0; i < 32; ++i) {
    for (int c = 0; c < 3; ++c) {
      expected_values.push_back(float_values[i]);
    }
  }
  std::sort(decoded_values.begin(), decoded_values.end());
  std::sort(expected_values.begin(), expected_values.end());
  for (int i = 0; i < 96; ++i) {
    EXPECT_NEAR(decoded_values[i], expected_values[i], 0.02f) << "index " << i;
  }
}

// Half precision attributes must survive a round trip through the PLY format
// (stored as "half" properties).
TEST_F(HalfAttributeEncodingTest, PlyRoundTrip) {
  const std::vector<uint16_t> bits = {
      0x0000, 0x8000, 0x3c00,  // +0.0, -0.0, 1.0
      0x7c00, 0xfc00, 0x7bff,  // +Inf, -Inf, 65504
      0x0001, 0x8001, 0x03ff,  // subnormals
      0x7e00, 0x7c01, 0xfe00,  // NaNs
  };
  const std::unique_ptr<PointCloud> pc = CreateHalfPointCloud(bits);
  PlyEncoder ply_encoder;
  EncoderBuffer buffer;
  ASSERT_TRUE(ply_encoder.EncodeToBuffer(*pc, &buffer));

  DecoderBuffer decoder_buffer;
  decoder_buffer.Init(buffer.data(), buffer.size());
  PointCloud decoded;
  PlyDecoder ply_decoder;
  ASSERT_TRUE(ply_decoder.DecodeFromBuffer(&decoder_buffer, &decoded).ok());
  ExpectHalfPositions(decoded, bits);
}

// Deduplication must work for half precision attribute values.
TEST_F(HalfAttributeEncodingTest, DeduplicateHalfValues) {
  // 6 points but only 3 unique values (note that +0.0 and -0.0 have distinct
  // bit patterns and are treated as different values).
  const std::vector<uint16_t> bits = {
      0x3c00, 0x0000, 0x8000, 0x3c00, 0x0000, 0x8000,
      0x3c00, 0x0000, 0x8000, 0x2e66, 0x2e66, 0x2e66,
      0x7bff, 0x7bff, 0x7bff, 0x2e66, 0x2e66, 0x2e66,
  };
  std::unique_ptr<PointCloud> pc = CreateHalfPointCloud(bits);
  ASSERT_TRUE(pc->DeduplicateAttributeValues());
  const PointAttribute* const att =
      pc->GetNamedAttribute(GeometryAttribute::POSITION);
  ASSERT_NE(att, nullptr);
  ASSERT_EQ(att->data_type(), DT_FLOAT16);
  // 3 unique values: (1,0,-0) repeated 3 times, (0.1,...) repeated twice
  // and (65504,...) once. Note that +0.0 and -0.0 have distinct bit patterns
  // and are treated as different values.
  EXPECT_EQ(att->size(), 3);
  // The point cloud must still map 6 points to the deduplicated values.
  EXPECT_EQ(pc->num_points(), 6);
}

}  // namespace draco
