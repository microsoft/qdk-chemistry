// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_scalar.hpp>
#include <stdexcept>

using namespace qdk::chemistry::data;

using SBS = SymmetryBlockedScalar<std::size_t>;

TEST(SymmetryBlockedScalarTest, SpinBlockedHoldsIndependentChannels) {
  auto scalar = make_spin_blocked_scalar<std::size_t>(5, 3);

  EXPECT_TRUE(scalar.has_block({axes::alpha()}));
  EXPECT_TRUE(scalar.has_block({axes::beta()}));
  EXPECT_EQ(scalar.value(axes::alpha()), 5u);
  EXPECT_EQ(scalar.value(axes::beta()), 3u);
  // Independent channels are not aliased.
  EXPECT_NE(scalar.block_ptr({axes::alpha()}).get(),
            scalar.block_ptr({axes::beta()}).get());
  EXPECT_EQ(scalar.num_blocks(), 2u);
  EXPECT_TRUE(scalar.symmetries()[0]->has_axis(AxisName::Spin));
}

TEST(SymmetryBlockedScalarTest, TrivialHoldsAggregate) {
  auto scalar = make_trivial_blocked_scalar<std::size_t>(8);

  EXPECT_FALSE(scalar.symmetries()[0]->has_axis(AxisName::Spin));
  EXPECT_EQ(scalar.num_blocks(), 1u);
  // The single block resolves via the trivial (empty) label.
  EXPECT_EQ(scalar.value(SymmetryLabel{}), 8u);
}

TEST(SymmetryBlockedScalarTest, MissingBlockThrows) {
  auto scalar = make_trivial_blocked_scalar<std::size_t>(4);
  EXPECT_THROW(scalar.value(axes::alpha()), std::invalid_argument);
}

TEST(SymmetryBlockedScalarTest, JsonRoundTripSpinBlocked) {
  auto scalar = make_spin_blocked_scalar<std::size_t>(7, 2);
  auto restored = SBS::from_json(scalar.to_json());

  EXPECT_EQ(restored->value(axes::alpha()), 7u);
  EXPECT_EQ(restored->value(axes::beta()), 2u);
  EXPECT_EQ(restored->num_blocks(), 2u);
}

TEST(SymmetryBlockedScalarTest, JsonRoundTripTrivial) {
  auto scalar = make_trivial_blocked_scalar<std::size_t>(11);
  auto restored = SBS::from_json(scalar.to_json());

  EXPECT_FALSE(restored->symmetries()[0]->has_axis(AxisName::Spin));
  EXPECT_EQ(restored->value(SymmetryLabel{}), 11u);
}

TEST(SymmetryBlockedScalarTest, Hdf5RoundTrip) {
  const std::filesystem::path filename = "symmetry_blocked_scalar_roundtrip.h5";
  std::filesystem::remove(filename);

  auto scalar = make_spin_blocked_scalar<std::size_t>(6, 4);
  scalar.to_hdf5_file(filename.string());
  auto restored = SBS::from_hdf5_file(filename.string());

  EXPECT_EQ(restored->value(axes::alpha()), 6u);
  EXPECT_EQ(restored->value(axes::beta()), 4u);
  std::filesystem::remove(filename);
}

TEST(SymmetryBlockedScalarTest, JsonFromJsonRejectsMissingVersion) {
  auto j = make_trivial_blocked_scalar<std::size_t>(1).to_json();
  j.erase("version");
  EXPECT_THROW(SBS::from_json(j), std::runtime_error);
}

TEST(SymmetryBlockedScalarTest, JsonFromJsonRejectsMismatchedVersion) {
  auto j = make_trivial_blocked_scalar<std::size_t>(1).to_json();
  j["version"] = "99.0.0";
  EXPECT_THROW(SBS::from_json(j), std::runtime_error);
}
