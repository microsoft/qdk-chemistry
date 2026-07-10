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
#include <utility>

using namespace qdk::chemistry::data;

using SBS = SymmetryBlockedScalar<std::size_t>;

namespace qdk::chemistry::tests::test_support {

// Build a spin-blocked count with independent (non-aliased) alpha/beta blocks.
SBS make_spin_blocked(std::size_t alpha_value, std::size_t beta_value) {
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/false)}));
  SBS::BlockMap blocks;
  blocks[{axes::alpha()}] = std::make_shared<const std::size_t>(alpha_value);
  blocks[{axes::beta()}] = std::make_shared<const std::size_t>(beta_value);
  return SBS(SBS::SymmetriesArray{sym}, std::move(blocks));
}

// Build a trivial (axis-free) count carrying a single aggregate block.
SBS make_trivial_blocked(std::size_t value) {
  auto sym =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  SBS::BlockMap blocks;
  blocks[{SymmetryLabel{}}] = std::make_shared<const std::size_t>(value);
  return SBS(SBS::SymmetriesArray{sym}, std::move(blocks));
}

}  // namespace qdk::chemistry::tests::test_support

namespace test_support = qdk::chemistry::tests::test_support;

TEST(SymmetryBlockedScalarTest, SpinBlockedHoldsIndependentChannels) {
  auto scalar = test_support::make_spin_blocked(5, 3);

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
  auto scalar = test_support::make_trivial_blocked(8);

  EXPECT_FALSE(scalar.symmetries()[0]->has_axis(AxisName::Spin));
  EXPECT_EQ(scalar.num_blocks(), 1u);
  // The single block resolves via the trivial (empty) label.
  EXPECT_EQ(scalar.value(SymmetryLabel{}), 8u);
}

TEST(SymmetryBlockedScalarTest, MissingBlockThrows) {
  auto scalar = test_support::make_trivial_blocked(4);
  EXPECT_THROW(scalar.value(axes::alpha()), std::invalid_argument);
}

TEST(SymmetryBlockedScalarTest, JsonRoundTripSpinBlocked) {
  auto scalar = test_support::make_spin_blocked(7, 2);
  auto restored = SBS::from_json(scalar.to_json());

  EXPECT_EQ(restored->value(axes::alpha()), 7u);
  EXPECT_EQ(restored->value(axes::beta()), 2u);
  EXPECT_EQ(restored->num_blocks(), 2u);
}

TEST(SymmetryBlockedScalarTest, JsonRoundTripTrivial) {
  auto scalar = test_support::make_trivial_blocked(11);
  auto restored = SBS::from_json(scalar.to_json());

  EXPECT_FALSE(restored->symmetries()[0]->has_axis(AxisName::Spin));
  EXPECT_EQ(restored->value(SymmetryLabel{}), 11u);
}

TEST(SymmetryBlockedScalarTest, Hdf5RoundTrip) {
  const std::filesystem::path filename = "symmetry_blocked_scalar_roundtrip.h5";
  std::filesystem::remove(filename);

  auto scalar = test_support::make_spin_blocked(6, 4);
  scalar.to_hdf5_file(filename.string());
  auto restored = SBS::from_hdf5_file(filename.string());

  EXPECT_EQ(restored->value(axes::alpha()), 6u);
  EXPECT_EQ(restored->value(axes::beta()), 4u);
  std::filesystem::remove(filename);
}

TEST(SymmetryBlockedScalarTest, JsonFromJsonRejectsMissingVersion) {
  auto j = test_support::make_trivial_blocked(1).to_json();
  j.erase("version");
  EXPECT_THROW(SBS::from_json(j), std::runtime_error);
}

TEST(SymmetryBlockedScalarTest, JsonFromJsonRejectsMismatchedVersion) {
  auto j = test_support::make_trivial_blocked(1).to_json();
  j["version"] = "99.0.0";
  EXPECT_THROW(SBS::from_json(j), std::runtime_error);
}
