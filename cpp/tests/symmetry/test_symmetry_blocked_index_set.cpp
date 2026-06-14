// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <stdexcept>

using namespace qdk::chemistry::data;

static std::shared_ptr<const SymmetryProduct> spin_symmetries() {
  return std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, false)}));
}

TEST(SymmetryBlockedIndexSetTest, StoresAndReturnsIndices) {
  std::unordered_map<SymmetryLabel, std::size_t> extents;
  extents.emplace(SymmetryLabel({axes::alpha()}), 5);
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
  indices.emplace(SymmetryLabel({axes::alpha()}),
                  std::vector<std::uint32_t>{0, 2, 4});

  SymmetryBlockedIndexSet set(spin_symmetries(), extents, indices);

  EXPECT_TRUE(set.has(SymmetryLabel({axes::alpha()})));
  auto view = set.indices(SymmetryLabel({axes::alpha()}));
  ASSERT_EQ(view.size(), 3u);
  EXPECT_EQ(view[0], 0u);
  EXPECT_EQ(view[2], 4u);
  EXPECT_EQ(set.labels().size(), 1u);
}

TEST(SymmetryBlockedIndexSetTest, OutOfRangeRejected) {
  std::unordered_map<SymmetryLabel, std::size_t> extents;
  extents.emplace(SymmetryLabel({axes::alpha()}), 3);
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
  indices.emplace(SymmetryLabel({axes::alpha()}),
                  std::vector<std::uint32_t>{0, 3});

  EXPECT_THROW(SymmetryBlockedIndexSet(spin_symmetries(), extents, indices),
               std::out_of_range);
}

TEST(SymmetryBlockedIndexSetTest, UnsortedRejected) {
  std::unordered_map<SymmetryLabel, std::size_t> extents;
  extents.emplace(SymmetryLabel({axes::alpha()}), 5);
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
  indices.emplace(SymmetryLabel({axes::alpha()}),
                  std::vector<std::uint32_t>{2, 1});

  EXPECT_THROW(SymmetryBlockedIndexSet(spin_symmetries(), extents, indices),
               std::invalid_argument);
}

TEST(SymmetryBlockedIndexSetTest, DuplicateRejected) {
  std::unordered_map<SymmetryLabel, std::size_t> extents;
  extents.emplace(SymmetryLabel({axes::alpha()}), 5);
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
  indices.emplace(SymmetryLabel({axes::alpha()}),
                  std::vector<std::uint32_t>{1, 1});

  EXPECT_THROW(SymmetryBlockedIndexSet(spin_symmetries(), extents, indices),
               std::invalid_argument);
}

TEST(SymmetryBlockedIndexSetTest, MissingLabelThrows) {
  std::unordered_map<SymmetryLabel, std::size_t> extents;
  extents.emplace(SymmetryLabel({axes::alpha()}), 5);
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
  indices.emplace(SymmetryLabel({axes::alpha()}),
                  std::vector<std::uint32_t>{0, 1});
  SymmetryBlockedIndexSet set(spin_symmetries(), extents, indices);

  EXPECT_THROW(set.indices(SymmetryLabel({axes::beta()})),
               std::invalid_argument);
}

TEST(SymmetryBlockedIndexSetTest, JsonRoundTrip) {
  std::unordered_map<SymmetryLabel, std::size_t> extents;
  extents.emplace(SymmetryLabel({axes::alpha()}), 5);
  extents.emplace(SymmetryLabel({axes::beta()}), 5);
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
  indices.emplace(SymmetryLabel({axes::alpha()}),
                  std::vector<std::uint32_t>{0, 1, 2});
  indices.emplace(SymmetryLabel({axes::beta()}),
                  std::vector<std::uint32_t>{3, 4});
  SymmetryBlockedIndexSet set(spin_symmetries(), extents, indices);

  auto restored = SymmetryBlockedIndexSet::from_json(set.to_json());
  auto view = restored->indices(SymmetryLabel({axes::alpha()}));
  ASSERT_EQ(view.size(), 3u);
  EXPECT_EQ(view[1], 1u);
  EXPECT_EQ(restored->indices(SymmetryLabel({axes::beta()})).size(), 2u);
}

// Verify the per-label extent contract documented on the ctor: alpha and beta
// labels may have unequal universe sizes AND unequal subset sizes, and both
// must be preserved through construction, accessors, JSON, and HDF5 I/O.
TEST(SymmetryBlockedIndexSetTest, AsymmetricAlphaBetaExtents) {
  const SymmetryLabel alpha({axes::alpha()});
  const SymmetryLabel beta({axes::beta()});

  std::unordered_map<SymmetryLabel, std::size_t> extents;
  extents.emplace(alpha, 8u);  // universe size 8 for alpha
  extents.emplace(beta, 6u);   // universe size 6 for beta
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices;
  indices.emplace(alpha, std::vector<std::uint32_t>{0, 3, 5, 7});  // 4 of 8
  indices.emplace(beta, std::vector<std::uint32_t>{1, 2});         // 2 of 6
  SymmetryBlockedIndexSet set(spin_symmetries(), extents, indices);

  // Both labels present, with their independent extents and subset sizes.
  EXPECT_TRUE(set.has(alpha));
  EXPECT_TRUE(set.has(beta));
  EXPECT_EQ(set.labels().size(), 2u);
  EXPECT_EQ(set.extents().at(alpha), 8u);
  EXPECT_EQ(set.extents().at(beta), 6u);

  auto alpha_view = set.indices(alpha);
  auto beta_view = set.indices(beta);
  ASSERT_EQ(alpha_view.size(), 4u);
  ASSERT_EQ(beta_view.size(), 2u);
  EXPECT_EQ(alpha_view[0], 0u);
  EXPECT_EQ(alpha_view[3], 7u);
  EXPECT_EQ(beta_view[0], 1u);
  EXPECT_EQ(beta_view[1], 2u);

  // JSON round-trip preserves both labels' independent extents and indices.
  auto restored_json = SymmetryBlockedIndexSet::from_json(set.to_json());
  EXPECT_EQ(restored_json->extents().at(alpha), 8u);
  EXPECT_EQ(restored_json->extents().at(beta), 6u);
  EXPECT_EQ(restored_json->indices(alpha).size(), 4u);
  EXPECT_EQ(restored_json->indices(beta).size(), 2u);
  EXPECT_EQ(restored_json->indices(alpha)[3], 7u);
  EXPECT_EQ(restored_json->indices(beta)[1], 2u);

  // HDF5 round-trip preserves the same.
  const std::filesystem::path filename =
      "symmetry_blocked_index_set_asymmetric.h5";
  std::filesystem::remove(filename);
  set.to_hdf5_file(filename.string());
  auto restored_h5 = SymmetryBlockedIndexSet::from_hdf5_file(filename.string());
  std::filesystem::remove(filename);
  EXPECT_EQ(restored_h5->extents().at(alpha), 8u);
  EXPECT_EQ(restored_h5->extents().at(beta), 6u);
  EXPECT_EQ(restored_h5->indices(alpha).size(), 4u);
  EXPECT_EQ(restored_h5->indices(beta).size(), 2u);
  EXPECT_EQ(restored_h5->indices(alpha)[2], 5u);
  EXPECT_EQ(restored_h5->indices(beta)[0], 1u);
}

// Per-label extents are enforced independently: an index that would be legal
// for one label's universe must still be rejected when supplied for a label
// whose universe is smaller.
TEST(SymmetryBlockedIndexSetTest, AsymmetricExtentsEnforcedPerLabel) {
  const SymmetryLabel alpha({axes::alpha()});
  const SymmetryLabel beta({axes::beta()});

  {
    // 8 is out of range for alpha's universe of size 8 (valid range [0,8)).
    std::unordered_map<SymmetryLabel, std::size_t> extents{{alpha, 8u},
                                                           {beta, 6u}};
    std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices{
        {alpha, {0u, 8u}}, {beta, {1u}}};
    EXPECT_THROW(SymmetryBlockedIndexSet(spin_symmetries(), extents, indices),
                 std::out_of_range);
  }
  {
    // 6 is out of range for beta's universe of size 6 (valid range [0,6))
    // even though it would be legal under alpha's universe of size 8.
    std::unordered_map<SymmetryLabel, std::size_t> extents{{alpha, 8u},
                                                           {beta, 6u}};
    std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices{
        {alpha, {0u, 5u}}, {beta, {0u, 6u}}};
    EXPECT_THROW(SymmetryBlockedIndexSet(spin_symmetries(), extents, indices),
                 std::out_of_range);
  }
}
