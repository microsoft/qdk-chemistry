// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>

using namespace qdk::chemistry::data;

namespace {

std::shared_ptr<const Symmetries> spin_symmetries() {
  return std::make_shared<const Symmetries>(Symmetries({axes::spin(0, false)}));
}

}  // namespace

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
