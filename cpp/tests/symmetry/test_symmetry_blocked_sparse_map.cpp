// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_sparse_map.hpp>
#include <stdexcept>

using namespace qdk::chemistry::data;

using SBSM = SymmetryBlockedSparseMap<4, double>;

static SBSM make_unrestricted_sparse_map() {
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/false)}));
  std::unordered_map<SymmetryLabel, std::size_t> ext;
  ext[axes::alpha()] = 2;
  ext[axes::beta()] = 2;

  auto aaaa = std::make_shared<const SparseMapBlock<4, double>>(
      SparseMapBlock<4, double>{{{0, 0, 0, 0}, 1.5}, {{1, 1, 1, 1}, 2.5}});

  SBSM::BlockMap blocks;
  blocks[{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}] = aaaa;
  return SBSM({sym, sym, sym, sym}, {ext, ext, ext, ext}, std::move(blocks));
}

TEST(SymmetryBlockedSparseMapTest, EntriesAndLookup) {
  auto map = make_unrestricted_sparse_map();
  EXPECT_EQ(map.num_blocks(), 1u);
  EXPECT_EQ(map.num_entries(), 2u);
  EXPECT_DOUBLE_EQ(
      map.get({axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()},
              {0, 0, 0, 0}),
      1.5);
  // Absent entry returns zero.
  EXPECT_DOUBLE_EQ(
      map.get({axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()},
              {0, 1, 0, 1}),
      0.0);
}

TEST(SymmetryBlockedSparseMapTest, JsonRoundTrip) {
  auto map = make_unrestricted_sparse_map();
  auto restored = SBSM::from_json(map.to_json());
  EXPECT_EQ(restored->num_entries(), 2u);
  EXPECT_DOUBLE_EQ(restored->get({axes::alpha(), axes::alpha(), axes::alpha(),
                                  axes::alpha()},
                                 {1, 1, 1, 1}),
                   2.5);
}

TEST(SymmetryBlockedSparseMapTest, Hdf5RoundTrip) {
  const std::filesystem::path filename =
      "symmetry_blocked_sparse_map_roundtrip.h5";
  std::filesystem::remove(filename);

  auto map = make_unrestricted_sparse_map();
  map.to_hdf5_file(filename.string());
  auto restored = SBSM::from_hdf5_file(filename.string());

  EXPECT_EQ(restored->num_entries(), 2u);
  EXPECT_DOUBLE_EQ(restored->get({axes::alpha(), axes::alpha(), axes::alpha(),
                                  axes::alpha()},
                                 {0, 0, 0, 0}),
                   1.5);
  std::filesystem::remove(filename);
}

TEST(SymmetryBlockedSparseMapTest, JsonStampsSerializationVersion) {
  auto j = make_unrestricted_sparse_map().to_json();
  EXPECT_TRUE(j.contains("version"));
  EXPECT_NO_THROW(SBSM::from_json(j));
}

TEST(SymmetryBlockedSparseMapTest, JsonRejectsMissingVersion) {
  auto j = make_unrestricted_sparse_map().to_json();
  j.erase("version");
  EXPECT_THROW(SBSM::from_json(j), std::runtime_error);
}

TEST(SymmetryBlockedSparseMapTest, JsonRejectsMismatchedVersion) {
  auto j = make_unrestricted_sparse_map().to_json();
  j["version"] = "99.0.0";
  EXPECT_THROW(SBSM::from_json(j), std::runtime_error);
}
