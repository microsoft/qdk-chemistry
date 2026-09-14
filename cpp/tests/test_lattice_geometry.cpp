// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <qdk/chemistry/data/lattice_geometry.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace qdk::chemistry::data;

class LatticeGeometryTest : public ::testing::Test {};

namespace {
using Edge = std::pair<std::uint64_t, std::uint64_t>;

auto degree(const std::vector<Edge>& pairs, std::uint64_t site) {
  return std::count_if(pairs.begin(), pairs.end(), [site](const auto& edge) {
    return edge.first == site || edge.second == site;
  });
}

void expect_same_connections(const std::vector<NeighborConnection>& actual,
                             const std::vector<NeighborConnection>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i].site_i, expected[i].site_i);
    EXPECT_EQ(actual[i].site_j, expected[i].site_j);
    EXPECT_EQ(actual[i].bond_class.shell, expected[i].bond_class.shell);
    EXPECT_EQ(actual[i].bond_class.orientation,
              expected[i].bond_class.orientation);
    EXPECT_TRUE(
        actual[i].bond_class.axis.isApprox(expected[i].bond_class.axis));
    EXPECT_TRUE(actual[i].displacement.isApprox(expected[i].displacement));
    EXPECT_EQ(actual[i].image_shift, expected[i].image_shift);
    EXPECT_FALSE(actual[i].flavor.has_value());
    EXPECT_FALSE(expected[i].flavor.has_value());
    EXPECT_DOUBLE_EQ(actual[i].weight, 1.0);
    EXPECT_DOUBLE_EQ(expected[i].weight, 1.0);
  }
}
}  // namespace

TEST_F(LatticeGeometryTest, CartesianConstructionAndValidation) {
  Eigen::MatrixXd positions(3, 2);
  positions << 0.0, 0.0, 1.0, 0.0, 4.0, 0.0;
  const LatticeGeometry geometry(positions);
  EXPECT_EQ(geometry.num_sites(), 3);
  EXPECT_TRUE(geometry.positions().isApprox(positions));
  EXPECT_FALSE(geometry.periods().has_value());
  EXPECT_EQ(geometry.mth_nearest_neighbors(1), (std::vector<Edge>{{0, 1}}));
  EXPECT_EQ(geometry.mth_nearest_neighbors(2), (std::vector<Edge>{{1, 2}}));
  EXPECT_EQ(geometry.mth_nearest_neighbors(3), (std::vector<Edge>{{0, 2}}));

  EXPECT_THROW(LatticeGeometry(Eigen::MatrixXd::Zero(2, 3)),
               std::invalid_argument);
  for (double invalid : {std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    auto malformed = positions;
    malformed(0, 0) = invalid;
    EXPECT_THROW((LatticeGeometry(malformed)), std::invalid_argument);
    Eigen::MatrixXd periods(1, 2);
    periods << invalid, 0.0;
    EXPECT_THROW(LatticeGeometry(positions, periods), std::invalid_argument);
  }
  for (const auto& periods :
       {Eigen::MatrixXd::Zero(0, 2).eval(), Eigen::MatrixXd::Zero(1, 2).eval(),
        Eigen::MatrixXd::Ones(3, 2).eval(),
        Eigen::MatrixXd::Ones(1, 3).eval()}) {
    EXPECT_THROW(LatticeGeometry(positions, periods), std::invalid_argument);
  }
}

TEST_F(LatticeGeometryTest, FactoryDimensionsAreValidated) {
  EXPECT_THROW(LatticeGeometry::chain(0), std::invalid_argument);
  for (auto factory :
       {&LatticeGeometry::square, &LatticeGeometry::triangular,
        &LatticeGeometry::honeycomb, &LatticeGeometry::honeycomb_plaquettes,
        &LatticeGeometry::kagome}) {
    EXPECT_THROW(factory(0, 2, false, false), std::invalid_argument);
    EXPECT_THROW(factory(2, 0, false, false), std::invalid_argument);
    EXPECT_THROW(factory(1, 2, true, false), std::invalid_argument);
    EXPECT_THROW(factory(2, 1, false, true), std::invalid_argument);
    EXPECT_THROW(
        factory(std::numeric_limits<std::uint64_t>::max(), 2, false, false),
        std::overflow_error);
  }
}

TEST_F(LatticeGeometryTest, SquareGeometricNeighborShells) {
  const auto square = LatticeGeometry::square(5, 5);
  const auto shells = square.nearest_neighbor_shells({1, 2, 3});
  const auto& second = shells.at(2);
  const auto& third = shells.at(3);
  // Site 12 is the center; the distances are 1, sqrt(2), and 2.
  EXPECT_EQ(degree(shells.at(1), 12), 4);
  EXPECT_EQ(degree(second, 12), 4);
  EXPECT_EQ(degree(third, 12), 4);
  EXPECT_NE(std::find(second.begin(), second.end(), Edge{12, 18}),
            second.end());
  EXPECT_EQ(std::find(second.begin(), second.end(), Edge{12, 14}),
            second.end());
  EXPECT_NE(std::find(third.begin(), third.end(), Edge{12, 14}), third.end());
}

TEST_F(LatticeGeometryTest, HoneycombGeometricNeighborShells) {
  const auto shells =
      LatticeGeometry::honeycomb(4, 4).nearest_neighbor_shells({1, 2, 3});
  // A(1,1), site 10, has all of its first three shells inside this patch.
  EXPECT_EQ(degree(shells.at(1), 10), 3);
  EXPECT_EQ(degree(shells.at(2), 10), 6);
  EXPECT_EQ(degree(shells.at(3), 10), 3);
}

TEST_F(LatticeGeometryTest, FiniteNarrowShellsUsePresentDistances) {
  const auto distant_shell = std::numeric_limits<std::uint64_t>::max();
  const std::map<std::uint64_t, std::vector<Edge>> expected = {
      {1, {{0, 1}, {1, 2}}}, {2, {{0, 2}}}, {3, {}}, {distant_shell, {}}};
  for (const auto& geometry :
       {LatticeGeometry::chain(3), LatticeGeometry::square(1, 3),
        LatticeGeometry::triangular(1, 3)}) {
    EXPECT_EQ(geometry.nearest_neighbor_shells({3, 1, 2, 1, distant_shell}),
              expected);
  }

  // These four-site strips have no distance-2 bond: sqrt(7) is shell three.
  for (const auto& geometry :
       {LatticeGeometry::honeycomb(1, 2), LatticeGeometry::honeycomb(2, 1)}) {
    EXPECT_EQ(geometry.mth_nearest_neighbors(3), (std::vector<Edge>{{0, 3}}));
    EXPECT_NEAR(
        (geometry.positions().row(3) - geometry.positions().row(0)).norm(),
        std::sqrt(7.0), 1.0e-12);
    EXPECT_TRUE(geometry.mth_nearest_neighbors(4).empty());
  }
  EXPECT_EQ(LatticeGeometry::honeycomb(1, 1).mth_nearest_neighbors(1),
            (std::vector<Edge>{{0, 1}}));
  EXPECT_TRUE(
      LatticeGeometry::honeycomb(1, 1).neighbor_connections({2, 3}).empty());
  EXPECT_EQ(LatticeGeometry::kagome(1, 1).mth_nearest_neighbors(1).size(), 3);
  EXPECT_TRUE(LatticeGeometry::kagome(1, 1).mth_nearest_neighbors(2).empty());
}

TEST_F(LatticeGeometryTest, BondClassesAreUnlabeledAndShellLocal) {
  const std::vector<std::pair<LatticeGeometry, std::size_t>> lattices = {
      {LatticeGeometry::chain(5), 1},
      {LatticeGeometry::square(5, 5), 2},
      {LatticeGeometry::triangular(5, 5), 3},
      {LatticeGeometry::honeycomb(5, 5), 3},
      {LatticeGeometry::kagome(4, 4), 3}};
  for (const auto& [geometry, expected_orientations] : lattices) {
    std::set<std::uint32_t> orientations;
    for (const auto& connection : geometry.neighbor_connections({1})) {
      EXPECT_EQ(connection.bond_class.shell, 1);
      EXPECT_NEAR(connection.bond_class.axis.norm(), 1.0, 1.0e-12);
      EXPECT_FALSE(connection.flavor.has_value());
      EXPECT_DOUBLE_EQ(connection.weight, 1.0);
      orientations.insert(connection.bond_class.orientation);
    }
    EXPECT_EQ(orientations.size(), expected_orientations);
  }

  std::map<std::uint64_t, std::set<std::uint32_t>> orientations;
  for (const auto& connection :
       LatticeGeometry::square(5, 5).neighbor_connections({1, 2})) {
    orientations[connection.bond_class.shell].insert(
        connection.bond_class.orientation);
  }
  EXPECT_EQ(orientations.at(1).size(), 2);
  EXPECT_EQ(orientations.at(2).size(), 2);
}

TEST_F(LatticeGeometryTest, HoneycombPlaquetteShellDistances) {
  const auto hexagon = LatticeGeometry::honeycomb_plaquettes(1, 1);
  ASSERT_EQ(hexagon.num_sites(), 6);
  const auto shells = hexagon.nearest_neighbor_shells({1, 2, 3});
  EXPECT_EQ(shells.at(1).size(), 6);
  EXPECT_EQ(shells.at(2).size(), 6);
  EXPECT_EQ(shells.at(3).size(), 3);
  for (const auto& [shell, distance] :
       std::vector<std::pair<std::uint64_t, double>>{
           {1, 1.0}, {2, std::sqrt(3.0)}, {3, 2.0}}) {
    for (const auto& [i, j] : shells.at(shell)) {
      EXPECT_NEAR(
          (hexagon.positions().row(j) - hexagon.positions().row(i)).norm(),
          distance, 1.0e-12);
    }
  }
}

TEST_F(LatticeGeometryTest, PeriodicImagesRemainDistinctAndCanonical) {
  const auto connections =
      LatticeGeometry::chain(2, true).neighbor_connections({1});
  ASSERT_EQ(connections.size(), 2);
  std::set<std::array<std::int64_t, 2>> images;
  for (const auto& connection : connections) {
    EXPECT_EQ(connection.site_i, 0);
    EXPECT_EQ(connection.site_j, 1);
    EXPECT_DOUBLE_EQ(connection.displacement.x(),
                     1.0 + 2.0 * connection.image_shift[0]);
    EXPECT_DOUBLE_EQ(connection.weight, 1.0);
    EXPECT_FALSE(connection.flavor.has_value());
    images.insert(connection.image_shift);
  }
  EXPECT_EQ(images, (std::set<std::array<std::int64_t, 2>>{{-1, 0}, {0, 0}}));

  const LatticeGeometry single(Eigen::MatrixXd::Zero(1, 2),
                               Eigen::MatrixXd::Identity(2, 2));
  const auto self_images = single.neighbor_connections({1});
  ASSERT_EQ(self_images.size(), 2);
  images.clear();
  for (const auto& connection : self_images) {
    EXPECT_EQ(connection.site_i, 0);
    EXPECT_EQ(connection.site_j, 0);
    EXPECT_TRUE(connection.displacement.isApprox(Eigen::RowVector2d(
        connection.image_shift[0], connection.image_shift[1])));
    images.insert(connection.image_shift);
  }
  EXPECT_EQ(images, (std::set<std::array<std::int64_t, 2>>{{0, 1}, {1, 0}}));
}

TEST_F(LatticeGeometryTest, BuiltInConnectionsScaleToLargeLattices) {
  const auto honeycomb = LatticeGeometry::honeycomb_plaquettes(200, 200);
  const auto connections = honeycomb.neighbor_connections({1, 2, 3});
  std::array<std::size_t, 3> counts{};
  for (const auto& connection : connections) {
    ++counts.at(connection.bond_class.shell - 1);
  }
  EXPECT_EQ(honeycomb.num_sites(), 80800);
  EXPECT_EQ(counts, (std::array<std::size_t, 3>{120799, 240796, 120000}));
  EXPECT_TRUE(std::all_of(
      connections.begin(), connections.end(), [](const auto& connection) {
        return !connection.flavor && connection.weight == 1.0;
      }));

  const auto square = LatticeGeometry::square(200, 200);
  const auto shells = square.nearest_neighbor_shells({1, 2});
  EXPECT_EQ(square.num_sites(), 40000);
  EXPECT_EQ(shells.at(1).size(), 79600);
  EXPECT_EQ(shells.at(2).size(), 79202);
}

TEST_F(LatticeGeometryTest, BuiltInConnectionsMatchCartesianGeometry) {
  const auto compare = [](const LatticeGeometry& geometry) {
    const LatticeGeometry fallback(geometry.positions(), geometry.periods());
    expect_same_connections(geometry.neighbor_connections({1, 2, 3, 4, 5, 6}),
                            fallback.neighbor_connections({1, 2, 3, 4, 5, 6}));
  };
  for (bool periodic : {false, true}) {
    compare(LatticeGeometry::chain(5, periodic));
  }
  for (std::uint64_t nx = 1; nx <= 2; ++nx) {
    for (std::uint64_t ny = 1; ny <= 2; ++ny) {
      SCOPED_TRACE(::testing::Message() << "honeycomb " << nx << "x" << ny);
      compare(LatticeGeometry::honeycomb(nx, ny));
    }
  }
  for (const auto& [px, py] : std::vector<std::pair<bool, bool>>{
           {false, false}, {true, false}, {false, true}, {true, true}}) {
    SCOPED_TRACE(::testing::Message() << "periodic " << px << "," << py);
    const std::vector<std::pair<std::string, LatticeGeometry>> geometries = {
        {"square", LatticeGeometry::square(3, 4, px, py)},
        {"triangular", LatticeGeometry::triangular(3, 4, px, py)},
        {"honeycomb", LatticeGeometry::honeycomb(3, 4, px, py)},
        {"plaquettes", LatticeGeometry::honeycomb_plaquettes(4, 4, px, py)},
        {"kagome", LatticeGeometry::kagome(3, 2, px, py)}};
    for (const auto& [name, geometry] : geometries) {
      SCOPED_TRACE(name);
      compare(geometry);
    }
  }
}

TEST_F(LatticeGeometryTest, ShellValidationAndOpenBoundaryRequirement) {
  const auto open = LatticeGeometry::chain(3);
  const auto periodic = LatticeGeometry::square(4, 3, true, false);
  EXPECT_TRUE(open.neighbor_connections({}).empty());
  EXPECT_TRUE(open.nearest_neighbor_shells({}).empty());
  EXPECT_THROW(periodic.mth_nearest_neighbors(0), std::invalid_argument);
  EXPECT_THROW(periodic.nearest_neighbor_shells({1, 0}), std::invalid_argument);
  EXPECT_THROW(open.neighbor_connections({1, 0}), std::invalid_argument);
  EXPECT_THROW(periodic.mth_nearest_neighbors(1), std::runtime_error);
  EXPECT_THROW(periodic.nearest_neighbor_shells({1, 2}), std::runtime_error);
  for (double tolerance : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(open.neighbor_connections({1}, tolerance),
                 std::invalid_argument);
    EXPECT_THROW(periodic.mth_nearest_neighbors(1, tolerance),
                 std::invalid_argument);
  }
}

TEST_F(LatticeGeometryTest, GeometricShellsAreScaleInvariant) {
  for (double scale : {1.0e-200, 1.0e200}) {
    Eigen::MatrixXd positions(3, 2);
    positions << 0.0, 0.0, scale, 0.0, 2.0 * scale, 0.0;
    const LatticeGeometry open(positions);
    EXPECT_EQ(open.mth_nearest_neighbors(1),
              (std::vector<Edge>{{0, 1}, {1, 2}}));
    EXPECT_EQ(open.mth_nearest_neighbors(2), (std::vector<Edge>{{0, 2}}));
    Eigen::MatrixXd periods(1, 2);
    periods << 3.0 * scale, 0.0;
    const LatticeGeometry periodic(positions, periods);
    const auto connections = periodic.neighbor_connections({1});
    ASSERT_EQ(connections.size(), 3);
    for (const auto& connection : connections) {
      EXPECT_NEAR(
          std::hypot(connection.displacement.x(), connection.displacement.y()) /
              scale,
          1.0, 1.0e-12);
    }
    const LatticeGeometry two_periods(Eigen::MatrixXd::Zero(1, 2),
                                      scale * Eigen::MatrixXd::Identity(2, 2));
    EXPECT_EQ(two_periods.neighbor_connections({1}).size(), 2);
  }
  for (const auto& periods :
       std::vector<nlohmann::json>{{{1.0, 0.0}, {2.0, 0.0}},
                                   {{1.0, 1.0}, {-2.0, -2.0}},
                                   {{1.0e-200, 0.0}, {0.0, 1.0e200}}}) {
    EXPECT_THROW(LatticeGeometry::from_json(
                     {{"positions", {{0.0, 0.0}}}, {"periods", periods}}),
                 std::invalid_argument);
  }
}

TEST_F(LatticeGeometryTest, ConnectionsScaleBeforeSubtractingLargeCoordinates) {
  Eigen::MatrixXd positions(2, 2);
  positions << -9.0e307, 0.0, 9.0e307, 0.0;
  Eigen::MatrixXd periods(1, 2);
  periods << 1.5e308, 0.0;
  const auto connections =
      LatticeGeometry(positions, periods).neighbor_connections({1});
  ASSERT_EQ(connections.size(), 1);
  EXPECT_NEAR(connections[0].displacement.x() / 3.0e307, 1.0, 1.0e-12);
  EXPECT_EQ(connections[0].image_shift[0], -1);
}

TEST_F(LatticeGeometryTest, DistantFirstSiteDoesNotCollapseNearbyPositions) {
  Eigen::MatrixXd positions(3, 2);
  positions << std::ldexp(1.0, 60), 0.0, 0.0, 0.0, 1.0, 0.0;
  const LatticeGeometry geometry(positions);
  EXPECT_EQ(geometry.mth_nearest_neighbors(1), (std::vector<Edge>{{1, 2}}));
  const auto connections = geometry.neighbor_connections({1});
  ASSERT_EQ(connections.size(), 1);
  EXPECT_TRUE(
      connections[0].displacement.isApprox(Eigen::RowVector2d(1.0, 0.0)));
}

TEST_F(LatticeGeometryTest, TranslationPreservesRepresentableLocalDistances) {
  for (double origin : {-1.0e16, 1.0e16}) {
    Eigen::MatrixXd positions = Eigen::MatrixXd::Zero(4, 2);
    for (Eigen::Index site = 0; site < positions.rows(); ++site) {
      positions(site, 0) = origin + 2.0 * site;
    }
    const LatticeGeometry geometry(positions);
    EXPECT_EQ(geometry.mth_nearest_neighbors(1),
              (std::vector<Edge>{{0, 1}, {1, 2}, {2, 3}}));
    EXPECT_EQ(geometry.mth_nearest_neighbors(2),
              (std::vector<Edge>{{0, 2}, {1, 3}}));
    for (const auto& connection : geometry.neighbor_connections({1})) {
      EXPECT_DOUBLE_EQ(connection.displacement.x(), 2.0);
    }
    Eigen::MatrixXd periods(1, 2);
    periods << 8.0, 0.0;
    const auto periodic =
        LatticeGeometry(positions, periods).neighbor_connections({1});
    ASSERT_EQ(periodic.size(), 4);
    for (const auto& connection : periodic) {
      EXPECT_DOUBLE_EQ(std::abs(connection.displacement.x()), 2.0);
    }
  }
}

TEST_F(LatticeGeometryTest, SerializationRetainsCoordinatesHashAndConnections) {
  for (const auto& [px, py] : std::vector<std::pair<bool, bool>>{
           {false, false}, {true, false}, {false, true}, {true, true}}) {
    const auto original = LatticeGeometry::honeycomb_plaquettes(2, 2, px, py);
    std::vector<std::uint64_t> path(original.num_sites());
    std::iota(path.begin(), path.end(), 0);
    std::rotate(path.begin(), path.begin() + 1, path.end());
    const auto geometry = LatticeGeometry::permute(original, path);
    ASSERT_EQ(geometry.periods().has_value(), original.periods().has_value());
    if (original.periods()) {
      EXPECT_TRUE(geometry.periods()->isApprox(*original.periods()));
    }
    for (const std::string type : {"json", "hdf5"}) {
      const std::string filename = "test_roundtrip.lattice_geometry." +
                                   std::string(type == "json" ? "json" : "h5");
      geometry.to_file(filename, type);
      const auto restored = LatticeGeometry::from_file(filename, type);
      std::filesystem::remove(filename);
      EXPECT_TRUE(restored.positions().isApprox(geometry.positions()));
      ASSERT_EQ(restored.periods().has_value(), geometry.periods().has_value());
      if (geometry.periods()) {
        EXPECT_TRUE(restored.periods()->isApprox(*geometry.periods()));
      }
      EXPECT_EQ(restored.content_hash(), geometry.content_hash());
      expect_same_connections(restored.neighbor_connections({1, 2, 3}),
                              geometry.neighbor_connections({1, 2, 3}));
    }
  }
}

TEST_F(LatticeGeometryTest, EmptyAndCoincidentGeometryRoundTrips) {
  for (Eigen::Index count : {0, 2}) {
    const LatticeGeometry geometry(Eigen::MatrixXd::Zero(count, 2));
    EXPECT_TRUE(geometry.neighbor_connections({1, 2}).empty());
    const auto shells = geometry.nearest_neighbor_shells({1, 2});
    ASSERT_EQ(shells.size(), 2);
    EXPECT_TRUE(shells.at(1).empty());
    EXPECT_TRUE(shells.at(2).empty());
    const std::string filename = "test_empty.lattice_geometry.h5";
    geometry.to_hdf5_file(filename);
    const auto hdf5 = LatticeGeometry::from_hdf5_file(filename);
    std::filesystem::remove(filename);
    for (const auto& restored :
         {LatticeGeometry::from_json(geometry.to_json()), hdf5}) {
      EXPECT_EQ(restored.positions().rows(), count);
      EXPECT_EQ(restored.positions().cols(), 2);
      EXPECT_EQ(restored.content_hash(), geometry.content_hash());
      EXPECT_TRUE(restored.neighbor_connections({1}).empty());
    }
  }
}

TEST_F(LatticeGeometryTest, PermutationRemapsPositionsAndShells) {
  const auto geometry = LatticeGeometry::square(3, 2);
  const std::vector<std::uint64_t> path = {1, 2, 0, 4, 5, 3};
  const auto permuted = LatticeGeometry::permute(geometry, path);
  std::vector<std::uint64_t> inverse(path.size());
  for (std::uint64_t i = 0; i < path.size(); ++i) {
    inverse[path[i]] = i;
    EXPECT_TRUE(permuted.positions().row(i).isApprox(
        geometry.positions().row(path[i])));
  }
  for (std::uint64_t shell : {1, 2, 3}) {
    std::vector<Edge> expected;
    for (const auto& [i, j] : geometry.mth_nearest_neighbors(shell)) {
      const auto pair = std::minmax(inverse[i], inverse[j]);
      expected.emplace_back(pair.first, pair.second);
    }
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(permuted.mth_nearest_neighbors(shell), expected);
  }
  EXPECT_EQ(LatticeGeometry::permute(permuted, inverse).content_hash(),
            geometry.content_hash());
  EXPECT_THROW(LatticeGeometry::permute(geometry, {0, 1}),
               std::invalid_argument);
  EXPECT_THROW(LatticeGeometry::permute(geometry, {0, 1, 2, 3, 4, 4}),
               std::invalid_argument);
  EXPECT_THROW(LatticeGeometry::permute(geometry, {0, 1, 2, 3, 4, 6}),
               std::invalid_argument);
}

TEST_F(LatticeGeometryTest, JsonRejectsInvalidGeometry) {
  const auto valid = LatticeGeometry::square(2, 2, true, false).to_json();
  const std::vector<std::pair<std::string, nlohmann::json>> invalid = {
      {"/positions", 0.0},
      {"/positions", {0.0, 0.0}},
      {"/positions", {{0.0, 0.0, 0.0}}},
      {"/positions", {{0.0, 0.0}, {1.0}}},
      {"/positions/0/0", std::numeric_limits<double>::infinity()},
      {"/periods", nlohmann::json::array()},
      {"/periods", {{0.0, 0.0}}},
      {"/periods", {{1.0, 0.0}, {2.0, 0.0}}},
      {"/periods/0/0", std::numeric_limits<double>::quiet_NaN()}};
  for (const auto& [path, value] : invalid) {
    SCOPED_TRACE(path);
    auto malformed = valid;
    malformed[nlohmann::json::json_pointer(path)] = value;
    EXPECT_THROW(LatticeGeometry::from_json(malformed), std::invalid_argument);
  }
  for (const std::string field : {"positions", "periods"}) {
    SCOPED_TRACE(field);
    for (const auto& scalar :
         std::vector<nlohmann::json>{nullptr, true, "text"}) {
      auto malformed = valid;
      malformed[field][0][0] = scalar;
      EXPECT_THROW(LatticeGeometry::from_json(malformed),
                   nlohmann::json::type_error);
    }
  }
}

TEST_F(LatticeGeometryTest, Hdf5RejectsInvalidGeometry) {
  const auto geometry = LatticeGeometry::square(2, 2, true, false);
  const std::string filename = "test_invalid.lattice_geometry.h5";
  const auto expect_invalid = [&](const auto& mutate) {
    geometry.to_hdf5_file(filename);
    {
      H5::H5File file(filename, H5F_ACC_RDWR);
      auto root = file.openGroup("/");
      mutate(root);
    }
    EXPECT_THROW(LatticeGeometry::from_hdf5_file(filename),
                 std::invalid_argument);
    std::filesystem::remove(filename);
  };
  const hsize_t invalid_dimensions[3] = {2, 3, 2};
  const hsize_t matrix_dimensions[2] = {1, 2};
  for (const std::string name : {"positions", "periods"}) {
    SCOPED_TRACE(name);
    for (const auto& space :
         {H5::DataSpace(H5S_SCALAR), H5::DataSpace(1, invalid_dimensions),
          H5::DataSpace(2, invalid_dimensions),
          H5::DataSpace(3, invalid_dimensions)}) {
      expect_invalid([&](H5::Group& root) {
        root.unlink(name);
        root.createDataSet(name, H5::PredType::NATIVE_DOUBLE, space);
      });
    }
    expect_invalid([&](H5::Group& root) {
      root.unlink(name);
      root.createDataSet(name, H5::StrType(H5::PredType::C_S1, 8),
                         H5::DataSpace(2, matrix_dimensions));
    });
    expect_invalid([&](H5::Group& root) {
      Eigen::MatrixXd values =
          name == "positions" ? geometry.positions() : *geometry.periods();
      values(0, 0) = std::numeric_limits<double>::quiet_NaN();
      root.openDataSet(name).write(values.data(), H5::PredType::NATIVE_DOUBLE);
    });
  }
  expect_invalid([](H5::Group& root) {
    const double values[2] = {0.0, 0.0};
    root.openDataSet("periods").write(values, H5::PredType::NATIVE_DOUBLE);
  });
}
