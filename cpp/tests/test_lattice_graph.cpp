// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class LatticeGraphTest : public ::testing::Test {};

namespace {
using Edge = std::pair<std::uint64_t, std::uint64_t>;
constexpr BondFlavorId flavor_x = 10;
constexpr BondFlavorId flavor_y = 20;
constexpr BondFlavorId flavor_z = 30;

std::vector<BondFlavorDefinition> honeycomb_flavor_ids() {
  const double root_three = std::sqrt(3.0);
  return {
      {1, {0.5, root_three / 2.0}, flavor_x},
      {1, {0.5, -root_three / 2.0}, flavor_y},
      {1, {1.0, 0.0}, flavor_z},
      {2, {1.5, -root_three / 2.0}, flavor_x},
      {2, {1.5, root_three / 2.0}, flavor_y},
      {2, {0.0, root_three}, flavor_z},
      {3, {1.0, root_three}, flavor_x},
      {3, {1.0, -root_three}, flavor_y},
      {3, {2.0, 0.0}, flavor_z},
  };
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
    EXPECT_EQ(actual[i].flavor, expected[i].flavor);
    EXPECT_DOUBLE_EQ(actual[i].weight, expected[i].weight);
  }
}
}  // namespace

TEST_F(LatticeGraphTest, ChainConstructor) {
  // 4-site chain
  //
  //   0 --- 1 --- 2 --- 3

  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  std::map<Edge, double> expected_edges = {
      {{0, 1}, 1.0},
      {{1, 2}, 1.0},
      {{2, 3}, 1.0},
  };
  auto expected =
      LatticeGraph::make_bidirectional(LatticeGraph(expected_edges, 4));

  auto chain = LatticeGraph::chain(4);
  EXPECT_EQ(chain.num_sites(), 4);
  EXPECT_EQ(chain.num_edges(), 3);
  EXPECT_TRUE(chain.is_symmetric());
  EXPECT_TRUE(chain.adjacency_matrix().isApprox(expected.adjacency_matrix()));

  // Periodic (ring): wrap edge
  {
    std::map<Edge, double> ring_edges = expected_edges;
    ring_edges[{0, 3}] = 1.0;  // wrap
    auto expected_ring =
        LatticeGraph::make_bidirectional(LatticeGraph(ring_edges, 4));

    auto ring = LatticeGraph::chain(4, true);
    EXPECT_EQ(ring.num_sites(), 4);
    EXPECT_EQ(ring.num_edges(), 4);  // 3 + 1
    EXPECT_TRUE(ring.is_symmetric());
    EXPECT_TRUE(
        ring.adjacency_matrix().isApprox(expected_ring.adjacency_matrix()));
  }
}

TEST_F(LatticeGraphTest, SquareConstructor) {
  // 3x4 square lattice (12 sites)
  //
  //   9 -- 10 -- 11
  //   |     |     |
  //   6 --- 7 --- 8
  //   |     |     |
  //   3 --- 4 --- 5
  //   |     |     |
  //   0 --- 1 --- 2

  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  std::map<Edge, double> expected_edges = {
      // Right
      {{0, 1}, 1.0},
      {{1, 2}, 1.0},
      {{3, 4}, 1.0},
      {{4, 5}, 1.0},
      {{6, 7}, 1.0},
      {{7, 8}, 1.0},
      {{9, 10}, 1.0},
      {{10, 11}, 1.0},
      // Up
      {{0, 3}, 1.0},
      {{1, 4}, 1.0},
      {{2, 5}, 1.0},
      {{3, 6}, 1.0},
      {{4, 7}, 1.0},
      {{5, 8}, 1.0},
      {{6, 9}, 1.0},
      {{7, 10}, 1.0},
      {{8, 11}, 1.0},
  };
  auto expected =
      LatticeGraph::make_bidirectional(LatticeGraph(expected_edges, 12));

  auto sq = LatticeGraph::square(3, 4);
  EXPECT_EQ(sq.num_sites(), 12);
  EXPECT_EQ(sq.num_edges(), 17);
  EXPECT_TRUE(sq.is_symmetric());
  EXPECT_TRUE(sq.adjacency_matrix().isApprox(expected.adjacency_matrix()));

  // periodic_y only: up wraps (no right wraps)
  {
    std::map<Edge, double> py_edges = expected_edges;
    py_edges[{0, 9}] = 1.0;   // up wrap
    py_edges[{1, 10}] = 1.0;  // up wrap
    py_edges[{2, 11}] = 1.0;  // up wrap
    auto expected_py =
        LatticeGraph::make_bidirectional(LatticeGraph(py_edges, 12));

    auto sq_py = LatticeGraph::square(3, 4, false, true);
    EXPECT_EQ(sq_py.num_sites(), 12);
    EXPECT_EQ(sq_py.num_edges(), 20);  // 17 + 3
    EXPECT_TRUE(sq_py.is_symmetric());
    EXPECT_TRUE(
        sq_py.adjacency_matrix().isApprox(expected_py.adjacency_matrix()));
  }

  // periodic_x only: right wraps (no up wraps)
  {
    std::map<Edge, double> px_edges = expected_edges;
    px_edges[{0, 2}] = 1.0;   // right wrap
    px_edges[{3, 5}] = 1.0;   // right wrap
    px_edges[{6, 8}] = 1.0;   // right wrap
    px_edges[{9, 11}] = 1.0;  // right wrap
    auto expected_px =
        LatticeGraph::make_bidirectional(LatticeGraph(px_edges, 12));

    auto sq_px = LatticeGraph::square(3, 4, true, false);
    EXPECT_EQ(sq_px.num_sites(), 12);
    EXPECT_EQ(sq_px.num_edges(), 21);  // 17 + 4
    EXPECT_TRUE(sq_px.is_symmetric());
    EXPECT_TRUE(
        sq_px.adjacency_matrix().isApprox(expected_px.adjacency_matrix()));
  }

  // Both periodic: right wraps + up wraps
  {
    std::map<Edge, double> pxy_edges = expected_edges;
    pxy_edges[{0, 2}] = 1.0;   // right wrap
    pxy_edges[{3, 5}] = 1.0;   // right wrap
    pxy_edges[{6, 8}] = 1.0;   // right wrap
    pxy_edges[{9, 11}] = 1.0;  // right wrap
    pxy_edges[{0, 9}] = 1.0;   // up wrap
    pxy_edges[{1, 10}] = 1.0;  // up wrap
    pxy_edges[{2, 11}] = 1.0;  // up wrap
    auto expected_pxy =
        LatticeGraph::make_bidirectional(LatticeGraph(pxy_edges, 12));

    auto sq_pxy = LatticeGraph::square(3, 4, true, true);
    EXPECT_EQ(sq_pxy.num_sites(), 12);
    EXPECT_EQ(sq_pxy.num_edges(), 24);  // 17 + 4 + 3
    EXPECT_TRUE(sq_pxy.is_symmetric());
    EXPECT_TRUE(
        sq_pxy.adjacency_matrix().isApprox(expected_pxy.adjacency_matrix()));
  }
}

TEST_F(LatticeGraphTest, HoneycombFlavorsResolveSelectedShells) {
  const auto honeycomb =
      LatticeGraph::from_geometry(LatticeGeometry::honeycomb(5, 5), {1, 2, 3},
                                  honeycomb_flavor_ids(), 2.5, 1.0e-9);
  EXPECT_EQ(honeycomb.selected_shells(), (std::vector<std::uint64_t>{1, 2, 3}));
  constexpr std::uint64_t center = 24;
  std::map<std::uint64_t, std::map<BondFlavorId, std::size_t>> flavor_degree;
  for (const auto& connection : honeycomb.connections()) {
    ASSERT_TRUE(connection.flavor.has_value());
    EXPECT_DOUBLE_EQ(connection.weight, 2.5);
    if (connection.site_i == center || connection.site_j == center) {
      ++flavor_degree[connection.bond_class.shell][*connection.flavor];
    }
  }
  for (BondFlavorId flavor : {flavor_x, flavor_y, flavor_z}) {
    EXPECT_EQ(flavor_degree.at(1).at(flavor), 1);
    EXPECT_EQ(flavor_degree.at(2).at(flavor), 2);
    EXPECT_EQ(flavor_degree.at(3).at(flavor), 1);
  }
}

TEST_F(LatticeGraphTest, FactoriesSelectOnlyNearestShellWithoutFlavors) {
  for (const auto& graph :
       {LatticeGraph::chain(5), LatticeGraph::square(3, 3),
        LatticeGraph::triangular(3, 3), LatticeGraph::honeycomb(3, 3),
        LatticeGraph::honeycomb_plaquettes(2, 2), LatticeGraph::kagome(3, 3)}) {
    ASSERT_TRUE(graph.geometry());
    EXPECT_EQ(graph.geometry()->num_sites(), graph.num_sites());
    EXPECT_EQ(graph.selected_shells(), (std::vector<std::uint64_t>{1}));
    ASSERT_FALSE(graph.connections().empty());
    for (const auto& connection : graph.connections()) {
      EXPECT_EQ(connection.bond_class.shell, 1);
      EXPECT_FALSE(connection.flavor.has_value());
    }
  }
}

TEST_F(LatticeGraphTest, HoneycombOpenPlaquettePatches) {
  auto unit_cell = LatticeGraph::honeycomb(1, 1);
  EXPECT_EQ(unit_cell.num_sites(), 2);
  EXPECT_EQ(unit_cell.num_edges(), 1);

  auto hexagon = LatticeGraph::honeycomb_plaquettes(1, 1, false, false, 2.5);
  EXPECT_EQ(hexagon.num_sites(), 6);
  EXPECT_EQ(hexagon.num_edges(), 6);
  EXPECT_TRUE(hexagon.is_symmetric());
  ASSERT_TRUE(hexagon.geometry());
  const auto& positions = hexagon.geometry()->positions();
  EXPECT_EQ(positions.rows(), 6);
  EXPECT_EQ(positions.cols(), 2);
  for (Eigen::Index site = 0;
       site < hexagon.sparse_adjacency_matrix().outerSize(); ++site) {
    EXPECT_EQ(hexagon.sparse_adjacency_matrix().innerVector(site).nonZeros(),
              2);
  }

  const auto flavored_hexagon = LatticeGraph::from_geometry(
      *hexagon.geometry(), {1, 2, 3}, honeycomb_flavor_ids(), 2.5, 1.0e-9);
  std::map<std::uint64_t, std::map<BondFlavorId, std::size_t>> counts;
  for (const auto& connection : flavored_hexagon.connections()) {
    ASSERT_TRUE(connection.flavor.has_value());
    EXPECT_DOUBLE_EQ(connection.weight, 2.5);
    ++counts[connection.bond_class.shell][*connection.flavor];
  }
  for (BondFlavorId flavor : {flavor_x, flavor_y, flavor_z}) {
    EXPECT_EQ(counts.at(1).at(flavor), 2);
    EXPECT_EQ(counts.at(2).at(flavor), 2);
    EXPECT_EQ(counts.at(3).at(flavor), 1);
  }
  auto patch = LatticeGraph::honeycomb_plaquettes(4, 4);
  EXPECT_EQ(patch.num_sites(), 48);
  EXPECT_EQ(patch.num_edges(), 63);
  EXPECT_EQ(patch.num_edges() - patch.num_sites() + 1, 16);
  for (Eigen::Index site = 0;
       site < patch.sparse_adjacency_matrix().outerSize(); ++site) {
    EXPECT_GE(patch.sparse_adjacency_matrix().innerVector(site).nonZeros(), 2);
  }

  EXPECT_EQ(
      LatticeGraph::honeycomb(2, 2, true, true).content_hash(),
      LatticeGraph::honeycomb_plaquettes(2, 2, true, true).content_hash());
}

TEST_F(LatticeGraphTest, PeriodicConnectionsPreserveFlavorMultiplicity) {
  const auto honeycomb = LatticeGraph::from_geometry(
      LatticeGeometry::honeycomb(2, 2, true, true), {1, 2, 3},
      honeycomb_flavor_ids(), 1.0, 1.0e-9);
  std::map<Edge, std::set<BondFlavorId>> flavors_by_pair;
  std::size_t count = 0;
  for (const auto& connection : honeycomb.connections()) {
    if (connection.bond_class.shell != 3) continue;
    ++count;
    ASSERT_TRUE(connection.flavor.has_value());
    flavors_by_pair[{connection.site_i, connection.site_j}].insert(
        *connection.flavor);
  }
  EXPECT_TRUE(
      std::any_of(flavors_by_pair.begin(), flavors_by_pair.end(),
                  [](const auto& item) { return item.second.size() > 1; }));
  EXPECT_GT(count, flavors_by_pair.size());
}

TEST_F(LatticeGraphTest, ResolvedRecordsSurviveDataOperations) {
  const auto geometry = std::make_shared<LatticeGeometry>(
      LatticeGeometry::square(2, 2, true, true));
  // Labels deliberately differ from geometric shell/orientation indices.
  const auto graph = LatticeGraph::from_connections(
      4,
      {{1, 0, {7, 41, {1.0, 0.0}}, {-1.0, 0.0}, {0, 0}, flavor_x, 2.5},
       {0, 1, {9, 99, {1.0, 0.0}}, {-1.0, 0.0}, {-1, 0}, flavor_y, -0.75},
       {2, 2, {7, 3, {0.0, 1.0}}, {0.0, -2.0}, {0, -1}, std::nullopt, 4.0}},
      geometry, {11, 7, 11});
  const std::vector<NeighborConnection> expected = {
      {2, 2, {7, 3, {0.0, 1.0}}, {0.0, 2.0}, {0, 1}, std::nullopt, 4.0},
      {0, 1, {7, 41, {1.0, 0.0}}, {1.0, 0.0}, {0, 0}, flavor_x, 2.5},
      {0, 1, {9, 99, {1.0, 0.0}}, {-1.0, 0.0}, {-1, 0}, flavor_y, -0.75}};
  expect_same_connections(graph.connections(), expected);
  EXPECT_EQ(graph.selected_shells(), (std::vector<std::uint64_t>{7, 9, 11}));
  EXPECT_TRUE(graph.is_symmetric());
  EXPECT_DOUBLE_EQ(graph.weight(0, 1), 1.75);
  EXPECT_DOUBLE_EQ(graph.weight(1, 0), 1.75);
  EXPECT_DOUBLE_EQ(graph.weight(2, 2), 4.0);
  EXPECT_TRUE(graph.adjacency_matrix().row(3).isZero());

  const auto json = graph.to_json();
  EXPECT_EQ(json.at("geometry"), geometry->to_json());
  EXPECT_FALSE(json.contains("positions"));
  const std::string filename = "test_records.lattice_graph.h5";
  graph.to_hdf5_file(filename);
  const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
  std::filesystem::remove(filename);
  for (const auto& restored :
       {LatticeGraph::from_json(nlohmann::json::parse(json.dump())), hdf5}) {
    ASSERT_TRUE(restored.geometry());
    expect_same_connections(restored.connections(), expected);
    EXPECT_EQ(restored.to_json(), json);
    EXPECT_EQ(restored.content_hash(), graph.content_hash());
  }

  const std::vector<std::uint64_t> path = {1, 2, 0, 3};
  const auto permuted = LatticeGraph::permute(graph, path);
  const std::vector<NeighborConnection> expected_permuted = {
      {1, 1, {7, 3, {0.0, 1.0}}, {0.0, 2.0}, {0, 1}, std::nullopt, 4.0},
      {0, 2, {7, 41, {1.0, 0.0}}, {-1.0, 0.0}, {0, 0}, flavor_x, 2.5},
      {0, 2, {9, 99, {1.0, 0.0}}, {1.0, 0.0}, {1, 0}, flavor_y, -0.75}};
  expect_same_connections(permuted.connections(), expected_permuted);
  EXPECT_EQ(permuted.selected_shells(), graph.selected_shells());
  ASSERT_TRUE(permuted.geometry());
  for (std::uint64_t i = 0; i < path.size(); ++i) {
    EXPECT_TRUE(permuted.geometry()->positions().row(i).isApprox(
        geometry->positions().row(path[i])));
    for (std::uint64_t j = 0; j < path.size(); ++j) {
      EXPECT_DOUBLE_EQ(permuted.weight(i, j), graph.weight(path[i], path[j]));
    }
  }
  ASSERT_TRUE(permuted.geometry()->periods().has_value());
  EXPECT_TRUE(permuted.geometry()->periods()->isApprox(*geometry->periods()));
  const auto doubled = LatticeGraph::make_bidirectional(graph);
  auto expected_doubled = expected;
  for (auto& connection : expected_doubled) connection.weight *= 2.0;
  expect_same_connections(doubled.connections(), expected_doubled);
  EXPECT_TRUE(
      doubled.adjacency_matrix().isApprox(2.0 * graph.adjacency_matrix()));
  EXPECT_EQ(doubled.selected_shells(), graph.selected_shells());
  expect_same_connections(graph.connections(), expected);
}

TEST_F(LatticeGraphTest, BondFlavorAxesAreScaleInvariant) {
  const auto square = LatticeGraph::square(2, 2);
  const auto expected = square.with_bond_flavors({{1, {1.0, 0.0}, 1000}});
  for (double scale : {1.0e-200, 1.0e200, -1.0e-200, -1.0e200}) {
    const auto flavored = square.with_bond_flavors({{1, {scale, 0.0}, 1000}});
    expect_same_connections(flavored.connections(), expected.connections());
    EXPECT_EQ(
        std::count_if(
            flavored.connections().begin(), flavored.connections().end(),
            [](const auto& connection) { return connection.flavor == 1000; }),
        2);
  }
}

TEST_F(LatticeGraphTest, CancellingWeightsSurvivePermutationAndSerialization) {
  const auto geometry =
      std::make_shared<LatticeGeometry>(LatticeGeometry::chain(2, true));
  const double maximum = std::numeric_limits<double>::max();
  const auto graph = LatticeGraph::from_connections(
      2,
      {{0, 1, {1, 0, {1.0, 0.0}}, {1.0, 0.0}, {0, 0}, flavor_x, maximum},
       {0, 1, {3, 0, {1.0, 0.0}}, {-3.0, 0.0}, {-2, 0}, flavor_x, -maximum},
       {0, 1, {3, 0, {1.0, 0.0}}, {3.0, 0.0}, {1, 0}, flavor_x, maximum}},
      geometry);
  const auto permuted = LatticeGraph::permute(graph, {1, 0});
  EXPECT_DOUBLE_EQ(permuted.weight(0, 1), maximum);
  EXPECT_EQ(LatticeGraph::from_json(permuted.to_json()).content_hash(),
            permuted.content_hash());
  const std::string filename = "test_cancelling.lattice_graph.h5";
  permuted.to_hdf5_file(filename);
  EXPECT_EQ(LatticeGraph::from_hdf5_file(filename).content_hash(),
            permuted.content_hash());
  std::filesystem::remove(filename);

  const auto cancelling = LatticeGraph::from_connections(
      2, {{0, 1, {1, 0, {1.0, 0.0}}, {1.0, 0.0}, {0, 0}, flavor_x, 1.0e16},
          {0, 1, {1, 0, {1.0, 0.0}}, {-1.0, 0.0}, {-1, 0}, flavor_y, -1.0e16}});
  auto malformed = cancelling.to_json();
  malformed["adjacency_sparse"] = {{0, 1, 1.0}};
  EXPECT_THROW(LatticeGraph::from_json(malformed), std::invalid_argument);
}

TEST_F(LatticeGraphTest, ConnectionPersistencePreservesIntegerTypes) {
  constexpr auto shell = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t empty_shell = (std::uint64_t{1} << 53) + 1;
  constexpr std::int64_t image = (std::int64_t{1} << 53) + 1;
  constexpr auto orientation = std::numeric_limits<std::uint32_t>::max();
  constexpr BondFlavorId flavor = std::numeric_limits<BondFlavorId>::max();
  const auto graph = LatticeGraph::from_connections(
      3,
      {{0,
        1,
        {shell, orientation, {1.0, 0.0}},
        {1.0, 0.0},
        {image, -image},
        flavor,
        2.5},
       {0, 1, {2, 0, {1.0, 0.0}}, {-1.0, 0.0}, {0, 0}, BondFlavorId{0}, -2.5},
       {2, 2, {2, 99, {0.0, 1.0}}, {0.0, 1.0}, {0, 1}, std::nullopt, 0.125}},
      nullptr, {empty_shell});
  const auto json = graph.to_json();
  const auto& record = json.at("connections").back();
  EXPECT_TRUE(record.is_object());
  EXPECT_TRUE(record.at("bond_class").at("shell").is_number_unsigned());
  EXPECT_EQ(record.at("bond_class").at("shell").get<std::uint64_t>(), shell);
  EXPECT_EQ(record.at("bond_class").at("orientation").get<std::uint32_t>(),
            orientation);
  EXPECT_EQ(record.at("flavor").get<BondFlavorId>(), flavor);

  const std::string filename = "test_integer_types.lattice_graph.h5";
  graph.to_hdf5_file(filename);
  {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    auto records = file.openGroup("/connections");
    for (const std::string column :
         {"site_i", "site_j", "shells", "orientations", "flavors"}) {
      const auto dataset = records.openDataSet(column);
      EXPECT_EQ(dataset.getTypeClass(), H5T_INTEGER);
      EXPECT_EQ(dataset.getIntType().getSign(), H5T_SGN_NONE);
    }
    EXPECT_EQ(records.openDataSet("shells").getIntType().getSize(),
              sizeof(std::uint64_t));
    EXPECT_EQ(records.openDataSet("image_shifts").getIntType().getSign(),
              H5T_SGN_2);
  }
  const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
  std::filesystem::remove(filename);
  for (const auto& restored :
       {LatticeGraph::from_json(nlohmann::json::parse(json.dump())), hdf5}) {
    expect_same_connections(restored.connections(), graph.connections());
    EXPECT_EQ(restored.selected_shells(),
              (std::vector<std::uint64_t>{2, empty_shell, shell}));
    EXPECT_EQ(restored.content_hash(), graph.content_hash());
    EXPECT_FALSE(restored.are_connected(0, 1));
    EXPECT_DOUBLE_EQ(restored.weight(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(restored.weight(2, 2), 0.125);
  }
}

TEST_F(LatticeGraphTest, JsonRejectsMalformedConnectionMetadata) {
  const auto valid = LatticeGraph::chain(2).to_json();
  const auto out_of_range =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
  const std::vector<std::pair<std::string, nlohmann::json>> invalid = {
      {"/connections/0/site_i", -1},
      {"/connections/0/bond_class/shell", 1.5},
      {"/connections/0/bond_class/shell", -1},
      {"/connections/0/bond_class/orientation", out_of_range},
      {"/connections/0/bond_class/orientation", -1},
      {"/connections/0/flavor", 1.5},
      {"/connections/0/flavor", -1},
      {"/connections/0/flavor", out_of_range},
      {"/connections/0/image_shift", {0.5, 0}},
      {"/connections/0/bond_class/axis", {1.0}},
      {"/selected_shells", {0}},
      {"/adjacency_sparse", {{0, 1, 99.0}}}};
  for (const auto& [path, value] : invalid) {
    SCOPED_TRACE(path);
    auto malformed = valid;
    malformed[nlohmann::json::json_pointer(path)] = value;
    EXPECT_THROW(LatticeGraph::from_json(malformed), std::invalid_argument);
  }
  auto records_only = valid;
  records_only.erase("adjacency_sparse");
  EXPECT_EQ(LatticeGraph::from_json(records_only).content_hash(),
            LatticeGraph::from_json(valid).content_hash());
}

TEST_F(LatticeGraphTest, Hdf5RejectsMalformedConnectionMetadata) {
  const auto graph = LatticeGraph::chain(2);
  const std::string filename = "test_invalid_connections.lattice_graph.h5";
  const auto expect_invalid = [&](const auto& mutate) {
    graph.to_hdf5_file(filename);
    {
      H5::H5File file(filename, H5F_ACC_RDWR);
      auto records = file.openGroup("/connections");
      mutate(records);
    }
    EXPECT_THROW(LatticeGraph::from_hdf5_file(filename), std::invalid_argument);
    std::filesystem::remove(filename);
  };

  expect_invalid([](H5::Group& records) {
    records.unlink("axes");
    const hsize_t dims[1] = {2};
    auto dataset = records.createDataSet("axes", H5::PredType::NATIVE_DOUBLE,
                                         H5::DataSpace(1, dims));
    const double axes[2] = {1.0, 0.0};
    dataset.write(axes, H5::PredType::NATIVE_DOUBLE);
  });
  expect_invalid([](H5::Group& records) {
    records.unlink("shells");
    const hsize_t dims[1] = {1};
    auto dataset = records.createDataSet("shells", H5::PredType::NATIVE_DOUBLE,
                                         H5::DataSpace(1, dims));
    const double shell = 1.0;
    dataset.write(&shell, H5::PredType::NATIVE_DOUBLE);
  });
  for (const std::string column : {"flavors", "orientations"}) {
    expect_invalid([&](H5::Group& records) {
      records.unlink(column);
      const hsize_t dims[1] = {1};
      auto dataset = records.createDataSet(column, H5::PredType::NATIVE_UINT64,
                                           H5::DataSpace(1, dims));
      const std::uint64_t value = std::uint64_t{1} << 32;
      dataset.write(&value, H5::PredType::NATIVE_UINT64);
    });
  }
  expect_invalid([](H5::Group& records) {
    records.unlink("image_shifts");
    const hsize_t dims[2] = {1, 2};
    records.createDataSet("image_shifts", H5::PredType::NATIVE_UINT64,
                          H5::DataSpace(2, dims));
  });
}

TEST_F(LatticeGraphTest, FromGeometrySelectsShellsAndWeights) {
  const auto geometry = LatticeGeometry::chain(4);
  const auto graph = LatticeGraph::from_geometry(
      geometry, {3, 1, 5, 3},
      {{1, {1.0, 0.0}, flavor_x}, {2, {1.0, 0.0}, flavor_y}}, -2.5, 1.0e-9);
  EXPECT_EQ(graph.selected_shells(), (std::vector<std::uint64_t>{1, 3, 5}));
  ASSERT_EQ(graph.connections().size(), 4);
  for (const auto& connection : graph.connections()) {
    EXPECT_DOUBLE_EQ(connection.weight, -2.5);
    EXPECT_EQ(connection.flavor, connection.bond_class.shell == 1
                                     ? std::optional<BondFlavorId>(flavor_x)
                                     : std::nullopt);
  }
  EXPECT_DOUBLE_EQ(graph.weight(0, 3), -2.5);
  EXPECT_FALSE(graph.are_connected(0, 2));
  ASSERT_TRUE(graph.geometry());
  EXPECT_EQ(graph.geometry()->content_hash(), geometry.content_hash());
  const auto raw = graph.geometry()->neighbor_connections({2});
  ASSERT_EQ(raw.size(), 2);
  for (const auto& connection : raw) {
    EXPECT_FALSE(connection.flavor.has_value());
    EXPECT_DOUBLE_EQ(connection.weight, 1.0);
  }
}

TEST_F(LatticeGraphTest, BondFlavorsOnlyRelabelExistingConnections) {
  const auto square = LatticeGraph::square(2, 2, false, false, 2.5)
                          .with_bond_flavors({{1, {1.0, 0.0}, flavor_x}});
  const auto original_hash = square.content_hash();
  const auto relabeled = square.with_bond_flavors(
      {{1, {0.0, 1.0}, flavor_y}, {2, {1.0, 1.0}, flavor_z}});
  EXPECT_EQ(relabeled.selected_shells(), (std::vector<std::uint64_t>{1}));
  EXPECT_EQ(relabeled.edge_coloring(), square.edge_coloring());
  EXPECT_TRUE(relabeled.adjacency_matrix().isApprox(square.adjacency_matrix()));
  EXPECT_EQ(relabeled.geometry()->content_hash(),
            square.geometry()->content_hash());
  auto expected = square.connections();
  for (auto& connection : expected) {
    connection.flavor =
        connection.bond_class.axis.isApprox(Eigen::RowVector2d(0.0, 1.0))
            ? std::optional<BondFlavorId>(flavor_y)
            : std::nullopt;
  }
  expect_same_connections(relabeled.connections(), expected);
  EXPECT_NE(relabeled.content_hash(), original_hash);
  EXPECT_EQ(square.content_hash(), original_hash);
  for (auto& connection : expected) connection.flavor.reset();
  expect_same_connections(relabeled.with_bond_flavors({}).connections(),
                          expected);
}

TEST_F(LatticeGraphTest, FromGeometryRejectsInvalidInputs) {
  const auto geometry = LatticeGeometry::chain(3);
  EXPECT_THROW(LatticeGraph::from_geometry(geometry, {0}),
               std::invalid_argument);
  for (double value : {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(LatticeGraph::from_geometry(geometry, {1}, {}, value),
                 std::invalid_argument);
  }
  for (double tolerance : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(LatticeGraph::from_geometry(geometry, {1}, {}, 1.0, tolerance),
                 std::invalid_argument);
  }
  EXPECT_THROW(
      LatticeGraph::from_geometry(geometry, {1}, {{0, {1.0, 0.0}, flavor_x}}),
      std::invalid_argument);
  EXPECT_THROW(
      LatticeGraph::from_geometry(geometry, {1}, {{1, {0.0, 0.0}, flavor_x}}),
      std::invalid_argument);
  EXPECT_THROW(LatticeGraph::from_geometry(
                   geometry, {1},
                   {{1, {1.0, 0.0}, flavor_x}, {1, {-2.0, 0.0}, flavor_y}}),
               std::invalid_argument);
}

TEST_F(LatticeGraphTest, FromConnectionsRejectsInvalidRecords) {
  const NeighborConnection valid{
      0, 1, {7, 29, {1.0, 0.0}}, {1.0, 0.0}, {0, 0}, flavor_x, 1.0};
  const auto expect_invalid = [&](const auto& mutate) {
    auto record = valid;
    mutate(record);
    EXPECT_THROW(LatticeGraph::from_connections(2, {record}),
                 std::invalid_argument);
  };
  expect_invalid([](auto& record) { record.site_j = 2; });
  expect_invalid([](auto& record) { record.bond_class.shell = 0; });
  expect_invalid([](auto& record) { record.bond_class.axis *= 2.0; });
  expect_invalid([](auto& record) { record.displacement.setZero(); });
  expect_invalid([](auto& record) { record.site_j = record.site_i; });
  expect_invalid([](auto& record) {
    record.weight = std::numeric_limits<double>::infinity();
  });
  expect_invalid([](auto& record) {
    record.displacement.x() = std::numeric_limits<double>::quiet_NaN();
  });
  EXPECT_THROW(LatticeGraph::from_connections(2, {valid, valid}),
               std::invalid_argument);
  auto reversed = valid;
  reversed.site_i = 1;
  reversed.site_j = 0;
  reversed.displacement = -valid.displacement;
  reversed.bond_class.shell = 8;
  EXPECT_THROW(LatticeGraph::from_connections(2, {valid, reversed}),
               std::invalid_argument);
  reversed.image_shift[0] = std::numeric_limits<std::int64_t>::min();
  EXPECT_THROW(LatticeGraph::from_connections(2, {reversed}),
               std::overflow_error);
  EXPECT_THROW(LatticeGraph::from_connections(2, {}, nullptr, {0}),
               std::invalid_argument);
  EXPECT_THROW(
      LatticeGraph::from_connections(
          2, {}, std::make_shared<LatticeGeometry>(LatticeGeometry::chain(3))),
      std::invalid_argument);
  EXPECT_THROW(
      LatticeGraph::from_connections(
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()) + 1, {}),
      std::overflow_error);
  auto first = valid;
  auto second = valid;
  first.weight = second.weight = std::numeric_limits<double>::max();
  second.image_shift[0] = 1;
  EXPECT_THROW(LatticeGraph::from_connections(2, {first, second}),
               std::overflow_error);
}

TEST_F(LatticeGraphTest, EmptySelectionRemainsDistinctFromAbsentMetadata) {
  const auto absent =
      LatticeGraph::from_dense_matrix(Eigen::MatrixXd::Zero(1, 1));
  const auto empty = LatticeGraph::from_connections(1, {});
  const auto selected = LatticeGraph::from_connections(1, {}, nullptr, {64});
  const auto finite =
      LatticeGraph::from_geometry(LatticeGeometry::chain(1), {9, 1, 9});
  EXPECT_FALSE(absent.to_json().contains("connections"));
  EXPECT_TRUE(empty.to_json().at("connections").empty());
  EXPECT_TRUE(empty.selected_shells().empty());
  EXPECT_EQ(selected.selected_shells(), (std::vector<std::uint64_t>{64}));
  EXPECT_EQ(finite.selected_shells(), (std::vector<std::uint64_t>{1, 9}));
  EXPECT_NE(empty.content_hash(), absent.content_hash());
  EXPECT_NE(empty.content_hash(), selected.content_hash());
  const std::string filename = "test_empty_connections.lattice_graph.h5";
  for (const auto& graph : {absent, empty, selected, finite}) {
    EXPECT_TRUE(graph.connections().empty());
    graph.to_hdf5_file(filename);
    const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
    std::filesystem::remove(filename);
    for (const auto& restored :
         {LatticeGraph::from_json(graph.to_json()), hdf5}) {
      EXPECT_EQ(restored.to_json(), graph.to_json());
      EXPECT_EQ(restored.content_hash(), graph.content_hash());
    }
  }
}

TEST_F(LatticeGraphTest, FactoryWeightsAndColoringSurviveRoundTrips) {
  const std::string filename = "test_factory_weights.lattice_graph.h5";
  for (double weight : {0.0, -0.75, 1.0e-200, 1.0e200,
                        std::numeric_limits<double>::denorm_min()}) {
    // The two physical images share the legacy factory's one pair weight.
    const auto graph = LatticeGraph::chain(2, true, weight);
    ASSERT_EQ(graph.connections().size(), 2);
    EXPECT_DOUBLE_EQ(graph.weight(0, 1), weight);
    graph.to_hdf5_file(filename);
    const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
    std::filesystem::remove(filename);
    for (const auto& restored :
         {LatticeGraph::from_json(graph.to_json()), hdf5}) {
      EXPECT_DOUBLE_EQ(restored.weight(0, 1), weight);
      EXPECT_EQ(restored.edge_coloring(), graph.edge_coloring());
      EXPECT_EQ(restored.num_nonzeros(), graph.num_nonzeros());
      expect_same_connections(restored.connections(), graph.connections());
    }
    const auto doubled = LatticeGraph::make_bidirectional(graph);
    EXPECT_DOUBLE_EQ(doubled.weight(0, 1), 2.0 * weight);
    EXPECT_FALSE(doubled.edge_coloring().has_value());
  }
}

TEST_F(LatticeGraphTest, DirectedAdjacencyConstructorsAndRoundTrips) {
  const std::map<Edge, double> edges = {
      {{0, 1}, 2.5}, {{1, 0}, -0.5}, {{1, 2}, -3.0}, {{2, 2}, 1.25}};
  const LatticeGraph graph(edges, 4);
  EXPECT_EQ(LatticeGraph(edges).num_sites(), 3);
  const auto adjacency = graph.adjacency_matrix();
  const std::string filename = "test_directed.lattice_graph.h5";
  for (const auto& source :
       {graph, LatticeGraph::from_dense_matrix(adjacency),
        LatticeGraph::from_sparse_matrix(graph.sparse_adjacency_matrix())}) {
    EXPECT_FALSE(source.is_symmetric());
    EXPECT_FALSE(source.geometry());
    EXPECT_TRUE(source.connections().empty());
    EXPECT_TRUE(source.selected_shells().empty());
    EXPECT_FALSE(source.to_json().contains("connections"));
    EXPECT_TRUE(source.adjacency_matrix().isApprox(adjacency));
    source.to_hdf5_file(filename);
    const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
    std::filesystem::remove(filename);
    for (const auto& restored :
         {LatticeGraph::from_json(source.to_json()), hdf5}) {
      EXPECT_TRUE(restored.adjacency_matrix().isApprox(adjacency));
      EXPECT_FALSE(restored.geometry());
      EXPECT_FALSE(restored.is_symmetric());
      EXPECT_EQ(restored.content_hash(), source.content_hash());
    }
    const auto bidirectional = LatticeGraph::make_bidirectional(source);
    EXPECT_TRUE(bidirectional.is_symmetric());
    EXPECT_TRUE(bidirectional.adjacency_matrix().isApprox(
        adjacency + adjacency.transpose()));
    EXPECT_DOUBLE_EQ(bidirectional.weight(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(bidirectional.weight(2, 2), 2.5);
  }
  EXPECT_THROW(LatticeGraph::from_dense_matrix(Eigen::MatrixXd::Zero(2, 3)),
               std::invalid_argument);
  EXPECT_THROW(
      LatticeGraph::from_sparse_matrix(Eigen::SparseMatrix<double>(2, 3)),
      std::invalid_argument);
  EXPECT_THROW((LatticeGraph(edges, 2)), std::invalid_argument);
}

TEST_F(LatticeGraphTest, LegacyGeometryPreservesAdjacencyAndCoordinates) {
  const nlohmann::json legacy = {
      {"num_sites", 3},
      {"is_symmetric", false},
      {"adjacency_sparse", {{0, 2, 2.5}, {1, 0, -1.0}}},
      {"positions", {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}}},
      {"periods", {{10.0, 0.0}}}};
  Eigen::MatrixXd positions(3, 2);
  positions << 0.0, 0.0, 1.0, 0.0, 2.0, 0.0;
  Eigen::MatrixXd periods(1, 2);
  periods << 10.0, 0.0;
  const LatticeGeometry geometry(positions, periods);
  const LatticeGraph adjacency({{{0, 2}, 2.5}, {{1, 0}, -1.0}}, 3);
  const std::string filename = "test_legacy_geometry.lattice_graph.h5";
  adjacency.to_hdf5_file(filename);
  {
    H5::H5File file(filename, H5F_ACC_RDWR);
    auto root = file.openGroup("/");
    geometry.to_hdf5(root);
  }
  const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
  std::filesystem::remove(filename);
  for (const auto& restored : {LatticeGraph::from_json(legacy), hdf5}) {
    EXPECT_TRUE(
        restored.adjacency_matrix().isApprox(adjacency.adjacency_matrix()));
    EXPECT_FALSE(restored.is_symmetric());
    ASSERT_TRUE(restored.geometry());
    EXPECT_TRUE(restored.geometry()->positions().isApprox(positions));
    ASSERT_TRUE(restored.geometry()->periods().has_value());
    EXPECT_TRUE(restored.geometry()->periods()->isApprox(periods));
  }
}

TEST_F(LatticeGraphTest, TriangularConstructor) {
  // 3x4 triangular lattice (12 sites)
  //
  //   9 -- 10 -- 11
  //   |  /  |  /  |
  //   6 --- 7 --- 8
  //   |  /  |  /  |
  //   3 --- 4 --- 5
  //   |  /  |  /  |
  //   0 --- 1 --- 2

  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  std::map<Edge, double> expected_edges = {
      // Right
      {{0, 1}, 1.0},
      {{1, 2}, 1.0},
      {{3, 4}, 1.0},
      {{4, 5}, 1.0},
      {{6, 7}, 1.0},
      {{7, 8}, 1.0},
      {{9, 10}, 1.0},
      {{10, 11}, 1.0},
      // Up
      {{0, 3}, 1.0},
      {{1, 4}, 1.0},
      {{2, 5}, 1.0},
      {{3, 6}, 1.0},
      {{4, 7}, 1.0},
      {{5, 8}, 1.0},
      {{6, 9}, 1.0},
      {{7, 10}, 1.0},
      {{8, 11}, 1.0},
      // Diagonal (upper-right)
      {{0, 4}, 1.0},
      {{1, 5}, 1.0},
      {{3, 7}, 1.0},
      {{4, 8}, 1.0},
      {{6, 10}, 1.0},
      {{7, 11}, 1.0},
  };
  auto expected =
      LatticeGraph::make_bidirectional(LatticeGraph(expected_edges, 12));

  auto tri = LatticeGraph::triangular(3, 4);
  EXPECT_EQ(tri.num_sites(), 12);
  EXPECT_EQ(tri.num_edges(), 23);
  EXPECT_TRUE(tri.is_symmetric());
  EXPECT_TRUE(tri.adjacency_matrix().isApprox(expected.adjacency_matrix()));

  // periodic_y only: up wraps + diagonal y-wraps (no right wraps, no corner)
  {
    std::map<Edge, double> py_edges = expected_edges;
    py_edges[{0, 9}] = 1.0;   // up wrap
    py_edges[{1, 10}] = 1.0;  // up wrap
    py_edges[{2, 11}] = 1.0;  // up wrap
    py_edges[{1, 9}] = 1.0;   // diagonal y-wrap
    py_edges[{2, 10}] = 1.0;  // diagonal y-wrap
    auto expected_py =
        LatticeGraph::make_bidirectional(LatticeGraph(py_edges, 12));

    auto tri_py = LatticeGraph::triangular(3, 4, false, true);
    EXPECT_EQ(tri_py.num_sites(), 12);
    EXPECT_EQ(tri_py.num_edges(), 28);  // 23 + 5
    EXPECT_TRUE(tri_py.is_symmetric());
    EXPECT_TRUE(
        tri_py.adjacency_matrix().isApprox(expected_py.adjacency_matrix()));
  }

  // periodic_x only: right wraps + diagonal x-wraps (no up wraps, no corner)
  {
    std::map<Edge, double> px_edges = expected_edges;
    px_edges[{0, 2}] = 1.0;   // right wrap
    px_edges[{3, 5}] = 1.0;   // right wrap
    px_edges[{6, 8}] = 1.0;   // right wrap
    px_edges[{9, 11}] = 1.0;  // right wrap
    px_edges[{2, 3}] = 1.0;   // diagonal x-wrap
    px_edges[{5, 6}] = 1.0;   // diagonal x-wrap
    px_edges[{8, 9}] = 1.0;   // diagonal x-wrap
    auto expected_px =
        LatticeGraph::make_bidirectional(LatticeGraph(px_edges, 12));

    auto tri_px = LatticeGraph::triangular(3, 4, true, false);
    EXPECT_EQ(tri_px.num_sites(), 12);
    EXPECT_EQ(tri_px.num_edges(), 30);  // 23 + 7
    EXPECT_TRUE(tri_px.is_symmetric());
    EXPECT_TRUE(
        tri_px.adjacency_matrix().isApprox(expected_px.adjacency_matrix()));
  }

  // Both periodic: all wrap edges + corner diagonal
  {
    std::map<Edge, double> pxy_edges = expected_edges;
    pxy_edges[{0, 2}] = 1.0;   // right wrap
    pxy_edges[{3, 5}] = 1.0;   // right wrap
    pxy_edges[{6, 8}] = 1.0;   // right wrap
    pxy_edges[{9, 11}] = 1.0;  // right wrap
    pxy_edges[{0, 9}] = 1.0;   // up wrap
    pxy_edges[{1, 10}] = 1.0;  // up wrap
    pxy_edges[{2, 11}] = 1.0;  // up wrap
    pxy_edges[{2, 3}] = 1.0;   // diagonal x-wrap
    pxy_edges[{5, 6}] = 1.0;   // diagonal x-wrap
    pxy_edges[{8, 9}] = 1.0;   // diagonal x-wrap
    pxy_edges[{1, 9}] = 1.0;   // diagonal y-wrap
    pxy_edges[{2, 10}] = 1.0;  // diagonal y-wrap
    pxy_edges[{11, 0}] = 1.0;  // diagonal corner wrap
    auto expected_pxy =
        LatticeGraph::make_bidirectional(LatticeGraph(pxy_edges, 12));

    auto tri_pxy = LatticeGraph::triangular(3, 4, true, true);
    EXPECT_EQ(tri_pxy.num_sites(), 12);
    EXPECT_EQ(tri_pxy.num_edges(), 36);  // 23 + 8 + 5
    EXPECT_TRUE(tri_pxy.is_symmetric());
    EXPECT_TRUE(
        tri_pxy.adjacency_matrix().isApprox(expected_pxy.adjacency_matrix()));
  }
}

TEST_F(LatticeGraphTest, HoneycombConstructor) {
  // Fully periodic 3x4 honeycomb lattice (24 sites)
  //
  //           18-19-20-21-22-23
  //            |     |     |
  //        12-13-14-15-16-17
  //         |     |     |
  //      6--7--8--9-10-11
  //      |     |     |
  //   0--1--2--3--4--5

  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  std::map<Edge, double> expected_edges = {
      // horizontal
      {{0, 1}, 1.0},
      {{1, 2}, 1.0},
      {{2, 3}, 1.0},
      {{3, 4}, 1.0},
      {{4, 5}, 1.0},

      {{6, 7}, 1.0},
      {{7, 8}, 1.0},
      {{8, 9}, 1.0},
      {{9, 10}, 1.0},
      {{10, 11}, 1.0},

      {{12, 13}, 1.0},
      {{13, 14}, 1.0},
      {{14, 15}, 1.0},
      {{15, 16}, 1.0},
      {{16, 17}, 1.0},

      {{18, 19}, 1.0},
      {{19, 20}, 1.0},
      {{20, 21}, 1.0},
      {{21, 22}, 1.0},
      {{22, 23}, 1.0},
      // vertical
      {{1, 6}, 1.0},
      {{3, 8}, 1.0},
      {{5, 10}, 1.0},
      {{7, 12}, 1.0},
      {{9, 14}, 1.0},
      {{11, 16}, 1.0},
      {{13, 18}, 1.0},
      {{15, 20}, 1.0},
      {{17, 22}, 1.0},
  };
  auto expected =
      LatticeGraph::make_bidirectional(LatticeGraph(expected_edges, 24));

  auto hc = LatticeGraph::honeycomb(3, 4);
  EXPECT_EQ(hc.num_sites(), 24);
  EXPECT_EQ(hc.num_edges(), 29);
  EXPECT_TRUE(hc.is_symmetric());
  EXPECT_TRUE(hc.adjacency_matrix().isApprox(expected.adjacency_matrix()));

  // periodic_y only: vertical wraps
  {
    std::map<Edge, double> py_edges = expected_edges;
    py_edges[{0, 19}] = 1.0;
    py_edges[{2, 21}] = 1.0;
    py_edges[{4, 23}] = 1.0;
    auto expected_py =
        LatticeGraph::make_bidirectional(LatticeGraph(py_edges, 24));

    auto hc_py = LatticeGraph::honeycomb(3, 4, false, true);
    EXPECT_EQ(hc_py.num_sites(), 24);
    EXPECT_EQ(hc_py.num_edges(), 32);
    EXPECT_TRUE(hc_py.is_symmetric());
    EXPECT_TRUE(
        hc_py.adjacency_matrix().isApprox(expected_py.adjacency_matrix()));
  }

  // periodic_x only: horizontal wraps
  {
    std::map<Edge, double> px_edges = expected_edges;
    px_edges[{0, 5}] = 1.0;
    px_edges[{6, 11}] = 1.0;
    px_edges[{12, 17}] = 1.0;
    px_edges[{18, 23}] = 1.0;
    auto expected_px =
        LatticeGraph::make_bidirectional(LatticeGraph(px_edges, 24));

    auto hc_px = LatticeGraph::honeycomb(3, 4, true, false);
    EXPECT_EQ(hc_px.num_sites(), 24);
    EXPECT_EQ(hc_px.num_edges(), 33);
    EXPECT_TRUE(hc_px.is_symmetric());
    EXPECT_TRUE(
        hc_px.adjacency_matrix().isApprox(expected_px.adjacency_matrix()));
  }

  // Both periodic: horizontal + vertical wraps
  {
    std::map<Edge, double> pxy_edges = expected_edges;
    pxy_edges[{0, 5}] = 1.0;    // horizontal wrap
    pxy_edges[{6, 11}] = 1.0;   // horizontal wrap
    pxy_edges[{12, 17}] = 1.0;  // horizontal wrap
    pxy_edges[{18, 23}] = 1.0;  // horizontal wrap
    pxy_edges[{0, 19}] = 1.0;   // vertical wrap
    pxy_edges[{2, 21}] = 1.0;   // vertical wrap
    pxy_edges[{4, 23}] = 1.0;   // vertical wrap
    auto expected_pxy =
        LatticeGraph::make_bidirectional(LatticeGraph(pxy_edges, 24));

    auto hc_pxy = LatticeGraph::honeycomb(3, 4, true, true);
    EXPECT_EQ(hc_pxy.num_sites(), 24);
    EXPECT_EQ(hc_pxy.num_edges(), 36);  // 3 * nx * ny on a torus
    EXPECT_TRUE(hc_pxy.is_symmetric());
    EXPECT_TRUE(
        hc_pxy.adjacency_matrix().isApprox(expected_pxy.adjacency_matrix()));
  }
}

TEST_F(LatticeGraphTest, KagomeConstructor) {
  // 3x2 kagome lattice (18 sites)
  //
  //           11     14      17
  //          / \     / \     / \
  //         9--10--12--13--15--16
  //        /     \ /     \ /
  //       2       5       8
  //      / \     / \     / \
  //     0---1---3---4---6---7

  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  std::map<Edge, double> expected_edges = {
      // Horizontal
      {{0, 1}, 1.0},
      {{1, 3}, 1.0},
      {{3, 4}, 1.0},
      {{4, 6}, 1.0},
      {{6, 7}, 1.0},
      {{9, 10}, 1.0},
      {{10, 12}, 1.0},
      {{12, 13}, 1.0},
      {{13, 15}, 1.0},
      {{15, 16}, 1.0},
      // vertical
      {{0, 2}, 1.0},
      {{1, 2}, 1.0},
      {{3, 5}, 1.0},
      {{4, 5}, 1.0},
      {{6, 8}, 1.0},
      {{7, 8}, 1.0},
      {{2, 9}, 1.0},
      {{5, 10}, 1.0},
      {{5, 12}, 1.0},
      {{8, 13}, 1.0},
      {{8, 15}, 1.0},
      {{9, 11}, 1.0},
      {{10, 11}, 1.0},
      {{12, 14}, 1.0},
      {{13, 14}, 1.0},
      {{15, 17}, 1.0},
      {{16, 17}, 1.0},
  };
  auto expected =
      LatticeGraph::make_bidirectional(LatticeGraph(expected_edges, 18));

  auto kg = LatticeGraph::kagome(3, 2);
  EXPECT_EQ(kg.num_sites(), 18);
  EXPECT_EQ(kg.num_edges(), 27);
  EXPECT_TRUE(kg.is_symmetric());
  EXPECT_TRUE(kg.adjacency_matrix().isApprox(expected.adjacency_matrix()));

  // periodic_y only: vertical wraps + diagonal y-wraps
  {
    std::map<Edge, double> py_edges = expected_edges;
    // vertical wraps:
    py_edges[{0, 11}] = 1.0;  // vertical wrap
    py_edges[{3, 14}] = 1.0;  // vertical wrap
    py_edges[{6, 17}] = 1.0;  // vertical wrap
    // diagonal y-wraps:
    py_edges[{1, 14}] = 1.0;  // diagonal y-wrap
    py_edges[{4, 17}] = 1.0;  // diagonal y-wrap
    auto expected_py =
        LatticeGraph::make_bidirectional(LatticeGraph(py_edges, 18));

    auto kg_py = LatticeGraph::kagome(3, 2, false, true);
    EXPECT_EQ(kg_py.num_sites(), 18);
    EXPECT_EQ(kg_py.num_edges(), 32);  // 27 + 5
    EXPECT_TRUE(kg_py.is_symmetric());
    EXPECT_TRUE(
        kg_py.adjacency_matrix().isApprox(expected_py.adjacency_matrix()));
  }

  // periodic_x only: horizontal wraps + diagonal x-wraps
  {
    std::map<Edge, double> px_edges = expected_edges;
    // horizontal wraps:
    px_edges[{0, 7}] = 1.0;   // horizontal wrap
    px_edges[{9, 16}] = 1.0;  // horizontal wrap
    // diagonal x-wraps:
    px_edges[{2, 16}] = 1.0;  // diagonal x-wrap
    auto expected_px =
        LatticeGraph::make_bidirectional(LatticeGraph(px_edges, 18));

    auto kg_px = LatticeGraph::kagome(3, 2, true, false);
    EXPECT_EQ(kg_px.num_sites(), 18);
    EXPECT_EQ(kg_px.num_edges(), 30);  // 27 + 3
    EXPECT_TRUE(kg_px.is_symmetric());
    EXPECT_TRUE(
        kg_px.adjacency_matrix().isApprox(expected_px.adjacency_matrix()));
  }

  // Both periodic: all wraps + corner diagonal
  {
    std::map<Edge, double> pxy_edges = expected_edges;
    // horizontal wraps
    pxy_edges[{0, 7}] = 1.0;
    pxy_edges[{9, 16}] = 1.0;
    // vertical wraps
    pxy_edges[{0, 11}] = 1.0;
    pxy_edges[{3, 14}] = 1.0;
    pxy_edges[{6, 17}] = 1.0;
    // diagonal x-wrap
    pxy_edges[{2, 16}] = 1.0;
    // diagonal y-wraps
    pxy_edges[{1, 14}] = 1.0;
    pxy_edges[{4, 17}] = 1.0;
    // diagonal corner wrap:
    pxy_edges[{7, 11}] = 1.0;
    auto expected_pxy =
        LatticeGraph::make_bidirectional(LatticeGraph(pxy_edges, 18));

    auto kg_pxy = LatticeGraph::kagome(3, 2, true, true);
    EXPECT_EQ(kg_pxy.num_sites(), 18);
    EXPECT_EQ(kg_pxy.num_edges(), 36);  // 27 + 9
    EXPECT_TRUE(kg_pxy.is_symmetric());
    EXPECT_TRUE(
        kg_pxy.adjacency_matrix().isApprox(expected_pxy.adjacency_matrix()));
  }
}

// Coloring helper: confirm no two same-color edges share a vertex.
static void check_valid_edge_coloring(const EdgeColoring& coloring) {
  std::map<std::uint64_t, std::set<int>> incident;
  for (const auto& [edge, color] : coloring) {
    auto [a, b] = edge;
    EXPECT_EQ(incident[a].count(color), 0u)
        << "vertex " << a << " has two edges of color " << color;
    EXPECT_EQ(incident[b].count(color), 0u)
        << "vertex " << b << " has two edges of color " << color;
    incident[a].insert(color);
    incident[b].insert(color);
  }
}

TEST_F(LatticeGraphTest, ColorCount) {
  auto chain_open = LatticeGraph::chain(5, false);
  ASSERT_TRUE(chain_open.edge_coloring().has_value());
  std::set<int> chain_open_colors;
  for (const auto& [e, c] : *chain_open.edge_coloring())
    chain_open_colors.insert(c);
  // Open chain uses exactly 2 colors (alternating)
  EXPECT_EQ(chain_open_colors.size(), 2u);

  auto chain_periodic_even = LatticeGraph::chain(6, true);
  ASSERT_TRUE(chain_periodic_even.edge_coloring().has_value());
  std::set<int> chain_even_colors;
  for (const auto& [e, c] : *chain_periodic_even.edge_coloring())
    chain_even_colors.insert(c);
  // Even periodic chain uses exactly 2 colors
  EXPECT_EQ(chain_even_colors.size(), 2u);

  // Odd periodic chain needs 3 colors
  auto chain_periodic_odd = LatticeGraph::chain(5, true);
  ASSERT_TRUE(chain_periodic_odd.edge_coloring().has_value());
  std::set<int> chain_odd_colors;
  for (const auto& [e, c] : *chain_periodic_odd.edge_coloring())
    chain_odd_colors.insert(c);
  EXPECT_EQ(chain_odd_colors.size(), 3u);

  auto hc = LatticeGraph::honeycomb(3, 3, true, true);
  ASSERT_TRUE(hc.edge_coloring().has_value());
  // Honeycomb uses exactly 3 colors.
  std::set<int> hc_colors;
  for (const auto& [e, c] : *hc.edge_coloring()) hc_colors.insert(c);
  EXPECT_EQ(hc_colors.size(), 3u);
}

TEST_F(LatticeGraphTest, EdgeColoringIsValid) {
  // For every factory-built lattice, the coloring must be present and valid.
  std::vector<LatticeGraph> graphs;
  graphs.emplace_back(LatticeGraph::chain(8, true));
  graphs.emplace_back(LatticeGraph::square(4, 4, true, true));
  graphs.emplace_back(LatticeGraph::triangular(4, 4, true, true));
  graphs.emplace_back(LatticeGraph::honeycomb(3, 3, true, true));
  graphs.emplace_back(LatticeGraph::kagome(2, 3, true, true));

  for (const auto& g : graphs) {
    ASSERT_TRUE(g.edge_coloring().has_value());
    check_valid_edge_coloring(*g.edge_coloring());
  }

  // Custom adjacency: no coloring by default.
  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  std::map<Edge, double> custom_edges = {
      {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 3}, 1.0}, {{3, 0}, 1.0}};
  LatticeGraph custom(custom_edges, 4);
  EXPECT_FALSE(custom.edge_coloring().has_value());
}

TEST_F(LatticeGraphTest, EdgeColoringIsImmutable) {
  auto sq = LatticeGraph::square(4, 4, true, true);
  const auto& first = sq.edge_coloring();
  const auto& second = sq.edge_coloring();
  EXPECT_EQ(&first, &second);
}

TEST_F(LatticeGraphTest, TrivialEdgeColoring) {
  // Build a small graph and check trivial coloring assigns unique colors.
  auto chain = LatticeGraph::chain(5);
  const auto& adj = chain.sparse_adjacency_matrix();
  auto coloring = trivial_edge_coloring(adj);

  // 4 edges in a 5-site open chain
  EXPECT_EQ(coloring.size(), 4u);

  // Each edge should have a distinct color 0..3
  std::set<int> colors;
  for (const auto& [edge, c] : coloring) {
    colors.insert(c);
  }
  EXPECT_EQ(colors.size(), 4u);
  EXPECT_EQ(*colors.begin(), 0);
  EXPECT_EQ(*colors.rbegin(), 3);

  // Also valid as an edge coloring (trivially, since all colors differ)
  check_valid_edge_coloring(coloring);
}

TEST_F(LatticeGraphTest, TrivialEdgeColoringEmpty) {
  // Single-site graph has no edges → empty coloring
  auto single = LatticeGraph::chain(1);
  auto coloring = trivial_edge_coloring(single.sparse_adjacency_matrix());
  EXPECT_TRUE(coloring.empty());
}

TEST_F(LatticeGraphTest, ColoringSeedDeterministic) {
  // Same seed → same coloring.
  auto tri_a = LatticeGraph::triangular(3, 3, true, true, 1.0, 42);
  auto tri_b = LatticeGraph::triangular(3, 3, true, true, 1.0, 42);
  ASSERT_TRUE(tri_a.edge_coloring().has_value());
  ASSERT_TRUE(tri_b.edge_coloring().has_value());
  EXPECT_EQ(*tri_a.edge_coloring(), *tri_b.edge_coloring());

  // Different seed may produce a different coloring (or same, but at
  // least both must be valid).
  auto tri_c = LatticeGraph::triangular(3, 3, true, true, 1.0, 99);
  ASSERT_TRUE(tri_c.edge_coloring().has_value());
  check_valid_edge_coloring(*tri_c.edge_coloring());
}

TEST_F(LatticeGraphTest, KagomeColoringSeed) {
  auto kg_a = LatticeGraph::kagome(2, 2, true, true, 1.0, 7);
  auto kg_b = LatticeGraph::kagome(2, 2, true, true, 1.0, 7);
  ASSERT_TRUE(kg_a.edge_coloring().has_value());
  ASSERT_TRUE(kg_b.edge_coloring().has_value());
  EXPECT_EQ(*kg_a.edge_coloring(), *kg_b.edge_coloring());
  check_valid_edge_coloring(*kg_a.edge_coloring());
}

TEST_F(LatticeGraphTest, ColorEdgesMatchesGreedyTraversal) {
  const auto graph = LatticeGraph::square(3, 3, false, false, 0.0);
  std::vector<Edge> pairs = {{4, 8}, {0, 5}, {2, 7}, {1, 6}, {1, 2}, {0, 1},
                             {3, 7}, {3, 4}, {5, 8}, {2, 4}, {0, 3}, {4, 6},
                             {6, 8}, {1, 4}, {0, 2}, {4, 8}};
  std::map<Edge, double> weights;
  for (const auto& pair : pairs) weights[pair] = -2.5;
  const LatticeGraph support(weights, graph.num_sites());
  // Compare labels, not only color counts; shuffled trials depend on the
  // initial column-major traversal of the canonical support.
  for (int seed : {0, 7, 42}) {
    for (int trials : {1, 2, 32}) {
      const auto expected =
          greedy_edge_coloring(support.sparse_adjacency_matrix(), seed, trials);
      EXPECT_EQ(graph.color_edges(pairs, seed, trials), expected);
      std::reverse(pairs.begin(), pairs.end());
      EXPECT_EQ(graph.color_edges(pairs, seed, trials), expected);
      check_valid_edge_coloring(expected);
    }
  }
  EXPECT_EQ(graph.color_edges(pairs),
            greedy_edge_coloring(support.sparse_adjacency_matrix(), 0, 32));
}

TEST_F(LatticeGraphTest, ColorEdgesRecolorsSubsetsIndependently) {
  const auto graph = LatticeGraph::chain(3, true);
  const auto stored = graph.edge_coloring();
  const auto union_coloring = graph.color_edges({{0, 1}, {0, 2}, {1, 2}}, 0, 1);
  const auto subset = graph.color_edges({{1, 2}}, 0, 1);
  EXPECT_EQ(subset, (EdgeColoring{{{1, 2}, 0}}));
  EXPECT_NE(subset.at({1, 2}), union_coloring.at({1, 2}));
  EXPECT_EQ(graph.color_edges({{1, 2}}, 0, 1), subset);
  EXPECT_EQ(graph.edge_coloring(), stored);
  check_valid_edge_coloring(union_coloring);
}

TEST_F(LatticeGraphTest, ColorEdgesValidatesActiveSupport) {
  const auto graph = LatticeGraph::chain(3);
  for (const Edge pair : {Edge{1, 0}, Edge{1, 1}, Edge{0, 3},
                          Edge{0, std::numeric_limits<std::uint64_t>::max()}}) {
    EXPECT_THROW(graph.color_edges({pair}), std::invalid_argument);
  }
  EXPECT_TRUE(graph.color_edges({}).empty());
  EXPECT_TRUE(graph.color_edges({{0, 1}}, 0, 0).empty());
  EXPECT_TRUE(graph.color_edges({{0, 1}}, 0, -1).empty());
  EXPECT_EQ(graph.color_edges({{0, 1}, {0, 1}}), graph.color_edges({{0, 1}}));
}

TEST_F(LatticeGraphTest, Permute) {
  // Create a 2x3 square lattice (6 sites):
  // 3 -- 4 -- 5
  // |    |    |
  // 0 -- 1 -- 2
  //
  // Adjacency connections:
  // (0,1), (1,2), (3,4), (4,5) [horizontal]
  // (0,3), (1,4), (2,5) [vertical]
  auto sq = LatticeGraph::square(3, 2, false, false);
  ASSERT_TRUE(sq.edge_coloring().has_value());

  // Define a permutation path that is not an involution:
  std::vector<std::uint64_t> path = {1, 2, 0, 4, 5, 3};
  auto permuted = LatticeGraph::permute(sq, path);

  EXPECT_THROW(LatticeGraph::permute(sq, {0, 1}), std::invalid_argument);
  EXPECT_THROW(LatticeGraph::permute(sq, {0, 1, 2, 3, 4, 4}),
               std::invalid_argument);
  EXPECT_THROW(LatticeGraph::permute(sq, {0, 1, 2, 3, 4, 6}),
               std::invalid_argument);

  EXPECT_EQ(permuted.num_sites(), sq.num_sites());
  EXPECT_EQ(permuted.num_edges(), sq.num_edges());

  // Verify that new vertex i corresponds to old vertex path[i] (locks down
  // permutation direction)
  for (std::uint64_t i = 0; i < 6; ++i) {
    for (std::uint64_t j = i + 1; j < 6; ++j) {
      EXPECT_EQ(permuted.are_connected(i, j),
                sq.are_connected(path[i], path[j]));
    }
  }

  // Inverse permutation mapping for checking:
  // inv_p[old_site] = new_site
  std::vector<std::uint64_t> inv_p(6);
  for (std::uint64_t i = 0; i < 6; ++i) {
    inv_p[path[i]] = i;
  }

  // Assert are_connected(i,j) after permute matches the remapped coloring keys
  const auto& new_coloring = *permuted.edge_coloring();
  for (std::uint64_t i = 0; i < 6; ++i) {
    for (std::uint64_t j = i + 1; j < 6; ++j) {
      bool connected = permuted.are_connected(i, j);
      auto key = std::make_pair(i, j);
      bool in_coloring = (new_coloring.count(key) > 0);
      EXPECT_EQ(connected, in_coloring);

      if (connected) {
        // Find corresponding old vertices
        std::uint64_t old_u = path[i];
        std::uint64_t old_v = path[j];
        auto old_key = std::minmax(old_u, old_v);
        // Assert color matches
        EXPECT_EQ(new_coloring.at(key),
                  sq.edge_coloring()->at({old_key.first, old_key.second}));
      }
    }
  }

  // Confirms dfs_ordering=true yields path-consecutive adjacency
  auto ordered_sq = LatticeGraph::square(3, 3, false, false, 1.0, true);
  for (std::uint64_t i = 0; i < ordered_sq.num_sites() - 1; ++i) {
    EXPECT_TRUE(ordered_sq.are_connected(i, i + 1))
        << "ordered_sq sites " << i << " and " << (i + 1)
        << " are not connected";
  }
}
