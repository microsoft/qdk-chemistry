// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class LatticeGraphTest : public ::testing::Test {};

namespace {
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

TEST_F(LatticeGraphTest, SquareGeometricNeighborShells) {
  auto square = LatticeGraph::square(5, 5);
  const auto shells = square.nearest_neighbor_shells({1, 2, 3});
  const auto& first = shells.at(1);
  const auto& second = shells.at(2);
  const auto& third = shells.at(3);
  auto degree = [](const auto& shell, std::uint64_t site) {
    return std::count_if(shell.begin(), shell.end(), [site](const auto& edge) {
      return edge.first == site || edge.second == site;
    });
  };

  // Site 12 is the center. Square-lattice shell distances are 1, sqrt(2), 2.
  EXPECT_EQ(degree(first, 12), 4);
  EXPECT_EQ(degree(second, 12), 4);
  EXPECT_EQ(degree(third, 12), 4);
  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  EXPECT_NE(std::find(second.begin(), second.end(), Edge{12, 18}),
            second.end());
  EXPECT_EQ(std::find(second.begin(), second.end(), Edge{12, 14}),
            second.end());
  EXPECT_NE(std::find(third.begin(), third.end(), Edge{12, 14}), third.end());
}

TEST_F(LatticeGraphTest, HoneycombGeometricNeighborShells) {
  auto honeycomb = LatticeGraph::honeycomb(4, 4);
  const auto first = honeycomb.mth_nearest_neighbors(1);
  const auto second = honeycomb.mth_nearest_neighbors(2);
  const auto third = honeycomb.mth_nearest_neighbors(3);
  auto degree = [](const auto& shell, std::uint64_t site) {
    return std::count_if(shell.begin(), shell.end(), [site](const auto& edge) {
      return edge.first == site || edge.second == site;
    });
  };

  // A(1,1) is an interior site. The first three honeycomb shells have bulk
  // coordination numbers 3, 6, 3 and distances 1, sqrt(3), 2.
  constexpr std::uint64_t center = 10;
  EXPECT_EQ(degree(first, center), 3);
  EXPECT_EQ(degree(second, center), 6);
  EXPECT_EQ(degree(third, center), 3);
}

TEST_F(LatticeGraphTest, GeometricBondClassesAndHoneycombFlavors) {
  const std::vector<std::pair<LatticeGraph, std::size_t>> lattices = {
      {LatticeGraph::chain(5), 1},
      {LatticeGraph::square(5, 5), 2},
      {LatticeGraph::triangular(5, 5), 3},
      {LatticeGraph::kagome(4, 4), 3},
  };
  for (const auto& [lattice, expected_orientations] : lattices) {
    const auto connections = lattice.neighbor_connections({1});
    std::set<std::uint32_t> orientations;
    for (const auto& connection : connections) {
      orientations.insert(connection.bond_class.orientation);
      EXPECT_FALSE(connection.flavor.has_value());
    }
    EXPECT_EQ(orientations.size(), expected_orientations);
  }

  auto square = LatticeGraph::square(5, 5);
  const auto square_connections = square.neighbor_connections({1, 2});
  std::map<std::uint64_t, std::set<std::uint32_t>> square_orientations;
  for (const auto& connection : square_connections) {
    square_orientations[connection.bond_class.shell].insert(
        connection.bond_class.orientation);
    EXPECT_FALSE(connection.flavor.has_value());
  }
  EXPECT_EQ(square_orientations.at(1).size(), 2);
  EXPECT_EQ(square_orientations.at(2).size(), 2);

  auto honeycomb =
      LatticeGraph::honeycomb(5, 5).with_bond_flavors(honeycomb_flavor_ids());
  const auto connections = honeycomb.neighbor_connections({1, 2, 3});
  constexpr std::uint64_t center = 24;
  std::map<std::uint64_t, std::map<BondFlavorId, std::size_t>> flavor_degree;
  std::map<std::uint64_t, std::set<std::pair<std::uint64_t, std::uint64_t>>>
      projected_pairs;
  for (const auto& connection : connections) {
    ASSERT_TRUE(connection.flavor.has_value());
    projected_pairs[connection.bond_class.shell].emplace(connection.site_i,
                                                         connection.site_j);
    if (connection.site_i == center || connection.site_j == center) {
      ++flavor_degree[connection.bond_class.shell][*connection.flavor];
    }
  }
  for (std::uint64_t shell = 1; shell <= 3; ++shell) {
    const auto pairs = honeycomb.mth_nearest_neighbors(shell);
    EXPECT_EQ(projected_pairs.at(shell),
              (std::set<std::pair<std::uint64_t, std::uint64_t>>(pairs.begin(),
                                                                 pairs.end())));
  }
  for (BondFlavorId flavor : {flavor_x, flavor_y, flavor_z}) {
    EXPECT_EQ(flavor_degree.at(1).at(flavor), 1);
    EXPECT_EQ(flavor_degree.at(2).at(flavor), 2);
    EXPECT_EQ(flavor_degree.at(3).at(flavor), 1);
  }
}

TEST_F(LatticeGraphTest, HoneycombFactoriesDoNotAssignModelFlavors) {
  const auto connections =
      LatticeGraph::honeycomb(3, 3).neighbor_connections({1, 2, 3});
  EXPECT_TRUE(std::all_of(
      connections.begin(), connections.end(),
      [](const auto& connection) { return !connection.flavor.has_value(); }));
}

TEST_F(LatticeGraphTest, HoneycombOpenPlaquettePatches) {
  auto unit_cell = LatticeGraph::honeycomb(1, 1);
  EXPECT_EQ(unit_cell.num_sites(), 2);
  EXPECT_EQ(unit_cell.num_edges(), 1);

  auto hexagon = LatticeGraph::honeycomb_plaquettes(1, 1, false, false, 2.5);
  EXPECT_EQ(hexagon.num_sites(), 6);
  EXPECT_EQ(hexagon.num_edges(), 6);
  EXPECT_TRUE(hexagon.is_symmetric());
  ASSERT_TRUE(hexagon.positions().has_value());
  const auto& positions = *hexagon.positions();
  EXPECT_EQ(positions.rows(), 6);
  EXPECT_EQ(positions.cols(), 2);
  for (Eigen::Index site = 0;
       site < hexagon.sparse_adjacency_matrix().outerSize(); ++site) {
    EXPECT_EQ(hexagon.sparse_adjacency_matrix().innerVector(site).nonZeros(),
              2);
  }

  const auto flavored_hexagon =
      hexagon.with_bond_flavors(honeycomb_flavor_ids());
  const auto connections = flavored_hexagon.neighbor_connections({1, 2, 3});
  std::map<std::uint64_t, std::map<BondFlavorId, std::size_t>> counts;
  for (const auto& connection : connections) {
    ASSERT_TRUE(connection.flavor.has_value());
    ++counts[connection.bond_class.shell][*connection.flavor];
  }
  for (BondFlavorId flavor : {flavor_x, flavor_y, flavor_z}) {
    EXPECT_EQ(counts.at(1).at(flavor), 2);
    EXPECT_EQ(counts.at(2).at(flavor), 2);
    EXPECT_EQ(counts.at(3).at(flavor), 1);
  }
  EXPECT_EQ(hexagon.mth_nearest_neighbors(1).size(), 6);
  EXPECT_EQ(hexagon.mth_nearest_neighbors(2).size(), 6);
  EXPECT_EQ(hexagon.mth_nearest_neighbors(3).size(), 3);
  for (const auto& [shell, expected_distance] :
       std::vector<std::pair<std::uint64_t, double>>{
           {1, 1.0}, {2, std::sqrt(3.0)}, {3, 2.0}}) {
    for (const auto& [site_i, site_j] : hexagon.mth_nearest_neighbors(shell)) {
      EXPECT_NEAR((positions.row(site_j) - positions.row(site_i)).norm(),
                  expected_distance, 1.0e-12);
    }
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
  auto honeycomb = LatticeGraph::honeycomb(2, 2, true, true)
                       .with_bond_flavors(honeycomb_flavor_ids());
  const auto all_connections = honeycomb.neighbor_connections({1, 2, 3});
  for (std::uint64_t shell = 1; shell <= 3; ++shell) {
    std::set<std::pair<std::uint64_t, std::uint64_t>> projection;
    for (const auto& connection : all_connections) {
      if (connection.bond_class.shell == shell) {
        projection.emplace(connection.site_i, connection.site_j);
      }
    }
    const auto pairs = honeycomb.mth_nearest_neighbors(shell);
    EXPECT_EQ(projection, (std::set<std::pair<std::uint64_t, std::uint64_t>>(
                              pairs.begin(), pairs.end())));
  }
  std::vector<NeighborConnection> connections;
  std::copy_if(all_connections.begin(), all_connections.end(),
               std::back_inserter(connections), [](const auto& connection) {
                 return connection.bond_class.shell == 3;
               });
  std::map<std::pair<std::uint64_t, std::uint64_t>, std::set<BondFlavorId>>
      flavors_by_pair;
  for (const auto& connection : connections) {
    ASSERT_TRUE(connection.flavor.has_value());
    flavors_by_pair[{connection.site_i, connection.site_j}].insert(
        *connection.flavor);
  }
  EXPECT_TRUE(
      std::any_of(flavors_by_pair.begin(), flavors_by_pair.end(),
                  [](const auto& item) { return item.second.size() > 1; }));
  EXPECT_GT(connections.size(), honeycomb.mth_nearest_neighbors(3).size());
}

TEST_F(LatticeGraphTest, BondFlavorDefinitionsSurviveDataOperations) {
  auto honeycomb = LatticeGraph::honeycomb(3, 3, true, true)
                       .with_bond_flavors(honeycomb_flavor_ids());
  ASSERT_EQ(honeycomb.bond_flavor_definitions().size(), 9);

  auto flavor_signature = [](const LatticeGraph& graph) {
    std::vector<std::tuple<std::uint64_t, std::uint32_t, std::uint64_t,
                           std::uint64_t, BondFlavorId>>
        result;
    for (const auto& connection : graph.neighbor_connections({1, 2, 3})) {
      EXPECT_TRUE(connection.flavor.has_value());
      if (!connection.flavor.has_value()) continue;
      result.emplace_back(connection.bond_class.shell,
                          connection.bond_class.orientation, connection.site_i,
                          connection.site_j, *connection.flavor);
    }
    return result;
  };

  auto json_restored = LatticeGraph::from_json(honeycomb.to_json());
  EXPECT_EQ(flavor_signature(json_restored), flavor_signature(honeycomb));
  EXPECT_EQ(json_restored.content_hash(), honeycomb.content_hash());

  const std::filesystem::path filename =
      "test_lattice_graph_bond_flavors.lattice_graph.h5";
  honeycomb.to_hdf5_file(filename.string());
  auto hdf5_restored = LatticeGraph::from_hdf5_file(filename.string());
  EXPECT_EQ(flavor_signature(hdf5_restored), flavor_signature(honeycomb));
  EXPECT_EQ(hdf5_restored.content_hash(), honeycomb.content_hash());
  std::filesystem::remove(filename);

  std::vector<std::uint64_t> path(honeycomb.num_sites());
  std::iota(path.begin(), path.end(), 0);
  std::rotate(path.begin(), path.begin() + 1, path.end());
  auto permuted = LatticeGraph::permute(honeycomb, path);
  EXPECT_EQ(permuted.bond_flavor_definitions().size(), 9);
  EXPECT_EQ(permuted.neighbor_connections({1, 2, 3}).size(),
            honeycomb.neighbor_connections({1, 2, 3}).size());

  auto square =
      LatticeGraph::square(3, 3).with_bond_flavors({{1, {1.0, 0.0}, 1000}});
  const auto square_connections = square.neighbor_connections({1});
  EXPECT_TRUE(std::any_of(
      square_connections.begin(), square_connections.end(),
      [](const auto& connection) { return connection.flavor == 1000; }));
  EXPECT_TRUE(std::any_of(
      square_connections.begin(), square_connections.end(),
      [](const auto& connection) { return !connection.flavor.has_value(); }));
}

TEST_F(LatticeGraphTest, BondFlavorAxesAreScaleInvariant) {
  const auto square = LatticeGraph::square(2, 2);
  for (double scale : {1.0e-200, 1.0e200}) {
    const auto flavored = square.with_bond_flavors({{1, {scale, 0.0}, 1000}});
    ASSERT_EQ(flavored.bond_flavor_definitions().size(), 1);
    EXPECT_TRUE(flavored.bond_flavor_definitions()[0].axis.isApprox(
        Eigen::RowVector2d(1.0, 0.0)));
  }
}

TEST_F(LatticeGraphTest, BondFlavorPersistencePreservesTypes) {
  constexpr std::uint64_t shell = (std::uint64_t{1} << 53) + 1;
  constexpr BondFlavorId flavor = std::numeric_limits<BondFlavorId>::max();
  const auto graph = LatticeGraph::square(2, 2).with_bond_flavors(
      {{shell, {1.0, 0.0}, flavor}});
  const std::filesystem::path filename =
      "test_lattice_graph_bond_flavor_types.lattice_graph.h5";
  graph.to_hdf5_file(filename.string());
  const auto restored = LatticeGraph::from_hdf5_file(filename.string());
  std::filesystem::remove(filename);

  ASSERT_EQ(restored.bond_flavor_definitions().size(), 1);
  EXPECT_EQ(restored.bond_flavor_definitions()[0].shell, shell);
  EXPECT_EQ(restored.bond_flavor_definitions()[0].flavor, flavor);
  EXPECT_EQ(restored.content_hash(), graph.content_hash());
}

TEST_F(LatticeGraphTest, BondFlavorJsonRejectsMalformedMetadata) {
  const auto valid = LatticeGraph::square(2, 2)
                         .with_bond_flavors({{1, {1.0, 0.0}, 1000}})
                         .to_json();
  const auto expect_invalid = [&](const nlohmann::json& definition) {
    auto malformed = valid;
    malformed["bond_flavor_definitions"] = nlohmann::json::array({definition});
    EXPECT_THROW(LatticeGraph::from_json(malformed), std::runtime_error);
  };

  expect_invalid({1.5, 1.0, 0.0, 0});
  expect_invalid({-1, 1.0, 0.0, 0});
  expect_invalid({1, 1.0, 0.0, 1.5});
  expect_invalid(
      {1, 1.0, 0.0,
       static_cast<std::uint64_t>(std::numeric_limits<BondFlavorId>::max()) +
           1});
  expect_invalid({1, 1.0, 0.0});
}

TEST_F(LatticeGraphTest, BondFlavorHdf5RejectsMalformedMetadata) {
  const auto graph =
      LatticeGraph::square(2, 2).with_bond_flavors({{1, {1.0, 0.0}, 1000}});
  const std::filesystem::path filename =
      "test_lattice_graph_invalid_bond_flavors.lattice_graph.h5";
  const auto expect_invalid = [&](const auto& mutate) {
    graph.to_hdf5_file(filename.string());
    {
      H5::H5File file(filename.string(), H5F_ACC_RDWR);
      auto definitions = file.openGroup("/bond_flavor_definitions");
      mutate(definitions);
    }
    EXPECT_THROW(LatticeGraph::from_hdf5_file(filename.string()),
                 std::runtime_error);
    std::filesystem::remove(filename);
  };

  expect_invalid([](H5::Group& definitions) { definitions.unlink("axes"); });
  expect_invalid([](H5::Group& definitions) {
    definitions.unlink("axes");
    hsize_t dims[1] = {2};
    auto dataset = definitions.createDataSet(
        "axes", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, dims));
    const double axes[2] = {1.0, 0.0};
    dataset.write(axes, H5::PredType::NATIVE_DOUBLE);
  });
  expect_invalid([](H5::Group& definitions) {
    definitions.unlink("shells");
    hsize_t dims[1] = {1};
    auto dataset = definitions.createDataSet(
        "shells", H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, dims));
    const double shell = 1.0;
    dataset.write(&shell, H5::PredType::NATIVE_DOUBLE);
  });
  expect_invalid([](H5::Group& definitions) {
    definitions.unlink("flavors");
    hsize_t dims[1] = {1};
    auto dataset = definitions.createDataSet(
        "flavors", H5::PredType::NATIVE_UINT64, H5::DataSpace(1, dims));
    const std::uint64_t flavor =
        static_cast<std::uint64_t>(std::numeric_limits<BondFlavorId>::max()) +
        1;
    dataset.write(&flavor, H5::PredType::NATIVE_UINT64);
  });
}

TEST_F(LatticeGraphTest, BoundaryConditionsAffectGeometricNeighborShells) {
  auto degrees = [](const LatticeGraph& graph, std::uint64_t site) {
    std::array<std::size_t, 3> result{};
    for (std::uint64_t m = 1; m <= result.size(); ++m) {
      const auto shell = graph.mth_nearest_neighbors(m);
      result[m - 1] =
          std::count_if(shell.begin(), shell.end(), [site](const auto& edge) {
            return edge.first == site || edge.second == site;
          });
    }
    return result;
  };

  EXPECT_EQ(degrees(LatticeGraph::square(5, 5), 0),
            (std::array<std::size_t, 3>{2, 1, 2}));
  EXPECT_EQ(degrees(LatticeGraph::square(5, 5, true, false), 0),
            (std::array<std::size_t, 3>{3, 2, 3}));
  EXPECT_EQ(degrees(LatticeGraph::square(5, 5, false, true), 0),
            (std::array<std::size_t, 3>{3, 2, 3}));
  EXPECT_EQ(degrees(LatticeGraph::square(5, 5, true, true), 0),
            (std::array<std::size_t, 3>{4, 4, 4}));

  EXPECT_EQ(degrees(LatticeGraph::honeycomb(4, 4), 0),
            (std::array<std::size_t, 3>{1, 2, 0}));
  EXPECT_EQ(degrees(LatticeGraph::honeycomb(4, 4, true, false), 0),
            (std::array<std::size_t, 3>{2, 4, 1}));
  EXPECT_EQ(degrees(LatticeGraph::honeycomb(4, 4, false, true), 0),
            (std::array<std::size_t, 3>{2, 4, 1}));
  EXPECT_EQ(degrees(LatticeGraph::honeycomb(4, 4, true, true), 0),
            (std::array<std::size_t, 3>{3, 6, 3}));
}

TEST_F(LatticeGraphTest, GeometricShellValidationAndPeriodicBoundaries) {
  Eigen::MatrixXd adjacency = Eigen::MatrixXd::Zero(3, 3);
  auto graph = LatticeGraph::from_dense_matrix(adjacency);
  EXPECT_THROW(graph.mth_nearest_neighbors(1), std::runtime_error);
  EXPECT_TRUE(graph.neighbor_connections({}).empty());
  EXPECT_THROW(graph.neighbor_connections({1}), std::runtime_error);

  auto periodic_square = LatticeGraph::square(4, 3, true, false);
  EXPECT_THROW(periodic_square.mth_nearest_neighbors(0), std::invalid_argument);
  EXPECT_THROW(periodic_square.mth_nearest_neighbors(1, 0.0),
               std::invalid_argument);
  EXPECT_THROW(periodic_square.nearest_neighbor_shells({1, 0}),
               std::invalid_argument);
  EXPECT_TRUE(periodic_square.mth_nearest_neighbors(99).empty());

  const auto shells = periodic_square.nearest_neighbor_shells({2, 1, 99});
  EXPECT_EQ(shells.at(1), periodic_square.mth_nearest_neighbors(1));
  EXPECT_EQ(shells.at(2), periodic_square.mth_nearest_neighbors(2));
  EXPECT_TRUE(shells.at(99).empty());
  const auto first = periodic_square.mth_nearest_neighbors(1);
  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  EXPECT_NE(std::find(first.begin(), first.end(), Edge{0, 3}), first.end());
}

TEST_F(LatticeGraphTest, GeometricShellsAreScaleInvariant) {
  auto make_geometry = [](double scale, bool periodic) {
    nlohmann::json graph = {
        {"num_sites", 3},
        {"is_symmetric", true},
        {"adjacency_sparse", nlohmann::json::array()},
        {"positions", {{0.0, 0.0}, {scale, 0.0}, {2.0 * scale, 0.0}}}};
    if (periodic) {
      graph["periods"] = {{3.0 * scale, 0.0}};
    }
    return LatticeGraph::from_json(graph);
  };

  using Edge = std::pair<std::uint64_t, std::uint64_t>;
  for (double scale : {1.0e-200, 1.0e200}) {
    auto open = make_geometry(scale, false);
    EXPECT_EQ(open.mth_nearest_neighbors(1),
              (std::vector<Edge>{{0, 1}, {1, 2}}));
    EXPECT_EQ(open.mth_nearest_neighbors(2), (std::vector<Edge>{{0, 2}}));

    auto periodic = make_geometry(scale, true);
    EXPECT_EQ(periodic.mth_nearest_neighbors(1),
              (std::vector<Edge>{{0, 1}, {0, 2}, {1, 2}}));
    const auto connections = periodic.neighbor_connections({1});
    EXPECT_EQ(connections.size(), 3);
    for (const auto& connection : connections) {
      EXPECT_NEAR(
          std::hypot(connection.displacement.x(), connection.displacement.y()) /
              scale,
          1.0, 1.0e-12);
    }
  }

  nlohmann::json graph = {{"num_sites", 1},
                          {"is_symmetric", true},
                          {"adjacency_sparse", nlohmann::json::array()},
                          {"positions", {{0.0, 0.0}}}};
  for (double scale : {1.0e-200, 1.0e200}) {
    graph["periods"] = {{scale, 0.0}, {0.0, scale}};
    EXPECT_NO_THROW(LatticeGraph::from_json(graph));
  }

  for (const auto& periods :
       std::vector<nlohmann::json>{{{1.0e-200, 0.0}, {0.0, 1.0e200}},
                                   {{1.0e-200, 0.0}, {1.0e200, 1.0e200}},
                                   {{1.0e-200, 0.0}, {5.0e-201, 1.0e-200}}}) {
    EXPECT_THROW(LatticeGraph::from_json(
                     {{"num_sites", 3},
                      {"is_symmetric", true},
                      {"adjacency_sparse", nlohmann::json::array()},
                      {"positions", {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}}},
                      {"periods", periods}}),
                 std::invalid_argument);
  }
}

TEST_F(LatticeGraphTest, ConnectionsScaleBeforeSubtractingLargeCoordinates) {
  const auto graph = LatticeGraph::from_json({
      {"num_sites", 2},
      {"is_symmetric", true},
      {"adjacency_sparse", nlohmann::json::array()},
      {"positions", {{-9.0e307, 0.0}, {9.0e307, 0.0}}},
      {"periods", {{1.5e308, 0.0}}},
  });

  EXPECT_EQ(graph.mth_nearest_neighbors(1),
            (std::vector<std::pair<std::uint64_t, std::uint64_t>>{{0, 1}}));
  const auto connections = graph.neighbor_connections({1});
  ASSERT_EQ(connections.size(), 1);
  EXPECT_NEAR(connections[0].displacement.x() / 3.0e307, 1.0, 1.0e-12);
  EXPECT_EQ(connections[0].image_shift[0], -1);
}

TEST_F(LatticeGraphTest, SkewPeriodicDistancesSearchAllRelevantImages) {
  auto triangular = LatticeGraph::triangular(2, 12, true, false);
  auto triangular_shell = triangular.mth_nearest_neighbors(16);
  EXPECT_NE(std::find(triangular_shell.begin(), triangular_shell.end(),
                      std::pair<std::uint64_t, std::uint64_t>{0, 22}),
            triangular_shell.end());

  auto honeycomb = LatticeGraph::honeycomb(2, 12, true, false);
  auto honeycomb_shell = honeycomb.mth_nearest_neighbors(46);
  EXPECT_NE(std::find(honeycomb_shell.begin(), honeycomb_shell.end(),
                      std::pair<std::uint64_t, std::uint64_t>{0, 44}),
            honeycomb_shell.end());

  auto kagome = LatticeGraph::kagome(2, 12, true, false);
  auto kagome_shell = kagome.mth_nearest_neighbors(52);
  EXPECT_NE(std::find(kagome_shell.begin(), kagome_shell.end(),
                      std::pair<std::uint64_t, std::uint64_t>{0, 66}),
            kagome_shell.end());
}

TEST_F(LatticeGraphTest, GeometrySurvivesSerializationAndPermutation) {
  auto square = LatticeGraph::square(3, 2, true, false);
  const auto json = square.to_json();
  EXPECT_TRUE(json.contains("positions"));
  EXPECT_TRUE(json.contains("periods"));
  EXPECT_FALSE(json.contains("distance_matrix"));
  auto restored = LatticeGraph::from_json(json);
  EXPECT_EQ(restored.mth_nearest_neighbors(2), square.mth_nearest_neighbors(2));

  const std::filesystem::path filename =
      "test_lattice_graph_geometry.lattice_graph.h5";
  square.to_hdf5_file(filename.string());
  auto hdf5_restored = LatticeGraph::from_hdf5_file(filename.string());
  EXPECT_EQ(hdf5_restored.mth_nearest_neighbors(2),
            square.mth_nearest_neighbors(2));
  std::filesystem::remove(filename);

  std::vector<std::uint64_t> path = {1, 2, 0, 4, 5, 3};
  auto permuted = LatticeGraph::permute(square, path);
  std::vector<std::uint64_t> inverse_path(path.size());
  for (std::uint64_t i = 0; i < path.size(); ++i) {
    inverse_path[path[i]] = i;
  }
  for (std::uint64_t m = 1; m <= 3; ++m) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> expected;
    for (const auto& [i, j] : square.mth_nearest_neighbors(m)) {
      auto edge = std::minmax(inverse_path[i], inverse_path[j]);
      expected.emplace_back(edge.first, edge.second);
    }
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(permuted.mth_nearest_neighbors(m), expected);
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
