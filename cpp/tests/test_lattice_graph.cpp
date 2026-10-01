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
// Dynamic-size record vectors need an explicit type for two-component literals.
using Vec2 = Eigen::RowVector2d;
constexpr BondFlavorId flavor_x = 10;
constexpr BondFlavorId flavor_y = 20;
constexpr BondFlavorId flavor_z = 30;

std::vector<BondFlavorDefinition> honeycomb_flavor_ids() {
  const double root_three = std::sqrt(3.0);
  return {
      {1, Vec2(0.5, root_three / 2.0), flavor_x},
      {1, Vec2(0.5, -root_three / 2.0), flavor_y},
      {1, Vec2(1.0, 0.0), flavor_z},
      {2, Vec2(1.5, -root_three / 2.0), flavor_x},
      {2, Vec2(1.5, root_three / 2.0), flavor_y},
      {2, Vec2(0.0, root_three), flavor_z},
      {3, Vec2(1.0, root_three), flavor_x},
      {3, Vec2(1.0, -root_three), flavor_y},
      {3, Vec2(2.0, 0.0), flavor_z},
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

TEST_F(LatticeGraphTest, HoneycombFlavorsResolveSelectedShells) {
  const auto honeycomb =
      LatticeGraph::from_geometry(LatticeGeometry::honeycomb(5, 5), {1, 2, 3},
                                  honeycomb_flavor_ids(), 2.5, 1.0e-9);
  constexpr std::uint64_t center = 24;
  std::map<std::uint64_t, std::map<BondFlavorId, std::size_t>> flavor_degree;
  for (const auto& [edge, label] : honeycomb.edge_labels()) {
    ASSERT_TRUE(label.flavor.has_value());
    EXPECT_DOUBLE_EQ(honeycomb.weight(edge.first, edge.second), 2.5);
    if (edge.first == center || edge.second == center) {
      ++flavor_degree[label.shell][*label.flavor];
    }
  }
  for (BondFlavorId flavor : {flavor_x, flavor_y, flavor_z}) {
    EXPECT_EQ(flavor_degree.at(1).at(flavor), 1);
    EXPECT_EQ(flavor_degree.at(2).at(flavor), 2);
    EXPECT_EQ(flavor_degree.at(3).at(flavor), 1);
  }
}

TEST_F(LatticeGraphTest, FactoriesHaveUnlabelledEdges) {
  for (const auto& graph :
       {LatticeGraph::chain(5), LatticeGraph::square(3, 3),
        LatticeGraph::triangular(3, 3), LatticeGraph::honeycomb(3, 3),
        LatticeGraph::kagome(3, 3)}) {
    EXPECT_GT(graph.num_edges(), 0);
    EXPECT_TRUE(graph.edge_labels().empty());
    EXPECT_FALSE(graph.to_json().contains("edge_labels"));
  }
}

TEST_F(LatticeGraphTest, HoneycombOpenPlaquettePatches) {
  auto unit_cell = LatticeGraph::honeycomb(1, 1);
  EXPECT_EQ(unit_cell.num_sites(), 2);
  EXPECT_EQ(unit_cell.num_edges(), 1);

  const auto geometry = LatticeGeometry::honeycomb_plaquettes(1, 1);
  const auto hexagon = LatticeGraph::from_geometry(geometry, {1}, {}, 2.5);
  EXPECT_EQ(hexagon.num_sites(), 6);
  EXPECT_EQ(hexagon.num_edges(), 6);
  EXPECT_TRUE(hexagon.is_symmetric());
  for (Eigen::Index site = 0;
       site < hexagon.sparse_adjacency_matrix().outerSize(); ++site) {
    EXPECT_EQ(hexagon.sparse_adjacency_matrix().innerVector(site).nonZeros(),
              2);
  }

  const auto flavored_hexagon = LatticeGraph::from_geometry(
      geometry, {1, 2, 3}, honeycomb_flavor_ids(), 2.5, 1.0e-9);
  std::map<std::uint64_t, std::map<BondFlavorId, std::size_t>> counts;
  for (const auto& [edge, label] : flavored_hexagon.edge_labels()) {
    ASSERT_TRUE(label.flavor.has_value());
    EXPECT_DOUBLE_EQ(flavored_hexagon.weight(edge.first, edge.second), 2.5);
    ++counts[label.shell][*label.flavor];
  }
  for (BondFlavorId flavor : {flavor_x, flavor_y, flavor_z}) {
    EXPECT_EQ(counts.at(1).at(flavor), 2);
    EXPECT_EQ(counts.at(2).at(flavor), 2);
    EXPECT_EQ(counts.at(3).at(flavor), 1);
  }
  const auto patch =
      LatticeGraph::from_geometry(LatticeGeometry::honeycomb_plaquettes(4, 4));
  EXPECT_EQ(patch.num_sites(), 48);
  EXPECT_EQ(patch.num_edges(), 63);
  EXPECT_EQ(patch.num_edges() - patch.num_sites() + 1, 16);
  for (Eigen::Index site = 0;
       site < patch.sparse_adjacency_matrix().outerSize(); ++site) {
    EXPECT_GE(patch.sparse_adjacency_matrix().innerVector(site).nonZeros(), 2);
  }

  // Periodic directions add no boundary cell, matching the unit-cell factory.
  EXPECT_TRUE(
      LatticeGraph::from_geometry(
          LatticeGeometry::honeycomb_plaquettes(2, 2, true, true))
          .adjacency_matrix()
          .isApprox(
              LatticeGraph::honeycomb(2, 2, true, true).adjacency_matrix()));
}

TEST_F(LatticeGraphTest, FromGeometryRejectsRepeatedPeriodicImages) {
  // Both images of a two-site ring join the same pair.
  EXPECT_THROW(LatticeGraph::from_geometry(LatticeGeometry::chain(2, true)),
               std::invalid_argument);
  EXPECT_THROW(LatticeGraph::from_geometry(LatticeGeometry::chain(1, true)),
               std::invalid_argument);
  const auto ring = LatticeGraph::from_geometry(LatticeGeometry::chain(3, true),
                                                {1}, {}, 1.5);
  EXPECT_EQ(ring.num_edges(), 3);
  EXPECT_DOUBLE_EQ(ring.weight(0, 2), 1.5);
}

TEST_F(LatticeGraphTest, EdgeLabelsSurviveDataOperations) {
  const auto graph = LatticeGraph::from_geometry(
      LatticeGeometry::square(3, 2), {1, 2, 7},
      {{1, Vec2(1.0, 0.0), flavor_x}, {2, Vec2(1.0, 1.0), flavor_y}}, 2.5);
  const auto& labels = graph.edge_labels();
  EXPECT_EQ(labels.size(), graph.num_edges());
  EXPECT_EQ(labels.at({0, 1}), (EdgeLabel{1, flavor_x}));
  EXPECT_EQ(labels.at({0, 3}), (EdgeLabel{1, std::nullopt}));
  EXPECT_EQ(labels.at({0, 4}), (EdgeLabel{2, flavor_y}));
  EXPECT_EQ(labels.at({1, 3}), (EdgeLabel{2, std::nullopt}));

  const auto json = graph.to_json();
  EXPECT_EQ(json.at("edge_labels").front(),
            nlohmann::json({0, 1, 1, flavor_x}));
  const std::string filename = "test_edge_labels.lattice_graph.h5";
  graph.to_hdf5_file(filename);
  const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
  std::filesystem::remove(filename);
  for (const auto& restored :
       {LatticeGraph::from_json(nlohmann::json::parse(json.dump())), hdf5}) {
    EXPECT_EQ(restored.edge_labels(), labels);
    EXPECT_EQ(restored.to_json(), json);
    EXPECT_EQ(restored.content_hash(), graph.content_hash());
  }

  const std::vector<std::uint64_t> path = {1, 2, 0, 4, 5, 3};
  const auto permuted = LatticeGraph::permute(graph, path);
  for (std::uint64_t i = 0; i < path.size(); ++i) {
    for (std::uint64_t j = i + 1; j < path.size(); ++j) {
      const auto label = labels.find(std::minmax(path[i], path[j]));
      ASSERT_EQ(permuted.edge_labels().contains({i, j}), label != labels.end());
      if (label != labels.end()) {
        EXPECT_EQ(permuted.edge_labels().at({i, j}), label->second);
      }
    }
  }
  const auto doubled = LatticeGraph::make_bidirectional(graph);
  EXPECT_EQ(doubled.edge_labels(), labels);
  EXPECT_TRUE(
      doubled.adjacency_matrix().isApprox(2.0 * graph.adjacency_matrix()));
}

TEST_F(LatticeGraphTest, BondFlavorAxesAreScaleInvariant) {
  const auto geometry = LatticeGeometry::square(2, 2);
  const auto expected =
      LatticeGraph::from_geometry(geometry, {1}, {{1, Vec2(1.0, 0.0), 1000}});
  for (double scale : {1.0e-200, 1.0e200, -1.0e-200, -1.0e200}) {
    const auto flavored = LatticeGraph::from_geometry(
        geometry, {1}, {{1, Vec2(scale, 0.0), 1000}});
    EXPECT_EQ(flavored.edge_labels(), expected.edge_labels());
    EXPECT_EQ(std::count_if(
                  flavored.edge_labels().begin(), flavored.edge_labels().end(),
                  [](const auto& item) { return item.second.flavor == 1000; }),
              2);
  }
}

TEST_F(LatticeGraphTest, OppositeFlavorAxesMatchAboveHalfRootTwoTolerance) {
  // Above 1/sqrt(2), both diagonal components lie within the tolerance of 0.
  auto json = LatticeGeometry::square(3, 3).to_json();
  json["integer_embedding"]["primitive_vectors"] = {{1.0, 1.0}, {1.0, -1.0}};
  const auto geometry = LatticeGeometry::from_json(json);
  const auto flavored = [&](const Vec2& axis) {
    return LatticeGraph::from_geometry(geometry, {1}, {{1, axis, flavor_x}},
                                       1.0, 0.72)
        .edge_labels();
  };
  for (const auto& axis : {Vec2(1.0, -1.0), Vec2(1.0, 1.0)}) {
    const auto labels = flavored(axis);
    EXPECT_EQ(flavored(-axis), labels);
    EXPECT_TRUE(std::any_of(labels.begin(), labels.end(), [](const auto& item) {
      return item.second.flavor == flavor_x;
    }));
  }
}

TEST_F(LatticeGraphTest, NearbyFlavorAxesMatchAcrossSignFlipPoint) {
  // A per-axis sign rule at tolerance 0.3 flips the bond axis but not the
  // flavor axis, although they lie about 0.02 apart.
  auto json = LatticeGeometry::square(3, 3).to_json();
  json["integer_embedding"]["primitive_vectors"] = {{0.29, -0.957},
                                                    {0.957, 0.29}};
  const auto labels =
      LatticeGraph::from_geometry(LatticeGeometry::from_json(json), {1},
                                  {{1, Vec2(0.31, -0.95), flavor_x}}, 1.0, 0.3)
          .edge_labels();
  EXPECT_TRUE(std::any_of(labels.begin(), labels.end(), [](const auto& item) {
    return item.second.flavor == flavor_x;
  }));
}

TEST_F(LatticeGraphTest, BondAxisMatchingSeveralFlavorsIsRejected) {
  // At tolerance 0.8 the diagonals join shell 1, about 0.77 from both axes.
  const auto geometry = LatticeGeometry::square(3, 3);
  const std::vector<BondFlavorDefinition> definitions = {
      {1, Vec2(1.0, 0.0), flavor_x}, {1, Vec2(0.0, 1.0), flavor_y}};
  EXPECT_THROW(
      LatticeGraph::from_geometry(geometry, {1}, definitions, 1.0, 0.8),
      std::invalid_argument);
  const auto labels =
      LatticeGraph::from_geometry(geometry, {1}, definitions, 1.0, 0.7)
          .edge_labels();
  EXPECT_FALSE(labels.at({0, 4}).flavor.has_value());
}

TEST_F(LatticeGraphTest, JsonRejectsMalformedEdgeLabels) {
  const auto valid =
      LatticeGraph::from_geometry(LatticeGeometry::chain(3)).to_json();
  const auto out_of_range =
      static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
  const std::vector<std::pair<std::string, nlohmann::json>> invalid = {
      {"/edge_labels/0/1", 5},
      {"/edge_labels/1/1", 0},
      {"/edge_labels/0/2", 1.5},
      {"/edge_labels/0/2", 0},
      {"/edge_labels/0/3", -1},
      {"/edge_labels/0/3", out_of_range},
      {"/edge_labels/0", {0, 1, 1}},
      {"/edge_labels/1", {0, 1, 1, nullptr}},
      {"/edge_labels", {{"shell", 1}}},
      {"/adjacency_sparse", {{0, 2, 1.0}, {2, 0, 1.0}}}};
  for (const auto& [path, value] : invalid) {
    SCOPED_TRACE(path);
    auto malformed = valid;
    malformed[nlohmann::json::json_pointer(path)] = value;
    EXPECT_THROW(LatticeGraph::from_json(malformed), std::invalid_argument);
  }
}

TEST_F(LatticeGraphTest, Hdf5RejectsMalformedEdgeLabels) {
  const auto graph = LatticeGraph::from_geometry(LatticeGeometry::chain(2));
  const std::string filename = "test_invalid_labels.lattice_graph.h5";
  const auto replace = [&](const std::string& name, const auto& type,
                           std::vector<hsize_t> dims, const auto* values) {
    graph.to_hdf5_file(filename);
    {
      H5::H5File file(filename, H5F_ACC_RDWR);
      auto root = file.openGroup("/");
      root.unlink(name);
      root.createDataSet(
              name, type,
              H5::DataSpace(static_cast<int>(dims.size()), dims.data()))
          .write(values, type);
    }
    EXPECT_THROW(LatticeGraph::from_hdf5_file(filename), std::invalid_argument);
    std::filesystem::remove(filename);
  };
  const double narrow[3] = {0.0, 1.0, 1.0};
  replace("edge_labels", H5::PredType::NATIVE_DOUBLE, {1, 3}, narrow);
  for (const double flavor : {-2.0, 0.5, 4294967296.0}) {
    const double label[4] = {0.0, 1.0, 1.0, flavor};
    replace("edge_labels", H5::PredType::NATIVE_DOUBLE, {1, 4}, label);
  }
  const double duplicate[8] = {0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, -1.0};
  replace("edge_labels", H5::PredType::NATIVE_DOUBLE, {2, 4}, duplicate);
}

TEST_F(LatticeGraphTest, SerializationRequiresCompatibleVersion) {
  const auto graph = LatticeGraph::chain(3);
  const auto json = graph.to_json();
  ASSERT_TRUE(json.at("version").is_string());
  // Unversioned files predate graph versioning and must be migrated first.
  auto unversioned = json;
  unversioned.erase("version");
  auto incompatible = json;
  incompatible["version"] = "99.0.0";
  for (const auto& invalid : {unversioned, incompatible}) {
    EXPECT_THROW(LatticeGraph::from_json(invalid), std::runtime_error);
  }

  const std::string filename = "test_version.lattice_graph.h5";
  for (const std::string version : {"", "99.0.0"}) {
    SCOPED_TRACE(version);
    graph.to_hdf5_file(filename);
    {
      H5::H5File file(filename, H5F_ACC_RDWR);
      auto root = file.openGroup("/");
      root.removeAttr("version");
      if (!version.empty()) {
        const H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
        root.createAttribute("version", string_type, H5::DataSpace(H5S_SCALAR))
            .write(string_type, version);
      }
    }
    EXPECT_THROW(LatticeGraph::from_hdf5_file(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }
}

TEST_F(LatticeGraphTest, FromGeometrySelectsShellsAndWeights) {
  const auto geometry = LatticeGeometry::chain(4);
  const auto graph = LatticeGraph::from_geometry(
      geometry, {3, 1, 5, 3},
      {{1, Vec2(1.0, 0.0), flavor_x}, {2, Vec2(1.0, 0.0), flavor_y}}, -2.5,
      1.0e-9);
  ASSERT_EQ(graph.edge_labels().size(), 4);
  for (const auto& [edge, label] : graph.edge_labels()) {
    EXPECT_DOUBLE_EQ(graph.weight(edge.first, edge.second), -2.5);
    EXPECT_EQ(label.flavor, label.shell == 1
                                ? std::optional<BondFlavorId>(flavor_x)
                                : std::nullopt);
  }
  EXPECT_DOUBLE_EQ(graph.weight(0, 3), -2.5);
  EXPECT_FALSE(graph.are_connected(0, 2));
}

TEST_F(LatticeGraphTest, CustomGraphsAcceptEdgeLabels) {
  // A square plaquette with one second-shell diagonal and one unflavored edge.
  const EdgeLabels labels = {{{0, 1}, {1, flavor_x}},
                             {{1, 2}, {1, flavor_y}},
                             {{2, 3}, {1, flavor_x}},
                             {{0, 3}, {1, std::nullopt}},
                             {{0, 2}, {2, flavor_z}}};
  std::map<Edge, double> upper;
  for (const auto& [edge, label] : labels) upper[edge] = -1.5;
  const auto graph =
      LatticeGraph::make_bidirectional(LatticeGraph(upper, 0, labels));
  EXPECT_EQ(graph.edge_labels(), labels);
  EXPECT_FALSE(graph.edge_coloring().has_value());
  EXPECT_EQ(LatticeGraph::from_dense_matrix(graph.adjacency_matrix(), labels)
                .edge_labels(),
            labels);
  EXPECT_EQ(
      LatticeGraph::from_sparse_matrix(graph.sparse_adjacency_matrix(), labels)
          .content_hash(),
      graph.content_hash());
  const auto restored = LatticeGraph::from_json(graph.to_json());
  EXPECT_EQ(restored.edge_labels(), labels);
  EXPECT_EQ(restored.content_hash(), graph.content_hash());

  // Labels must cover exactly the stored pairs, each with a shell in [1, 2^53].
  auto missing = labels;
  missing.erase({0, 2});
  auto extra = labels;
  extra[{1, 3}] = {2, std::nullopt};
  auto zero_shell = labels;
  zero_shell.at({0, 1}).shell = 0;
  auto huge_shell = labels;
  huge_shell.at({0, 1}).shell = (std::uint64_t{1} << 53) + 1;
  for (const auto& invalid : {missing, extra, zero_shell, huge_shell}) {
    EXPECT_THROW((LatticeGraph(upper, 0, invalid)), std::invalid_argument);
    EXPECT_THROW(
        LatticeGraph::from_dense_matrix(graph.adjacency_matrix(), invalid),
        std::invalid_argument);
    EXPECT_THROW(LatticeGraph::from_sparse_matrix(
                     graph.sparse_adjacency_matrix(), invalid),
                 std::invalid_argument);
  }
  auto largest = labels;
  largest.at({0, 1}).shell = std::uint64_t{1} << 53;
  const LatticeGraph exact(upper, 0, largest);
  const std::string filename = "test_largest_shell.lattice_graph.h5";
  exact.to_hdf5_file(filename);
  const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
  std::filesystem::remove(filename);
  EXPECT_EQ(hdf5.edge_labels(), largest);
  EXPECT_EQ(LatticeGraph::from_json(exact.to_json()).edge_labels(), largest);
}

TEST_F(LatticeGraphTest, LabelsFollowEdgesStoredInEitherDirection) {
  // Swapping the endpoints of a directed labelled edge keeps its label.
  const LatticeGraph directed({{{0, 1}, 1.0}}, 2, {{{0, 1}, {2, flavor_x}}});
  const auto swapped = LatticeGraph::permute(directed, {1, 0});
  EXPECT_DOUBLE_EQ(swapped.weight(1, 0), 1.0);
  EXPECT_EQ(swapped.edge_labels(), directed.edge_labels());
  const auto restored = LatticeGraph::from_json(swapped.to_json());
  EXPECT_EQ(restored.content_hash(), swapped.content_hash());

  // Pair (1, 2) is stored only as (2, 1) and is still keyed with i < j.
  const EdgeLabels labels = {{{0, 1}, {1, std::nullopt}},
                             {{1, 2}, {2, flavor_y}}};
  const LatticeGraph mixed({{{0, 1}, 1.0}, {{2, 1}, 1.0}}, 3, labels);
  EXPECT_EQ(LatticeGraph::from_json(mixed.to_json()).edge_labels(), labels);
  const auto bidirectional = LatticeGraph::make_bidirectional(mixed);
  EXPECT_TRUE(bidirectional.is_symmetric());
  EXPECT_EQ(bidirectional.edge_labels(), labels);
  EXPECT_THROW(LatticeGraph({{{0, 1}, 1.0}, {{2, 1}, 1.0}}, 3,
                            {{{0, 1}, {1, std::nullopt}}}),
               std::invalid_argument);
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
  for (double tolerance :
       {0.0, -1.0, 1.0, 2.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(LatticeGraph::from_geometry(geometry, {1}, {}, 1.0, tolerance),
                 std::invalid_argument);
  }
  // Axis normalization uses its own bound, but still rejects subnormal axes.
  EXPECT_EQ(
      LatticeGraph::from_geometry(LatticeGeometry::square(2, 2), {2},
                                  {{2, Vec2(1.0, 1.0), flavor_x}}, 1.0, 1.0e-16)
          .edge_labels()
          .at({0, 3})
          .flavor,
      flavor_x);
  const double subnormal = std::numeric_limits<double>::denorm_min();
  EXPECT_THROW(LatticeGraph::from_geometry(
                   geometry, {1}, {{1, Vec2(subnormal, subnormal), flavor_x}}),
               std::invalid_argument);
  EXPECT_THROW(LatticeGraph::from_geometry(geometry, {1},
                                           {{0, Vec2(1.0, 0.0), flavor_x}}),
               std::invalid_argument);
  EXPECT_THROW(LatticeGraph::from_geometry(geometry, {1},
                                           {{1, Vec2(0.0, 0.0), flavor_x}}),
               std::invalid_argument);
  EXPECT_THROW(LatticeGraph::from_geometry(geometry, {1},
                                           {{1, Vec2(1.0, 0.0), flavor_x},
                                            {1, Vec2(-2.0, 0.0), flavor_y}}),
               std::invalid_argument);
  EXPECT_THROW(
      LatticeGraph::from_geometry(
          geometry, {1}, {{1, Eigen::RowVector3d(1.0, 0.0, 0.0), flavor_x}}),
      std::invalid_argument);
}

TEST_F(LatticeGraphTest, EdgelessGraphsRoundTrip) {
  const auto absent =
      LatticeGraph::from_dense_matrix(Eigen::MatrixXd::Zero(1, 1));
  const auto finite =
      LatticeGraph::from_geometry(LatticeGeometry::chain(1), {9, 1, 9});
  const std::string filename = "test_empty_labels.lattice_graph.h5";
  for (const auto& graph : {absent, finite}) {
    EXPECT_TRUE(graph.edge_labels().empty());
    EXPECT_FALSE(graph.to_json().contains("edge_labels"));
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
    const auto graph = LatticeGraph::chain(2, true, weight);
    EXPECT_DOUBLE_EQ(graph.weight(0, 1), weight);
    graph.to_hdf5_file(filename);
    const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
    std::filesystem::remove(filename);
    for (const auto& restored :
         {LatticeGraph::from_json(graph.to_json()), hdf5}) {
      EXPECT_DOUBLE_EQ(restored.weight(0, 1), weight);
      EXPECT_EQ(restored.edge_coloring(), graph.edge_coloring());
      EXPECT_EQ(restored.num_nonzeros(), graph.num_nonzeros());
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
    EXPECT_TRUE(source.edge_labels().empty());
    EXPECT_FALSE(source.to_json().contains("edge_labels"));
    EXPECT_TRUE(source.adjacency_matrix().isApprox(adjacency));
    source.to_hdf5_file(filename);
    const auto hdf5 = LatticeGraph::from_hdf5_file(filename);
    std::filesystem::remove(filename);
    for (const auto& restored :
         {LatticeGraph::from_json(source.to_json()), hdf5}) {
      EXPECT_TRUE(restored.adjacency_matrix().isApprox(adjacency));
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

TEST_F(LatticeGraphTest, PermutedDirectedColoringStaysValid) {
  auto json = LatticeGraph(std::map<Edge, double>{{{0, 1}, 1.0}}, 2).to_json();
  json["edge_coloring"] = {{0, 1, 0}};
  const auto graph = LatticeGraph::from_json(json);
  const auto permuted = LatticeGraph::permute(graph, {1, 0});
  EXPECT_DOUBLE_EQ(permuted.weight(1, 0), 1.0);
  EXPECT_EQ(permuted.edge_coloring(), graph.edge_coloring());
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

TEST_F(LatticeGraphTest, FromGeometryStoresTopologyColoring) {
  const auto geometry = LatticeGeometry::square(3, 3);
  const auto unit = LatticeGraph::from_geometry(geometry, {1, 2});
  const auto expected =
      greedy_edge_coloring(unit.sparse_adjacency_matrix(), 0, 32);
  for (double weight : {1.0, -2.5, 0.0}) {
    const auto graph =
        LatticeGraph::from_geometry(geometry, {1, 2}, {}, weight);
    ASSERT_TRUE(graph.edge_coloring());
    EXPECT_EQ(*graph.edge_coloring(), expected);
    check_valid_edge_coloring(*graph.edge_coloring());
  }
  const auto empty = LatticeGraph::from_geometry(geometry, {});
  ASSERT_TRUE(empty.edge_coloring());
  EXPECT_TRUE(empty.edge_coloring()->empty());
  for (int seed : {0, 5}) {
    const auto seeded =
        LatticeGraph::from_geometry(geometry, {1, 2}, {}, 1.0, 1.0e-9, seed);
    EXPECT_EQ(*seeded.edge_coloring(),
              greedy_edge_coloring(unit.sparse_adjacency_matrix(), seed, 32));
  }
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
