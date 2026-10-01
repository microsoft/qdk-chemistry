// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <string>
#include <utility>
#include <vector>

using namespace qdk::chemistry::data;

class LatticeGeometryTest : public ::testing::Test {};

namespace {
using Edge = std::pair<std::uint64_t, std::uint64_t>;
using ShellPairs = std::map<std::uint64_t, std::vector<Edge>>;

const std::vector<std::pair<bool, bool>> boundary_modes = {
    {false, false}, {true, false}, {false, true}, {true, true}};

ShellPairs shell_pairs(const LatticeGeometry& geometry,
                       const std::vector<std::uint64_t>& shells) {
  ShellPairs result;
  for (std::uint64_t shell : shells) result[shell];
  const auto graph = LatticeGraph::from_geometry(geometry, shells);
  for (const auto& [edge, label] : graph.edge_labels()) {
    result[label.shell].push_back(edge);
  }
  return result;
}

auto degree(const std::vector<Edge>& pairs, std::uint64_t site) {
  return std::count_if(pairs.begin(), pairs.end(), [site](const auto& edge) {
    return edge.first == site || edge.second == site;
  });
}

// Oracle for the stencil search: rank every site-to-image distance directly.
// Two images per periodic direction reach beyond shell three on these cells.
std::map<Edge, std::uint64_t> brute_force_shells(
    const LatticeGeometry& geometry, std::uint64_t max_shell) {
  const auto& positions = geometry.positions();
  const Eigen::MatrixXd periods =
      geometry.periods().value_or(Eigen::MatrixXd(0, 2));
  const int range_0 = periods.rows() >= 1 ? 2 : 0;
  const int range_1 = periods.rows() == 2 ? 2 : 0;
  std::vector<std::pair<double, Edge>> candidates;
  for (Eigen::Index i = 0; i < positions.rows(); ++i) {
    for (Eigen::Index j = i; j < positions.rows(); ++j) {
      for (int a = -range_0; a <= range_0; ++a) {
        for (int b = -range_1; b <= range_1; ++b) {
          Eigen::RowVector2d displacement = positions.row(j) - positions.row(i);
          if (range_0 != 0) displacement += a * periods.row(0);
          if (range_1 != 0) displacement += b * periods.row(1);
          const double distance = displacement.norm();
          if (distance > 0.0) candidates.emplace_back(distance, Edge(i, j));
        }
      }
    }
  }
  std::sort(candidates.begin(), candidates.end());
  std::map<Edge, std::uint64_t> result;
  std::uint64_t shell = 0;
  double shell_distance = 0.0;
  for (const auto& [distance, edge] : candidates) {
    if (shell == 0 || distance - shell_distance > 1.0e-9 * distance) {
      if (++shell > max_shell) break;
      shell_distance = distance;
    }
    result.emplace(edge, shell);
  }
  return result;
}
}  // namespace

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
  const auto shells = shell_pairs(LatticeGeometry::square(5, 5), {1, 2, 3});
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
  const auto shells = shell_pairs(LatticeGeometry::honeycomb(4, 4), {1, 2, 3});
  // A(1,1), site 10, has all of its first three shells inside this patch.
  EXPECT_EQ(degree(shells.at(1), 10), 3);
  EXPECT_EQ(degree(shells.at(2), 10), 6);
  EXPECT_EQ(degree(shells.at(3), 10), 3);
}

TEST_F(LatticeGeometryTest, FiniteNarrowShellsUsePresentDistances) {
  const auto distant_shell = std::numeric_limits<std::uint64_t>::max();
  const ShellPairs expected = {
      {1, {{0, 1}, {1, 2}}}, {2, {{0, 2}}}, {3, {}}, {distant_shell, {}}};
  for (const auto& geometry :
       {LatticeGeometry::chain(3), LatticeGeometry::square(1, 3),
        LatticeGeometry::triangular(1, 3)}) {
    EXPECT_EQ(shell_pairs(geometry, {3, 1, 2, 1, distant_shell}), expected);
  }

  // These four-site strips have no distance-2 bond: sqrt(7) is shell three.
  for (const auto& geometry :
       {LatticeGeometry::honeycomb(1, 2), LatticeGeometry::honeycomb(2, 1)}) {
    const auto shells = shell_pairs(geometry, {3, 4});
    EXPECT_EQ(shells.at(3), (std::vector<Edge>{{0, 3}}));
    EXPECT_NEAR(
        (geometry.positions().row(3) - geometry.positions().row(0)).norm(),
        std::sqrt(7.0), 1.0e-12);
    EXPECT_TRUE(shells.at(4).empty());
  }
  EXPECT_EQ(shell_pairs(LatticeGeometry::honeycomb(1, 1), {1, 2, 3}),
            (ShellPairs{{1, {{0, 1}}}, {2, {}}, {3, {}}}));
  const auto kagome = shell_pairs(LatticeGeometry::kagome(1, 1), {1, 2});
  EXPECT_EQ(kagome.at(1).size(), 3);
  EXPECT_TRUE(kagome.at(2).empty());
}

TEST_F(LatticeGeometryTest, HoneycombPlaquetteShellDistances) {
  const auto hexagon = LatticeGeometry::honeycomb_plaquettes(1, 1);
  ASSERT_EQ(hexagon.num_sites(), 6);
  const auto shells = shell_pairs(hexagon, {1, 2, 3});
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

TEST_F(LatticeGeometryTest, BuiltInShellsScaleToLargeLattices) {
  // An open N x N plaquette patch has 3N^2 + 4N - 1, 6N^2 + 4N - 4, and 3N^2
  // bonds in its first three shells.
  const auto geometry = LatticeGeometry::honeycomb_plaquettes(60, 60);
  const auto shells = shell_pairs(geometry, {1, 2, 3});
  EXPECT_EQ(geometry.num_sites(), 7440);
  EXPECT_EQ(shells.at(1).size(), 11039);
  EXPECT_EQ(shells.at(2).size(), 21836);
  EXPECT_EQ(shells.at(3).size(), 10800);
}

TEST_F(LatticeGeometryTest, ShellsMatchBruteForceDistanceRanking) {
  std::vector<std::pair<std::string, LatticeGeometry>> geometries = {
      {"chain", LatticeGeometry::chain(5)},
      {"ring", LatticeGeometry::chain(8, true)},
      {"square strip", LatticeGeometry::square(1, 5)},
      {"honeycomb strip", LatticeGeometry::honeycomb(1, 5)}};
  for (const auto& [px, py] : boundary_modes) {
    const std::string modes = std::to_string(px) + std::to_string(py);
    geometries.emplace_back("square" + modes,
                            LatticeGeometry::square(6, 6, px, py));
    geometries.emplace_back("triangular" + modes,
                            LatticeGeometry::triangular(6, 6, px, py));
    geometries.emplace_back("honeycomb" + modes,
                            LatticeGeometry::honeycomb(3, 3, px, py));
    geometries.emplace_back(
        "plaquettes" + modes,
        LatticeGeometry::honeycomb_plaquettes(3, 3, px, py));
    geometries.emplace_back("kagome" + modes,
                            LatticeGeometry::kagome(3, 3, px, py));
  }
  for (const auto& [name, geometry] : geometries) {
    SCOPED_TRACE(name);
    const auto graph = LatticeGraph::from_geometry(geometry, {1, 2, 3});
    std::map<Edge, std::uint64_t> actual;
    for (const auto& [edge, label] : graph.edge_labels()) {
      actual.emplace(edge, label.shell);
    }
    EXPECT_EQ(actual, brute_force_shells(geometry, 3));
  }
}

TEST_F(LatticeGeometryTest, PeriodicShellsDoNotDependOnPrimitiveVectors) {
  // a2 = (3, 1) spans the same 6 x 6 torus, placing cell (x, y) at square site
  // ((x + 3y) mod 6, y), so its nearest neighbors are three cells away.
  const auto square = LatticeGeometry::square(6, 6, true, true);
  auto json = square.to_json();
  json["integer_embedding"]["primitive_vectors"] = {{1.0, 0.0}, {3.0, 1.0}};
  const auto skewed = LatticeGeometry::from_json(json);
  const auto to_square = [](std::uint64_t site) {
    return (site % 6 + 3 * (site / 6)) % 6 + 6 * (site / 6);
  };
  const auto graph = LatticeGraph::from_geometry(skewed, {1, 2, 3});
  EdgeLabels mapped;
  for (const auto& [edge, label] : graph.edge_labels()) {
    const auto i = to_square(edge.first);
    const auto j = to_square(edge.second);
    mapped.emplace(Edge{std::min(i, j), std::max(i, j)}, label);
  }
  EXPECT_EQ(mapped,
            LatticeGraph::from_geometry(square, {1, 2, 3}).edge_labels());
  for (std::uint64_t shell : {1, 2, 3}) {
    SCOPED_TRACE(shell);
    const auto single = LatticeGraph::from_geometry(skewed, {shell});
    for (const auto& [edge, label] : single.edge_labels()) {
      EXPECT_EQ(graph.edge_labels().at(edge), label);
    }
    EXPECT_EQ(single.edge_labels().size(), 72);
  }
}

TEST_F(LatticeGeometryTest, MergedPeriodicShellsDoNotDependOnSelection) {
  // At tolerance 0.5, distances 1 and 2 form shell 1 wherever it is requested.
  const auto ring = LatticeGeometry::chain(20, true);
  const auto first = LatticeGraph::from_geometry(ring, {1}, {}, 1.0, 0.5);
  const auto both = LatticeGraph::from_geometry(ring, {1, 2}, {}, 1.0, 0.5);
  EXPECT_EQ(first.edge_labels().size(), 40);
  EXPECT_EQ(both.edge_labels().size(), 120);
  for (const auto& [edge, label] : first.edge_labels()) {
    EXPECT_EQ(both.edge_labels().at(edge), label);
  }
}

TEST_F(LatticeGeometryTest, SerializationRetainsLayoutAndHash) {
  for (const auto& [px, py] : boundary_modes) {
    const auto geometry = LatticeGeometry::honeycomb_plaquettes(2, 2, px, py);
    const auto graph_hash =
        LatticeGraph::from_geometry(geometry).content_hash();
    for (const std::string type : {"json", "hdf5"}) {
      const std::string filename = "test_roundtrip.lattice_geometry." +
                                   std::string(type == "json" ? "json" : "h5");
      geometry.to_file(filename, type);
      const auto restored = LatticeGeometry::from_file(filename, type);
      std::filesystem::remove(filename);
      EXPECT_EQ(restored.to_json(), geometry.to_json());
      EXPECT_EQ(restored.positions(), geometry.positions());
      ASSERT_EQ(restored.periods().has_value(), geometry.periods().has_value());
      if (geometry.periods()) {
        EXPECT_EQ(*restored.periods(), *geometry.periods());
      }
      EXPECT_EQ(restored.content_hash(), geometry.content_hash());
      EXPECT_EQ(LatticeGraph::from_geometry(restored).content_hash(),
                graph_hash);
    }
  }
}

TEST_F(LatticeGeometryTest, IntegerLayoutSurvivesSerialization) {
  const auto geometry =
      LatticeGeometry::honeycomb_plaquettes(2, 2, true, false);
  const auto json = geometry.to_json();
  ASSERT_TRUE(json.contains("integer_embedding"));
  EXPECT_FALSE(json.contains("positions"));
  EXPECT_EQ(LatticeGeometry::from_json(json).to_json(), json);
  const std::string filename = "test_layout.lattice_geometry.h5";
  geometry.to_hdf5_file(filename);
  const auto hdf5 = LatticeGeometry::from_hdf5_file(filename);
  std::filesystem::remove(filename);
  EXPECT_EQ(hdf5.to_json(), json);

  const std::vector<std::pair<std::string, nlohmann::json>> invalid = {
      {"/integer_embedding/nx", 0},
      {"/integer_embedding/site_by_coordinate/0", 99},
      {"/integer_embedding/site_by_coordinate/1", 0},
      {"/integer_embedding/primitive_vectors", {{1.0, 0.0}}},
      {"/integer_embedding/primitive_vectors/0",
       {std::numeric_limits<double>::infinity(), 0.0}},
      {"/integer_embedding/basis/0/0",
       std::numeric_limits<double>::quiet_NaN()},
      {"/integer_embedding/primitive_vectors/0", {0.0, 0.0}}};
  for (const auto& [path, value] : invalid) {
    SCOPED_TRACE(path);
    auto malformed = json;
    malformed[nlohmann::json::json_pointer(path)] = value;
    EXPECT_THROW(LatticeGeometry::from_json(malformed), std::invalid_argument);
  }
  // Two periodic directions require independent primitive vectors.
  auto dependent = json;
  dependent["integer_embedding"]["periodic_y"] = true;
  dependent["integer_embedding"]["primitive_vectors"] = {{1.0, 0.0},
                                                         {-2.0, 0.0}};
  EXPECT_THROW(LatticeGeometry::from_json(dependent), std::invalid_argument);
  // Open directions the patch spans need nonzero, independent vectors too.
  const auto open = LatticeGeometry::square(2, 2).to_json();
  for (const auto& vectors :
       std::vector<nlohmann::json>{{{0.0, 0.0}, {0.0, 1.0}},
                                   {{1.0, 0.0}, {0.0, 0.0}},
                                   {{1.0, 0.0}, {-2.0, 0.0}}}) {
    SCOPED_TRACE(vectors.dump());
    auto degenerate = open;
    degenerate["integer_embedding"]["primitive_vectors"] = vectors;
    EXPECT_THROW(LatticeGeometry::from_json(degenerate), std::invalid_argument);
  }
  // A chain never spans its second direction, which stays zero.
  EXPECT_NO_THROW(
      LatticeGeometry::from_json(LatticeGeometry::chain(3).to_json()));
  for (const auto& layout : std::vector<nlohmann::json>{
           {{"version", json.at("version")}},
           {{"version", json.at("version")}, {"positions", {{0.0, 0.0}}}}}) {
    EXPECT_THROW(LatticeGeometry::from_json(layout), std::invalid_argument);
  }
  auto text = json;
  text["integer_embedding"]["site_by_coordinate"][0] = "text";
  EXPECT_THROW(LatticeGeometry::from_json(text), nlohmann::json::type_error);
}

TEST_F(LatticeGeometryTest, SerializationRequiresCompatibleVersion) {
  const auto geometry = LatticeGeometry::square(2, 2, true, false);
  const auto json = geometry.to_json();
  ASSERT_TRUE(json.at("version").is_string());
  auto unversioned = json;
  unversioned.erase("version");
  auto incompatible = json;
  incompatible["version"] = "99.0.0";
  for (const auto& invalid : std::vector<nlohmann::json>{
           nlohmann::json::array(), unversioned, incompatible}) {
    EXPECT_THROW(LatticeGeometry::from_json(invalid), std::runtime_error);
  }

  const std::string filename = "test_version.lattice_geometry.h5";
  for (const std::string version : {"", "99.0.0"}) {
    SCOPED_TRACE(version);
    geometry.to_hdf5_file(filename);
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
    EXPECT_THROW(LatticeGeometry::from_hdf5_file(filename), std::runtime_error);
    std::filesystem::remove(filename);
  }
}

TEST_F(LatticeGeometryTest, Hdf5RejectsInvalidLayout) {
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
  expect_invalid([](H5::Group& root) { root.unlink("integer_embedding"); });
  const hsize_t dimensions[3] = {2, 2, 2};
  for (const std::string name : {"primitive_vectors", "basis"}) {
    SCOPED_TRACE(name);
    for (int rank : {1, 3}) {
      expect_invalid([&](H5::Group& root) {
        auto layout = root.openGroup("integer_embedding");
        layout.unlink(name);
        layout.createDataSet(name, H5::PredType::NATIVE_DOUBLE,
                             H5::DataSpace(rank, dimensions));
      });
    }
    expect_invalid([&](H5::Group& root) {
      auto layout = root.openGroup("integer_embedding");
      layout.unlink(name);
      layout.createDataSet(name, H5::StrType(H5::PredType::C_S1, 8),
                           H5::DataSpace(2, dimensions));
    });
  }
  expect_invalid([](H5::Group& root) {
    const double values[4] = {std::numeric_limits<double>::quiet_NaN(), 0.0,
                              0.0, 1.0};
    root.openGroup("integer_embedding")
        .openDataSet("primitive_vectors")
        .write(values, H5::PredType::NATIVE_DOUBLE);
  });
  expect_invalid([](H5::Group& root) {
    auto layout = root.openGroup("integer_embedding");
    layout.unlink("shape");
    const int values[3] = {2, 2, 1};
    const hsize_t size = 3;
    layout
        .createDataSet("shape", H5::PredType::NATIVE_INT,
                       H5::DataSpace(1, &size))
        .write(values, H5::PredType::NATIVE_INT);
  });
  // Integer fields must have an integer type, and periodic flags must be 0/1.
  expect_invalid([](H5::Group& root) {
    auto layout = root.openGroup("integer_embedding");
    layout.unlink("shape");
    const double values[4] = {2.0, 2.0, 1.0, 0.0};
    const hsize_t size = 4;
    layout
        .createDataSet("shape", H5::PredType::NATIVE_DOUBLE,
                       H5::DataSpace(1, &size))
        .write(values, H5::PredType::NATIVE_DOUBLE);
  });
  for (int flag : {2, -1}) {
    SCOPED_TRACE(flag);
    expect_invalid([flag](H5::Group& root) {
      auto layout = root.openGroup("integer_embedding");
      layout.unlink("shape");
      const int values[4] = {2, 2, flag, 0};
      const hsize_t size = 4;
      layout
          .createDataSet("shape", H5::PredType::NATIVE_INT,
                         H5::DataSpace(1, &size))
          .write(values, H5::PredType::NATIVE_INT);
    });
  }
}
