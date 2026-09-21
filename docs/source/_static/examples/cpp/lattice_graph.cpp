// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Lattice geometry, explicit connectivity, and edge-coloring examples.
// --------------------------------------------------------------------------------------------
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::data;

int main() {
  // start-cell-create-geometry
  auto geometry = LatticeGeometry::honeycomb_plaquettes(1, 1);
  const auto& positions = geometry.positions();

  Eigen::MatrixXd custom_positions(3, 2);
  custom_positions << 0.0, 0.0, 1.0, 0.0, 0.0, 2.0;
  LatticeGeometry custom_geometry(custom_positions);
  // end-cell-create-geometry

  // --------------------------------------------------------------------------------------------
  // start-cell-query-geometry
  auto shell_pairs = geometry.nearest_neighbor_shells({1, 2, 3});
  // Shells 1, 2, and 3 contain 6, 6, and 3 pairs, respectively.
  auto geometric_connections = geometry.neighbor_connections({1, 2, 3});
  // Geometry queries return no flavor and unit weight; no graph is created.
  // end-cell-query-geometry

  // --------------------------------------------------------------------------------------------
  // start-cell-periodic-geometry
  auto periodic_geometry = LatticeGeometry::chain(2, /*periodic=*/true);
  auto images = periodic_geometry.neighbor_connections({1});
  // Two physical images connect the same finite-lattice pair.
  // end-cell-periodic-geometry

  // --------------------------------------------------------------------------------------------
  // start-cell-from-geometry
  auto square_geometry = LatticeGeometry::square(3, 3);
  auto graph = LatticeGraph::from_geometry(square_geometry, {1, 2});
  const auto& selected_shells = graph.selected_shells();
  auto interaction_edges = graph.num_edges();
  // end-cell-from-geometry

  // --------------------------------------------------------------------------------------------
  // start-cell-bond-flavors
  auto flavored_graph = graph.with_bond_flavors({
      {1, Eigen::RowVector2d(1.0, 0.0), 10},
      {1, Eigen::RowVector2d(0.0, 1.0), 20},
      {2, Eigen::RowVector2d(1.0, 1.0), 30},
      {2, Eigen::RowVector2d(1.0, -1.0), 40},
  });
  const auto& connection = flavored_graph.connections().front();
  auto flavor = connection.flavor;  // Optional integer ID, not an edge color.
  // end-cell-bond-flavors

  // --------------------------------------------------------------------------------------------
  // start-cell-coloring
  // Here XX acts on shell 1 and ZZ on shell 2, with nonzero couplings
  // throughout.
  const auto& coloring = graph.edge_coloring().value();
  EdgeColoring xx_coloring, zz_coloring;
  for (const auto& c : graph.connections()) {
    const auto pair = std::make_pair(c.site_i, c.site_j);
    if (c.bond_class.shell == 1) {
      xx_coloring.emplace(pair, coloring.at(pair));
    } else if (c.bond_class.shell == 2) {
      zz_coloring.emplace(pair, coloring.at(pair));
    }
  }
  // end-cell-coloring

  // --------------------------------------------------------------------------------------------
  // start-cell-create-chain
  // Create a 6-site open chain
  auto chain = LatticeGraph::chain(6);

  // Create a 6-site periodic chain (ring)
  auto ring = LatticeGraph::chain(6, /*periodic=*/true);

  // Create a chain with custom hopping weight
  auto chain_weighted = LatticeGraph::chain(4, /*periodic=*/false, /*t=*/0.5);
  // end-cell-create-chain

  // --------------------------------------------------------------------------------------------
  // start-cell-create-2d
  // Create a 4x3 square lattice
  auto square = LatticeGraph::square(4, 3);

  // Create a 3x3 triangular lattice
  auto triangular = LatticeGraph::triangular(3, 3);

  // Create a 3x2 honeycomb lattice (2 sites per unit cell)
  auto honeycomb = LatticeGraph::honeycomb(3, 2);

  // Create one isolated six-site honeycomb plaquette
  auto hexagon = LatticeGraph::honeycomb_plaquettes(1, 1);

  // Create a 3x2 kagome lattice (3 sites per unit cell)
  auto kagome = LatticeGraph::kagome(3, 2);

  // Create a periodic square lattice (torus topology)
  auto torus =
      LatticeGraph::square(4, 4, /*periodic_x=*/true, /*periodic_y=*/true);
  // end-cell-create-2d

  // --------------------------------------------------------------------------------------------
  // start-cell-periodic
  // Periodic chain (ring)
  auto ring_example = LatticeGraph::chain(6, /*periodic=*/true);

  // Cylinder: periodic in x only
  auto cylinder =
      LatticeGraph::square(4, 4, /*periodic_x=*/true, /*periodic_y=*/false);

  // Torus: periodic in both directions
  auto torus_example =
      LatticeGraph::square(4, 4, /*periodic_x=*/true, /*periodic_y=*/true);
  // end-cell-periodic

  // --------------------------------------------------------------------------------------------
  // start-cell-from-matrix
  // Create a lattice from a dense adjacency matrix (star graph)
  Eigen::MatrixXd adj = Eigen::MatrixXd::Zero(5, 5);
  for (int i = 1; i < 5; i++) {
    adj(0, i) = 1.0;
    adj(i, 0) = 1.0;
  }
  auto star_graph = LatticeGraph::from_dense_matrix(adj);

  // Create a lattice from an edge-weight map
  std::map<std::pair<std::uint64_t, std::uint64_t>, double> edges = {
      {{0, 1}, 1.0}, {{1, 0}, 1.0}, {{1, 2}, 0.5}, {{2, 1}, 0.5}};
  LatticeGraph custom_lattice(edges, /*num_sites=*/3);

  // Make a directed graph bidirectional
  std::map<std::pair<std::uint64_t, std::uint64_t>, double> directed_edges = {
      {{0, 1}, 1.0}, {{1, 2}, 1.0}, {{2, 3}, 1.0}};
  LatticeGraph directed(directed_edges, 4);
  auto bidirectional = LatticeGraph::make_bidirectional(directed);
  // end-cell-from-matrix

  // --------------------------------------------------------------------------------------------
  // start-cell-properties
  // Query lattice properties
  auto lattice = LatticeGraph::chain(4);

  // Built-in graph factories retain their geometry.
  const auto& lattice_positions = lattice.geometry()->positions();

  // Check connectivity
  bool connected_01 = lattice.are_connected(0, 1);  // true
  bool connected_02 = lattice.are_connected(0, 2);  // false

  // Get edge weight
  double w01 = lattice.weight(0, 1);  // 1.0
  double w02 = lattice.weight(0, 2);  // 0.0

  // Check symmetry
  bool symmetric = lattice.is_symmetric();  // true

  // Get the full adjacency matrix
  Eigen::MatrixXd adj_matrix = lattice.adjacency_matrix();
  // end-cell-properties

  // --------------------------------------------------------------------------------------------
  // start-cell-serialization
  auto lg = flavored_graph;

  // Save to JSON
  lg.to_json_file("square.lattice_graph.json");

  // Load from JSON
  auto loaded = LatticeGraph::from_json_file("square.lattice_graph.json");

  // Save to HDF5
  lg.to_hdf5_file("square.lattice_graph.hdf5");
  // end-cell-serialization

  return 0;
}
