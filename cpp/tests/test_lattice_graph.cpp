// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <qdk/chemistry/data/lattice_graph.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class LatticeGraphTest : public ::testing::Test {};

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
  // 3x4 honeycomb lattice (24 sites)
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
    py_edges[{0, 19}] = 1.0;  // vertical wrap
    py_edges[{2, 21}] = 1.0;  // vertical wrap
    py_edges[{4, 23}] = 1.0;  // vertical wrap
    auto expected_py =
        LatticeGraph::make_bidirectional(LatticeGraph(py_edges, 24));

    auto hc_py = LatticeGraph::honeycomb(3, 4, false, true);
    EXPECT_EQ(hc_py.num_sites(), 24);
    EXPECT_EQ(hc_py.num_edges(), 32);  // 29 + 3
    EXPECT_TRUE(hc_py.is_symmetric());
    EXPECT_TRUE(
        hc_py.adjacency_matrix().isApprox(expected_py.adjacency_matrix()));
  }

  // periodic_x only: horizontal wraps
  {
    std::map<Edge, double> px_edges = expected_edges;
    px_edges[{0, 5}] = 1.0;    // horizontal wrap
    px_edges[{6, 11}] = 1.0;   // horizontal wrap
    px_edges[{12, 17}] = 1.0;  // horizontal wrap
    px_edges[{18, 23}] = 1.0;  // horizontal wrap
    auto expected_px =
        LatticeGraph::make_bidirectional(LatticeGraph(px_edges, 24));

    auto hc_px = LatticeGraph::honeycomb(3, 4, true, false);
    EXPECT_EQ(hc_px.num_sites(), 24);
    EXPECT_EQ(hc_px.num_edges(), 33);  // 29 + 4
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
