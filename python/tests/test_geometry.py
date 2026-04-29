"""Tests for geometry module: hypergraph, edge coloring, and lattice geometries."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.geometry import (
    Chain1D,
    CompleteBipartiteGraph,
    CompleteGraph,
    Hyperedge,
    Hypergraph,
    HypergraphEdgeColoring,
    Patch2D,
    Ring1D,
    Torus2D,
)


class TestHyperedge:
    """Tests for Hyperedge."""

    def test_sorted_vertices(self):
        edge = Hyperedge([3, 1, 2])
        assert edge.vertices == (1, 2, 3)

    def test_deduplication(self):
        edge = Hyperedge([1, 2, 2, 3])
        assert edge.vertices == (1, 2, 3)

    def test_equality(self):
        assert Hyperedge([0, 1]) == Hyperedge([1, 0])
        assert Hyperedge([0, 1]) != Hyperedge([0, 2])

    def test_hash(self):
        s = {Hyperedge([0, 1]), Hyperedge([1, 0])}
        assert len(s) == 1


class TestHypergraph:
    """Tests for Hypergraph."""

    def test_basic_properties(self):
        edges = [Hyperedge([0, 1]), Hyperedge([1, 2]), Hyperedge([0, 2])]
        g = Hypergraph(edges)
        assert g.nvertices == 3
        assert g.nedges == 3

    def test_add_edge(self):
        g = Hypergraph([Hyperedge([0, 1])])
        g.add_edge(Hyperedge([1, 2]))
        assert g.nvertices == 3
        assert g.nedges == 2

    def test_greedy_coloring(self):
        # Triangle graph — needs 3 colors
        edges = [Hyperedge([0, 1]), Hyperedge([1, 2]), Hyperedge([0, 2])]
        g = Hypergraph(edges)
        coloring = g.edge_coloring()
        assert coloring.ncolors == 3
        _validate_coloring(coloring)

    def test_greedy_coloring_empty(self):
        g = Hypergraph([])
        coloring = g.edge_coloring()
        assert coloring.ncolors == 0

    def test_from_lattice_graph(self):
        """Test creating a Hypergraph from a LatticeGraph."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(4)
            g = Hypergraph.from_lattice_graph(lattice)
            assert g.nvertices == 4
            assert g.nedges == 3
        except ImportError:
            pytest.skip("LatticeGraph not available")


class TestHypergraphEdgeColoring:
    """Tests for HypergraphEdgeColoring."""

    def test_add_and_query(self):
        edges = [Hyperedge([0, 1]), Hyperedge([2, 3])]
        g = Hypergraph(edges)
        coloring = HypergraphEdgeColoring(g)
        for edge in g.edges():
            coloring.add_edge(edge, 0)
        assert coloring.ncolors == 1
        for edge in g.edges():
            assert coloring.color(edge.vertices) == 0

    def test_conflict_detection(self):
        edges = [Hyperedge([0, 1]), Hyperedge([1, 2])]
        g = Hypergraph(edges)
        coloring = HypergraphEdgeColoring(g)

        e0_1 = [e for e in g.edges() if e.vertices == (0, 1)][0]
        e1_2 = [e for e in g.edges() if e.vertices == (1, 2)][0]

        coloring.add_edge(e0_1, 0)
        with pytest.raises(RuntimeError):
            coloring.add_edge(e1_2, 0)

    def test_self_loop_color(self):
        edges = [Hyperedge([0]), Hyperedge([0, 1])]
        g = Hypergraph(edges)
        coloring = HypergraphEdgeColoring(g)
        self_loop = [e for e in g.edges() if len(e.vertices) == 1][0]
        coloring.add_edge(self_loop, -1)
        assert coloring.color(self_loop.vertices) == -1
        assert coloring.ncolors == 0  # -1 doesn't count


class TestChain1D:
    """Tests for Chain1D geometry."""

    def test_basic(self):
        chain = Chain1D(5)
        assert chain.nvertices == 5
        assert chain.nedges == 4
        assert chain.length == 5

    def test_edge_coloring(self):
        chain = Chain1D(6)
        coloring = chain.edge_coloring()
        assert coloring.ncolors == 2
        _validate_coloring(coloring)

    def test_with_self_loops(self):
        chain = Chain1D(4, self_loops=True)
        assert chain.nedges == 4 + 3  # 4 self-loops + 3 edges

    def test_single_vertex(self):
        chain = Chain1D(1)
        assert chain.nvertices == 1
        assert chain.nedges == 0


class TestRing1D:
    """Tests for Ring1D geometry."""

    def test_basic(self):
        ring = Ring1D(5)
        assert ring.nvertices == 5
        assert ring.nedges == 5

    def test_even_ring_coloring(self):
        ring = Ring1D(6)
        coloring = ring.edge_coloring()
        assert coloring.ncolors == 2
        _validate_coloring(coloring)

    def test_odd_ring_coloring(self):
        ring = Ring1D(5)
        coloring = ring.edge_coloring()
        assert coloring.ncolors == 3
        _validate_coloring(coloring)


class TestPatch2D:
    """Tests for Patch2D geometry."""

    def test_basic(self):
        patch = Patch2D(3, 2)
        assert patch.nvertices == 6
        # 3 horizontal per row × 2 rows = 4 horizontal, 3 vertical per column × 1 = 3 vertical = 7
        assert patch.nedges == 7

    def test_edge_coloring(self):
        patch = Patch2D(4, 4)
        coloring = patch.edge_coloring()
        assert coloring.ncolors == 4
        _validate_coloring(coloring)

    def test_small_patch(self):
        patch = Patch2D(2, 2)
        coloring = patch.edge_coloring()
        assert coloring.ncolors <= 4
        _validate_coloring(coloring)


class TestTorus2D:
    """Tests for Torus2D geometry."""

    def test_basic(self):
        torus = Torus2D(3, 2)
        assert torus.nvertices == 6
        # Periodic wrapping may produce duplicate edges (stored in a set)
        # 3 horizontal per row × 2 rows = 6 horizontal
        # height=2 means vertical edges wrap back, so only 3 unique vertical edges
        assert torus.nedges == 9

    def test_even_torus_coloring(self):
        torus = Torus2D(4, 4)
        coloring = torus.edge_coloring()
        assert coloring.ncolors == 4
        _validate_coloring(coloring)

    def test_odd_width_torus_coloring(self):
        torus = Torus2D(3, 4)
        coloring = torus.edge_coloring()
        assert coloring.ncolors <= 6
        _validate_coloring(coloring)

    def test_odd_both_torus_coloring(self):
        torus = Torus2D(3, 3)
        coloring = torus.edge_coloring()
        assert coloring.ncolors <= 6
        _validate_coloring(coloring)


class TestCompleteGraph:
    """Tests for CompleteGraph."""

    def test_basic(self):
        g = CompleteGraph(4)
        assert g.nvertices == 4
        assert g.nedges == 6

    def test_even_coloring(self):
        g = CompleteGraph(4)
        coloring = g.edge_coloring()
        assert coloring.ncolors == 3  # n-1 for even n
        _validate_coloring(coloring)

    def test_odd_coloring(self):
        g = CompleteGraph(5)
        coloring = g.edge_coloring()
        assert coloring.ncolors == 5  # n for odd n
        _validate_coloring(coloring)


class TestCompleteBipartiteGraph:
    """Tests for CompleteBipartiteGraph."""

    def test_basic(self):
        g = CompleteBipartiteGraph(2, 3)
        assert g.nvertices == 5
        assert g.nedges == 6

    def test_coloring(self):
        g = CompleteBipartiteGraph(2, 3)
        coloring = g.edge_coloring()
        assert coloring.ncolors == 3  # n colors
        _validate_coloring(coloring)


def _validate_coloring(coloring: HypergraphEdgeColoring) -> None:
    """Validate that a coloring has no vertex conflicts within each color."""
    for c in coloring.colors():
        edges_in_color = coloring.edges_of_color(c)
        vertices_seen: set[int] = set()
        for edge in edges_in_color:
            for v in edge.vertices:
                assert v not in vertices_seen, f"Vertex {v} appears in multiple edges of color {c}"
                vertices_seen.add(v)
