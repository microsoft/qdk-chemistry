"""Complete graph geometries for quantum simulations.

This module provides classes for representing complete graphs and complete
bipartite graphs as hypergraphs with optimal edge colorings.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.geometry.hypergraph import Hyperedge, Hypergraph, HypergraphEdgeColoring

__all__ = ["CompleteBipartiteGraph", "CompleteGraph"]


class CompleteGraph(Hypergraph):
    """A complete graph where every vertex is connected to every other vertex.

    In a complete graph K_n, there are n vertices and n(n-1)/2 edges,
    with each pair of distinct vertices connected by exactly one edge.

    Attributes:
        n: Number of vertices in the graph.

    """

    def __init__(self, n: int, self_loops: bool = False) -> None:
        """Initialize a complete graph.

        Args:
            n: Number of vertices in the graph.
            self_loops: If ``True``, include self-loop edges on each vertex for single-site terms.

        """
        if self_loops:
            _edges = [Hyperedge([i]) for i in range(n)]
        else:
            _edges = []

        for i in range(n):
            for j in range(i + 1, n):
                _edges.append(Hyperedge([i, j]))
        super().__init__(_edges)

        self.n = n

    def edge_coloring(self, seed: int | None = 0, trials: int = 1) -> HypergraphEdgeColoring:
        """Compute an optimal edge coloring for this complete graph.

        Uses a round-robin tournament schedule to achieve the chromatic index:
        (n-1) colors for even n and n colors for odd n (Vizing's theorem).

        """
        coloring = HypergraphEdgeColoring(self)
        for edge in self.edges():
            if len(edge.vertices) == 1:
                coloring.add_edge(edge, -1)
            else:
                i, j = edge.vertices
                if self.n % 2 == 0:
                    m = self.n - 1
                    if j == m:
                        coloring.add_edge(edge, i)
                    else:
                        coloring.add_edge(edge, ((i + j) * (self.n // 2)) % m)
                else:
                    coloring.add_edge(edge, ((i + j) * ((self.n + 1) // 2)) % self.n)
        return coloring

    def __str__(self) -> str:
        return f"Complete graph K_{self.n} with {self.nvertices} vertices and {self.nedges} edges"

    def __repr__(self) -> str:
        return f"CompleteGraph(n={self.n})"


class CompleteBipartiteGraph(Hypergraph):
    """A complete bipartite graph with two vertex sets.

    In a complete bipartite graph K_{m,n} (m <= n), there are m + n
    vertices partitioned into two sets of sizes m and n. Every vertex
    in the first set is connected to every vertex in the second set.

    Vertices 0 to m-1 form the first set, and vertices m to m+n-1
    form the second set.

    Attributes:
        m: Number of vertices in the first set.
        n: Number of vertices in the second set.

    """

    def __init__(self, m: int, n: int, self_loops: bool = False) -> None:
        """Initialize a complete bipartite graph.

        Args:
            m: Number of vertices in the first set (vertices 0 to m-1).
            n: Number of vertices in the second set (vertices m to m+n-1).
            self_loops: If ``True``, include self-loop edges on each vertex for single-site terms.

        Raises:
            AssertionError: If m > n.

        """
        assert m <= n, "Require m <= n for CompleteBipartiteGraph."
        total_vertices = m + n

        if self_loops:
            _edges = [Hyperedge([i]) for i in range(total_vertices)]
        else:
            _edges = []

        for i in range(m):
            for j in range(m, m + n):
                _edges.append(Hyperedge([i, j]))
        super().__init__(_edges)

        self.m = m
        self.n = n

    def edge_coloring(self, seed: int | None = 0, trials: int = 1) -> HypergraphEdgeColoring:
        """Compute an optimal edge coloring for this complete bipartite graph.

        Uses n colors, which is provably optimal (König's theorem).

        """
        coloring = HypergraphEdgeColoring(self)
        m = self.m
        n = self.n
        for edge in self.edges():
            if len(edge.vertices) == 1:
                coloring.add_edge(edge, -1)
            else:
                i, j = edge.vertices
                coloring.add_edge(edge, (i + j - m) % n)
        return coloring

    def __str__(self) -> str:
        return (
            f"Complete bipartite graph K_{{{self.m},{self.n}}} with {self.nvertices} vertices and {self.nedges} edges"
        )

    def __repr__(self) -> str:
        return f"CompleteBipartiteGraph(m={self.m}, n={self.n})"
