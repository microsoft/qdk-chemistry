"""Hypergraph data structures for representing quantum system geometries.

This module provides classes for representing hypergraphs, which generalize
graphs by allowing edges (hyperedges) to connect any number of vertices.
Hypergraphs are useful for representing interaction terms in quantum
Hamiltonians, where multi-body interactions can involve more than two sites.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import random
from collections.abc import Iterator

__all__ = ["Hyperedge", "Hypergraph", "HypergraphEdgeColoring"]


class Hyperedge:
    """A hyperedge connecting one or more vertices in a hypergraph.

    A hyperedge generalizes the concept of an edge in a graph. While a
    traditional edge connects exactly two vertices, a hyperedge can connect
    any number of vertices. Each hyperedge is defined by a set of unique
    vertex indices, stored as a sorted tuple for consistency and hashability.

    Attributes:
        vertices: Sorted tuple of vertex indices connected by this hyperedge.

    """

    def __init__(self, vertices: list[int]) -> None:
        """Initialize a hyperedge with the given vertices.

        Args:
            vertices: List of vertex indices. Will be sorted and deduplicated internally.

        """
        self.vertices: tuple[int, ...] = tuple(sorted(set(vertices)))

    def __str__(self) -> str:
        return str(self.vertices)

    def __repr__(self) -> str:
        return f"Hyperedge({list(self.vertices)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hyperedge):
            return NotImplemented
        return self.vertices == other.vertices

    def __hash__(self) -> int:
        return hash(self.vertices)


class Hypergraph:
    """A hypergraph consisting of vertices connected by hyperedges.

    A hypergraph is a generalization of a graph where edges (hyperedges) can
    connect any number of vertices. This class serves as the base class for
    various lattice geometries used in quantum simulations.

    Edge colors are managed separately by :class:`HypergraphEdgeColoring`.
    Use :meth:`edge_coloring` to generate a coloring for this hypergraph.

    """

    def __init__(self, edges: list[Hyperedge]) -> None:
        """Initialize a hypergraph with the given edges.

        Args:
            edges: List of hyperedges defining the hypergraph structure.

        """
        self._vertex_set: set[int] = set()
        self._edge_set: set[Hyperedge] = set(edges)
        for edge in edges:
            self._vertex_set.update(edge.vertices)

    @property
    def nvertices(self) -> int:
        """Return the number of vertices in the hypergraph."""
        return len(self._vertex_set)

    def vertices(self) -> Iterator[int]:
        """Iterate over all vertex indices in ascending order."""
        return iter(sorted(self._vertex_set))

    @property
    def nedges(self) -> int:
        """Return the number of hyperedges in the hypergraph."""
        return len(self._edge_set)

    def edges(self) -> Iterator[Hyperedge]:
        """Iterate over all hyperedges in the hypergraph."""
        return iter(self._edge_set)

    def add_edge(self, edge: Hyperedge) -> None:
        """Add a hyperedge to the hypergraph.

        Args:
            edge: The Hyperedge instance to add.

        """
        self._edge_set.add(edge)
        self._vertex_set.update(edge.vertices)

    def edge_coloring(self, seed: int | None = 0, trials: int = 1) -> HypergraphEdgeColoring:
        """Compute a greedy edge coloring of this hypergraph.

        Uses randomized greedy coloring with multiple trials. Each trial shuffles
        the edge order and assigns the smallest valid color. The coloring with the
        fewest colors across all trials is returned.

        Args:
            seed: Random seed for reproducibility. ``None`` for non-deterministic.
            trials: Number of randomized trials to attempt.

        Returns:
            A :class:`HypergraphEdgeColoring` with the fewest colors found.

        """
        all_edges = sorted(self.edges(), key=lambda edge: edge.vertices)

        if not all_edges:
            return HypergraphEdgeColoring(self)

        num_trials = max(trials, 1)
        best_coloring: HypergraphEdgeColoring | None = None
        least_colors: int | None = None

        for trial in range(num_trials):
            trial_seed = None if seed is None else seed + trial
            rng = random.Random(trial_seed)

            edge_order = list(all_edges)
            rng.shuffle(edge_order)

            coloring = HypergraphEdgeColoring(self)
            num_colors = 0

            for edge in edge_order:
                if len(edge.vertices) == 1:
                    coloring.add_edge(edge, -1)
                    continue

                assigned = False
                for color in range(num_colors):
                    used_vertices = set().union(*(candidate.vertices for candidate in coloring.edges_of_color(color)))
                    if not any(vertex in used_vertices for vertex in edge.vertices):
                        coloring.add_edge(edge, color)
                        assigned = True
                        break

                if not assigned:
                    coloring.add_edge(edge, num_colors)
                    num_colors += 1

            if least_colors is None or coloring.ncolors < least_colors:
                least_colors = coloring.ncolors
                best_coloring = coloring

        assert best_coloring is not None
        return best_coloring

    @staticmethod
    def from_lattice_graph(lattice_graph: object, self_loops: bool = False) -> Hypergraph:
        """Create a Hypergraph from a :class:`~qdk_chemistry.data.LatticeGraph`.

        Extracts the edges from the lattice graph's adjacency matrix and constructs
        a hypergraph with the same connectivity.

        Args:
            lattice_graph: A ``LatticeGraph`` instance.
            self_loops: If ``True``, include self-loop edges for each site.

        Returns:
            A :class:`Hypergraph` matching the lattice graph connectivity.

        """
        n = lattice_graph.num_sites
        adj = lattice_graph.adjacency_matrix()

        edges: list[Hyperedge] = []
        if self_loops:
            edges.extend(Hyperedge([i]) for i in range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] != 0.0:
                    edges.append(Hyperedge([i, j]))

        return Hypergraph(edges)

    def __str__(self) -> str:
        return f"Hypergraph with {self.nvertices} vertices and {self.nedges} edges."

    def __repr__(self) -> str:
        return f"Hypergraph({list(self._edge_set)})"


class HypergraphEdgeColoring:
    """Edge-color assignment for a :class:`Hypergraph`.

    Stores colors separately from :class:`Hypergraph` and enforces the rule
    that multi-vertex edges sharing a color do not share any vertices.

    Conventions:

    - Colors for nontrivial edges must be nonnegative integers.
    - Single-vertex edges may use a special color (for example ``-1``).
    - Only nonnegative colors contribute to :attr:`ncolors`.

    Attributes:
        hypergraph: The supporting :class:`Hypergraph` whose edges can be colored.

    """

    def __init__(self, hypergraph: Hypergraph) -> None:
        self.hypergraph = hypergraph
        self._colors: dict[tuple[int, ...], int] = {}
        self._used_vertices: dict[int, set[int]] = {}

    @property
    def ncolors(self) -> int:
        """Return the number of distinct nonnegative colors in the coloring."""
        return len(self._used_vertices)

    def color(self, vertices: tuple[int, ...]) -> int | None:
        """Return the color assigned to the edge with the given vertices.

        Args:
            vertices: Canonical vertex tuple for the edge to query.

        Returns:
            The color assigned to ``vertices``, or ``None`` if not colored.

        """
        if not isinstance(vertices, tuple) or not all(isinstance(v, int) for v in vertices):
            raise TypeError("vertices must be tuple[int, ...]")
        return self._colors.get(vertices)

    def colors(self) -> Iterator[int]:
        """Iterate over distinct nonnegative colors present in the coloring."""
        return iter(self._used_vertices.keys())

    def add_edge(self, edge: Hyperedge, color: int) -> None:
        """Add ``edge`` to this coloring with the specified ``color``.

        For multi-vertex edges, this enforces that no previously added edge
        with the same color shares a vertex with ``edge``.

        Args:
            edge: The Hyperedge instance to add (must belong to :attr:`hypergraph`).
            color: Color index for the edge.

        Raises:
            TypeError: If ``edge`` is not a :class:`Hyperedge`.
            ValueError: If ``edge`` is not part of :attr:`hypergraph`.
            ValueError: If ``color`` is negative for a nontrivial edge.
            RuntimeError: If adding ``edge`` would create a same-color vertex conflict.

        """
        if not isinstance(edge, Hyperedge):
            raise TypeError(f"edge must be Hyperedge, got {type(edge).__name__}")

        if edge not in self.hypergraph.edges():
            raise ValueError("edge must belong to the supporting Hypergraph")

        vertices = edge.vertices

        if len(vertices) == 1:
            self._colors[vertices] = color
        else:
            if color < 0:
                raise ValueError("Color index must be nonnegative for multi-vertex edges.")
            if color not in self._used_vertices:
                self._colors[vertices] = color
                self._used_vertices[color] = set(vertices)
            else:
                if any(v in self._used_vertices[color] for v in vertices):
                    raise RuntimeError("Edge conflicts with existing edge of same color.")
                self._colors[vertices] = color
                self._used_vertices[color].update(vertices)

    def edges_of_color(self, color: int) -> list[Hyperedge]:
        """Return hyperedges with a specific color.

        Args:
            color: Color index for filtering edges.

        Returns:
            List of edges currently assigned to ``color``.

        """
        return [edge for edge in self.hypergraph.edges() if self._colors.get(edge.vertices) == color]
