"""Two-dimensional lattice geometries for quantum simulations.

This module provides classes for representing 2D lattice structures as
hypergraphs with optimal edge colorings. These lattices are commonly used
in quantum spin system simulations.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.geometry.hypergraph import Hyperedge, Hypergraph, HypergraphEdgeColoring

__all__ = ["Patch2D", "Torus2D"]


class Patch2D(Hypergraph):
    """A two-dimensional open rectangular lattice.

    Represents a rectangular grid of vertices with nearest-neighbor edges
    and open boundary conditions.

    Vertices are indexed in row-major order: vertex ``(x, y)`` has index
    ``y * width + x``.

    Attributes:
        width: Number of vertices in the horizontal direction.
        height: Number of vertices in the vertical direction.

    """

    def __init__(self, width: int, height: int, self_loops: bool = False) -> None:
        """Initialize a 2D patch lattice.

        Args:
            width: Number of vertices in the horizontal direction.
            height: Number of vertices in the vertical direction.
            self_loops: If ``True``, include self-loop edges on each vertex for single-site terms.

        """
        self.width = width
        self.height = height

        if self_loops:
            _edges = [Hyperedge([i]) for i in range(width * height)]
        else:
            _edges = []

        # Horizontal edges
        for y in range(height):
            for x in range(width - 1):
                _edges.append(Hyperedge([self._index(x, y), self._index(x + 1, y)]))

        # Vertical edges
        for y in range(height - 1):
            for x in range(width):
                _edges.append(Hyperedge([self._index(x, y), self._index(x, y + 1)]))

        super().__init__(_edges)

    def _index(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to vertex index."""
        return y * self.width + x

    def edge_coloring(self, seed: int | None = 0, trials: int = 1) -> HypergraphEdgeColoring:
        """Compute an optimal 4-color edge coloring for this 2D patch.

        Uses a deterministic parity-based assignment:

        - Color 0: horizontal edges with even left-x
        - Color 1: horizontal edges with odd left-x
        - Color 2: vertical edges with even top-y
        - Color 3: vertical edges with odd top-y

        """
        coloring = HypergraphEdgeColoring(self)
        for edge in self.edges():
            if len(edge.vertices) == 1:
                coloring.add_edge(edge, -1)
                continue

            u, v = edge.vertices
            x_u, y_u = u % self.width, u // self.width
            x_v, y_v = v % self.width, v // self.width

            if y_u == y_v:
                color = 0 if min(x_u, x_v) % 2 == 0 else 1
            else:
                color = 2 if min(y_u, y_v) % 2 == 0 else 3
            coloring.add_edge(edge, color)
        return coloring

    def __str__(self) -> str:
        return f"{self.width}x{self.height} lattice patch with {self.nvertices} vertices and {self.nedges} edges"

    def __repr__(self) -> str:
        return f"Patch2D(width={self.width}, height={self.height})"


class Torus2D(Hypergraph):
    """A two-dimensional toroidal (periodic) lattice.

    Represents a rectangular grid of vertices with nearest-neighbor edges
    and periodic boundary conditions in both directions.

    Vertices are indexed in row-major order: vertex ``(x, y)`` has index
    ``y * width + x``.

    Attributes:
        width: Number of vertices in the horizontal direction.
        height: Number of vertices in the vertical direction.

    """

    def __init__(self, width: int, height: int, self_loops: bool = False) -> None:
        """Initialize a 2D torus lattice.

        Args:
            width: Number of vertices in the horizontal direction.
            height: Number of vertices in the vertical direction.
            self_loops: If ``True``, include self-loop edges on each vertex for single-site terms.

        """
        self.width = width
        self.height = height

        if self_loops:
            _edges = [Hyperedge([i]) for i in range(width * height)]
        else:
            _edges = []

        # Horizontal edges (periodic)
        for y in range(height):
            for x in range(width):
                _edges.append(Hyperedge([self._index(x, y), self._index((x + 1) % width, y)]))

        # Vertical edges (periodic)
        for y in range(height):
            for x in range(width):
                _edges.append(Hyperedge([self._index(x, y), self._index(x, (y + 1) % height)]))

        super().__init__(_edges)

    def _index(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to vertex index."""
        return y * self.width + x

    def edge_coloring(self, seed: int | None = 0, trials: int = 1) -> HypergraphEdgeColoring:
        """Compute an edge coloring for this 2D torus.

        Uses 4 colors for even dimensions and up to 6 colors when
        wrap-around edges need extra colors due to odd dimensions.

        """
        coloring = HypergraphEdgeColoring(self)
        for edge in self.edges():
            if len(edge.vertices) == 1:
                coloring.add_edge(edge, -1)
                continue

            u, v = edge.vertices
            x_u, y_u = u % self.width, u // self.width
            x_v, y_v = v % self.width, v // self.width

            if y_u == y_v:
                if {x_u, x_v} == {0, self.width - 1}:
                    color = 1 if self.width % 2 == 0 else 4
                else:
                    color = 0 if min(x_u, x_v) % 2 == 0 else 1
            elif {y_u, y_v} == {0, self.height - 1}:
                color = 3 if self.height % 2 == 0 else 5
            else:
                color = 2 if min(y_u, y_v) % 2 == 0 else 3
            coloring.add_edge(edge, color)
        return coloring

    def __str__(self) -> str:
        return f"{self.width}x{self.height} lattice torus with {self.nvertices} vertices and {self.nedges} edges"

    def __repr__(self) -> str:
        return f"Torus2D(width={self.width}, height={self.height})"
