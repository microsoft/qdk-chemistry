"""One-dimensional lattice geometries for quantum simulations.

This module provides classes for representing 1D lattice structures as
hypergraphs with optimal edge colorings. These lattices are commonly used
in quantum spin chain simulations.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.geometry.hypergraph import Hyperedge, Hypergraph, HypergraphEdgeColoring

__all__ = ["Chain1D", "Ring1D"]


class Chain1D(Hypergraph):
    """A one-dimensional open chain lattice.

    Represents a linear chain of vertices with nearest-neighbor edges.
    The chain has open boundary conditions.

    Attributes:
        length: Number of vertices in the chain.

    """

    def __init__(self, length: int, self_loops: bool = False) -> None:
        """Initialize a 1D chain lattice.

        Args:
            length: Number of vertices in the chain.
            self_loops: If ``True``, include self-loop edges on each vertex for single-site terms.

        """
        if self_loops:
            _edges = [Hyperedge([i]) for i in range(length)]
        else:
            _edges = []

        for i in range(length - 1):
            _edges.append(Hyperedge([i, i + 1]))

        super().__init__(_edges)
        # Ensure all sites are represented even without edges
        for i in range(length):
            self._vertex_set.add(i)
        self.length = length

    def edge_coloring(self, seed: int | None = 0, trials: int = 1) -> HypergraphEdgeColoring:
        """Compute an optimal 2-color edge coloring for this chain.

        Uses even/odd parity of the left vertex to assign colors 0 and 1.
        This is provably optimal for open chains.

        """
        coloring = HypergraphEdgeColoring(self)
        for edge in self.edges():
            if len(edge.vertices) == 1:
                coloring.add_edge(edge, -1)
            else:
                i, j = edge.vertices
                color = min(i, j) % 2
                coloring.add_edge(edge, color)
        return coloring

    def __str__(self) -> str:
        return f"1D chain with {self.length} vertices and {self.nedges} edges"

    def __repr__(self) -> str:
        return f"Chain1D(length={self.length})"


class Ring1D(Hypergraph):
    """A one-dimensional ring (periodic chain) lattice.

    Represents a circular chain of vertices with nearest-neighbor edges
    and periodic boundary conditions.

    Attributes:
        length: Number of vertices in the ring.

    """

    def __init__(self, length: int, self_loops: bool = False) -> None:
        """Initialize a 1D ring lattice.

        Args:
            length: Number of vertices in the ring.
            self_loops: If ``True``, include self-loop edges on each vertex for single-site terms.

        """
        if self_loops:
            _edges = [Hyperedge([i]) for i in range(length)]
        else:
            _edges = []

        for i in range(length):
            _edges.append(Hyperedge([i, (i + 1) % length]))
        super().__init__(_edges)
        # Ensure all sites are represented even without edges
        for i in range(length):
            self._vertex_set.add(i)
        self.length = length

    def edge_coloring(self, seed: int | None = 0, trials: int = 1) -> HypergraphEdgeColoring:
        """Compute an optimal edge coloring for this ring.

        Uses 2 colors for even-length rings and 3 colors for odd-length rings.
        The wrap-around edge gets a third color when the ring has odd length.

        """
        coloring = HypergraphEdgeColoring(self)
        for edge in self.edges():
            if len(edge.vertices) == 1:
                coloring.add_edge(edge, -1)
            else:
                i, j = edge.vertices
                if {i, j} == {0, self.length - 1}:
                    color = (self.length % 2) + 1
                else:
                    color = min(i, j) % 2
                coloring.add_edge(edge, color)
        return coloring

    def __str__(self) -> str:
        return f"1D ring with {self.length} vertices and {self.nedges} edges"

    def __repr__(self) -> str:
        return f"Ring1D(length={self.length})"
