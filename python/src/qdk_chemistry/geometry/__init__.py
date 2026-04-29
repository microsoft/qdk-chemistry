"""Geometry module for representing quantum system topologies.

This module provides hypergraph data structures for representing the
geometric structure of quantum systems, including lattice topologies,
interaction graphs, and optimal edge colorings for Trotterized
Hamiltonian simulation.

The key classes are:

- :class:`Hyperedge`: An edge connecting one or more vertices.
- :class:`Hypergraph`: A collection of hyperedges with generic greedy edge coloring.
- :class:`HypergraphEdgeColoring`: A mapping from edges to integer color labels.
- :class:`Chain1D`, :class:`Ring1D`: 1D lattice geometries with optimal colorings.
- :class:`Patch2D`, :class:`Torus2D`: 2D lattice geometries with optimal colorings.
- :class:`CompleteGraph`, :class:`CompleteBipartiteGraph`: Complete graph geometries.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.geometry.complete import CompleteBipartiteGraph, CompleteGraph
from qdk_chemistry.geometry.hypergraph import Hyperedge, Hypergraph, HypergraphEdgeColoring
from qdk_chemistry.geometry.lattice1d import Chain1D, Ring1D
from qdk_chemistry.geometry.lattice2d import Patch2D, Torus2D

__all__ = [
    "Chain1D",
    "CompleteBipartiteGraph",
    "CompleteGraph",
    "Hyperedge",
    "Hypergraph",
    "HypergraphEdgeColoring",
    "Patch2D",
    "Ring1D",
    "Torus2D",
]
