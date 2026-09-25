"""Lattice geometry, explicit connectivity, and edge-coloring examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
from qdk_chemistry.data import BondFlavorDefinition, LatticeGeometry, LatticeGraph

################################################################################
# start-cell-create-geometry
geometry = LatticeGeometry.honeycomb_plaquettes(1, 1)
print(f"Hexagon geometry: {geometry.num_sites} sites")
print(f"Positions:\n{geometry.positions}")

custom_geometry = LatticeGeometry(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]))
# end-cell-create-geometry
################################################################################

################################################################################
# start-cell-query-geometry
shell_pairs = geometry.nearest_neighbor_shells([1, 2, 3])
print({shell: len(pairs) for shell, pairs in shell_pairs.items()})  # {1: 6, 2: 6, 3: 3}
geometric_connections = geometry.neighbor_connections([1, 2, 3])
# Geometry queries return flavor=None and weight=1.0; no graph is created.
# end-cell-query-geometry
################################################################################

################################################################################
# start-cell-periodic-geometry
periodic_geometry = LatticeGeometry.chain(2, periodic=True)
for connection in periodic_geometry.neighbor_connections([1]):
    print(connection.site_i, connection.site_j, connection.image_shift)
# Two physical images connect the same finite-lattice pair.
# end-cell-periodic-geometry
################################################################################

################################################################################
# start-cell-from-geometry
square_geometry = LatticeGeometry.square(3, 3)
graph = LatticeGraph.from_geometry(square_geometry, shells=[1, 2])
print(f"Selected shells: {graph.selected_shells}")
print(f"Interaction edges: {graph.num_edges}")
# end-cell-from-geometry
################################################################################

################################################################################
# start-cell-bond-flavors
flavored_graph = graph.with_bond_flavors(
    [
        BondFlavorDefinition(1, np.array([1.0, 0.0]), 10),
        BondFlavorDefinition(1, np.array([0.0, 1.0]), 20),
        BondFlavorDefinition(2, np.array([1.0, 1.0]), 30),
        BondFlavorDefinition(2, np.array([1.0, -1.0]), 40),
    ]
)
connection = flavored_graph.connections[0]
print(
    connection.site_i,
    connection.site_j,
    connection.bond_class.shell,
    connection.flavor,
    connection.weight,
)
# end-cell-bond-flavors
################################################################################

################################################################################
# start-cell-coloring
# Here XX acts on shell 1 and ZZ on shell 2, with nonzero couplings throughout.
xx_pairs = [(c.site_i, c.site_j) for c in graph.connections if c.bond_class.shell == 1]
zz_pairs = [(c.site_i, c.site_j) for c in graph.connections if c.bond_class.shell == 2]
# Restrict the constructor's coloring; neither family is recolored.
coloring = graph.edge_coloring
assert coloring is not None
xx_coloring = {pair: coloring[pair] for pair in xx_pairs}
zz_coloring = {pair: coloring[pair] for pair in zz_pairs}
# end-cell-coloring
################################################################################

################################################################################
# start-cell-create-chain
# Create a 6-site open chain
chain = LatticeGraph.chain(6)
print(f"Chain: {chain.num_sites} sites, {chain.num_edges} edges")

# Create a 6-site periodic chain (ring)
ring = LatticeGraph.chain(6, periodic=True)
print(f"Ring: {ring.num_sites} sites, {ring.num_edges} edges")

# Create a chain with custom hopping weight
chain_weighted = LatticeGraph.chain(4, t=0.5)
# end-cell-create-chain
################################################################################

################################################################################
# start-cell-create-2d
# Create a 4x3 square lattice
square = LatticeGraph.square(4, 3)
print(f"Square: {square.num_sites} sites, {square.num_edges} edges")

# Create a 3x3 triangular lattice
triangular = LatticeGraph.triangular(3, 3)
print(f"Triangular: {triangular.num_sites} sites, {triangular.num_edges} edges")

# Create a 3x2 honeycomb lattice (2 sites per unit cell)
honeycomb = LatticeGraph.honeycomb(3, 2)
print(f"Honeycomb: {honeycomb.num_sites} sites, {honeycomb.num_edges} edges")

# Create one isolated six-site honeycomb plaquette
hexagon = LatticeGraph.honeycomb_plaquettes(1, 1)
print(f"Hexagon: {hexagon.num_sites} sites, {hexagon.num_edges} edges")

# Create a 3x2 kagome lattice (3 sites per unit cell)
kagome = LatticeGraph.kagome(3, 2)
print(f"Kagome: {kagome.num_sites} sites, {kagome.num_edges} edges")

# Create a periodic square lattice (torus topology)
torus = LatticeGraph.square(4, 4, periodic_x=True, periodic_y=True)
print(f"Torus: {torus.num_sites} sites, {torus.num_edges} edges")
# end-cell-create-2d
################################################################################

################################################################################
# start-cell-periodic
# Periodic chain (ring)
ring = LatticeGraph.chain(6, periodic=True)
print(f"Ring: {ring.num_sites} sites, {ring.num_edges} edges")

# Cylinder: periodic in x only
cylinder = LatticeGraph.square(4, 4, periodic_x=True, periodic_y=False)
print(f"Cylinder: {cylinder.num_sites} sites, {cylinder.num_edges} edges")

# Torus: periodic in both directions
torus = LatticeGraph.square(4, 4, periodic_x=True, periodic_y=True)
print(f"Torus: {torus.num_sites} sites, {torus.num_edges} edges")
# end-cell-periodic
################################################################################

################################################################################
# start-cell-from-matrix
# Create a lattice from a dense adjacency matrix (star graph)
adj = np.zeros((5, 5))
for i in range(1, 5):
    adj[0, i] = 1.0
    adj[i, 0] = 1.0

star_graph = LatticeGraph.from_dense_matrix(adj)
print(f"Star graph: {star_graph.num_sites} sites, {star_graph.num_edges} edges")

# Create a lattice from an edge dictionary
edges = {(0, 1): 1.0, (1, 0): 1.0, (1, 2): 0.5, (2, 1): 0.5}
custom_lattice = LatticeGraph(edge_weights=edges, num_sites=3)
print(f"Custom: {custom_lattice.num_sites} sites, {custom_lattice.num_edges} edges")

# Make a directed graph bidirectional
directed_edges = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
directed = LatticeGraph(edge_weights=directed_edges, num_sites=4)
bidirectional = LatticeGraph.make_bidirectional(directed)
print(f"Bidirectional: is_symmetric = {bidirectional.is_symmetric}")
# end-cell-from-matrix
################################################################################

################################################################################
# start-cell-properties
# Query lattice properties
lattice = LatticeGraph.chain(4)

# Geometry is optional for general graphs, but present on built-in factories.
assert lattice.geometry is not None
print(f"Positions:\n{lattice.geometry.positions}")

# Check connectivity
print(f"Sites 0-1 connected: {lattice.are_connected(0, 1)}")
print(f"Sites 0-2 connected: {lattice.are_connected(0, 2)}")

# Get edge weight
print(f"Weight(0, 1) = {lattice.weight(0, 1)}")
print(f"Weight(0, 2) = {lattice.weight(0, 2)}")

# Check symmetry
print(f"Is symmetric: {lattice.is_symmetric}")

# Get the full adjacency matrix
adj_matrix = lattice.adjacency_matrix()
print(f"Adjacency matrix:\n{adj_matrix}")
# end-cell-properties
################################################################################

################################################################################
# start-cell-serialization
lattice = flavored_graph

# Save to JSON
lattice.to_json_file(Path("square.lattice_graph.json"))

# Load from JSON
loaded = LatticeGraph.from_json_file(Path("square.lattice_graph.json"))
print(f"Loaded lattice: {loaded.num_sites} sites, {loaded.num_edges} edges")

# Save to HDF5
lattice.to_hdf5_file(Path("square.lattice_graph.hdf5"))
# end-cell-serialization
Path("square.lattice_graph.json").unlink()
Path("square.lattice_graph.hdf5").unlink()
################################################################################
