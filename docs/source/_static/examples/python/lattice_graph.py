"""LatticeGraph creation and manipulation example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np

from qdk_chemistry.data import LatticeGraph

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
lattice = LatticeGraph.chain(4)

# Save to JSON
lattice.to_json_file(Path("chain.lattice_graph.json"))

# Load from JSON
loaded = LatticeGraph.from_json_file(Path("chain.lattice_graph.json"))
print(f"Loaded lattice: {loaded.num_sites} sites, {loaded.num_edges} edges")

# Save to HDF5
lattice.to_hdf5_file(Path("chain.lattice_graph.hdf5"))
# end-cell-serialization
Path("chain.lattice_graph.json").unlink()
Path("chain.lattice_graph.hdf5").unlink()
################################################################################
