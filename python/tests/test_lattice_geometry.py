"""Tests for lattice geometry and explicitly selected interaction graphs."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qdk_chemistry.data import BondFlavorDefinition, EdgeLabel, LatticeGeometry, LatticeGraph

if TYPE_CHECKING:
    from pathlib import Path


class TestLatticeGeometry:
    """Check factory geometry properties, shell selection, and persistence."""

    def test_shell_selection_validates_indices(self) -> None:
        """Shell selection rejects zero, negative, and nonintegral shell indices."""
        geometry = LatticeGeometry.chain(3)
        with pytest.raises(ValueError, match="shell index must be > 0"):
            LatticeGraph.from_geometry(geometry, [0])
        for shell in (-1, 1.5):
            with pytest.raises(TypeError):
                LatticeGraph.from_geometry(geometry, [shell])

    @pytest.mark.parametrize("tolerance", [1.0, 2.0])
    def test_shell_selection_requires_tolerance_below_unit_length(self, tolerance: float) -> None:
        """Tolerances of at least the lattice unit length would merge every distance into one shell."""
        with pytest.raises(ValueError, match="less than 1"):
            LatticeGraph.from_geometry(LatticeGeometry.chain(3, periodic=True), [1], tolerance=tolerance)

    def test_properties_return_copies(self) -> None:
        """Mutating returned arrays leaves stored positions and periods unchanged."""
        geometry = LatticeGeometry.square(2, 1, periodic_x=True)
        copied_positions = geometry.positions
        copied_periods = geometry.periods
        assert copied_periods is not None
        copied_positions[:] = -7.0
        copied_periods[:] = -7.0

        np.testing.assert_array_equal(geometry.positions, [[0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_equal(geometry.periods, [[2.0, 0.0]])

    @pytest.mark.parametrize("periodic", [False, True])
    @pytest.mark.parametrize("format_name", ["json", "hdf5", "pickle"])
    def test_round_trip_preserves_geometry(self, tmp_path: Path, periodic: bool, format_name: str) -> None:
        """Serialization preserves the layout, coordinates, periods, hashes, and derived graphs."""
        geometry = LatticeGeometry.honeycomb_plaquettes(2, 2, periodic_x=periodic, periodic_y=periodic)
        if format_name == "json":
            restored = LatticeGeometry.from_json(geometry.to_json())
            path = tmp_path / "geometry.lattice_geometry.json"
            restored.to_json_file(path)
            restored = LatticeGeometry.from_json_file(path)
        elif format_name == "hdf5":
            path = tmp_path / "geometry.lattice_geometry.h5"
            geometry.to_hdf5_file(path)
            restored = LatticeGeometry.from_hdf5_file(path)
        else:
            restored = pickle.loads(pickle.dumps(geometry))

        assert restored.to_json() == geometry.to_json()
        assert restored.num_sites == geometry.num_sites
        np.testing.assert_array_equal(restored.positions, geometry.positions)
        if geometry.periods is None:
            assert restored.periods is None
        else:
            np.testing.assert_array_equal(restored.periods, geometry.periods)
        assert restored.content_hash() == geometry.content_hash()
        assert (
            LatticeGraph.from_geometry(restored).content_hash() == LatticeGraph.from_geometry(geometry).content_hash()
        )


class TestSelectedLatticeGraph:
    """Check edge labels and graph persistence."""

    def test_from_geometry_materializes_selected_union(self) -> None:
        """Selected shells yield a deduplicated weighted, flavored union without mutating geometry."""
        geometry = LatticeGeometry.chain(3)
        definitions = [BondFlavorDefinition(shell, np.array([1.0, 0.0]), 1000 + shell) for shell in (1, 2)]
        graph = LatticeGraph.from_geometry(
            geometry, [2, 1, 99, 2], bond_flavors=definitions, weight=2.5, tolerance=1e-9
        )

        labels = graph.edge_labels
        assert {pair: label.shell for pair, label in labels.items()} == {(0, 1): 1, (0, 2): 2, (1, 2): 1}
        assert all(label.flavor == 1000 + label.shell for label in labels.values())
        np.testing.assert_array_equal(graph.adjacency_matrix(), 2.5 * (np.ones((3, 3)) - np.eye(3)))

    def test_from_geometry_forwards_coloring_seed(self) -> None:
        """The coloring seed only changes the greedy coloring, and seed 0 is the default."""
        geometry = LatticeGeometry.triangular(4, 4, periodic_x=True, periodic_y=True)
        default = LatticeGraph.from_geometry(geometry, [1])
        seeded = LatticeGraph.from_geometry(geometry, [1], coloring_seed=7)

        assert LatticeGraph.from_geometry(geometry, [1], coloring_seed=0).edge_coloring == default.edge_coloring
        assert LatticeGraph.from_geometry(geometry, [1], coloring_seed=7).edge_coloring == seeded.edge_coloring
        assert seeded.edge_labels == default.edge_labels
        assert seeded.edge_coloring.keys() == default.edge_coloring.keys()

    def test_custom_graphs_accept_edge_labels(self) -> None:
        """Adjacency-built graphs retain user shells and flavors through data operations."""
        labels = {(0, 1): EdgeLabel(1, flavor=10), (1, 2): EdgeLabel(1), (0, 2): EdgeLabel(2, flavor=20)}
        adjacency = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.0]])
        dense = LatticeGraph.from_dense_matrix(adjacency, edge_labels=labels)
        upper = LatticeGraph({pair: float(adjacency[pair]) for pair in labels}, edge_labels=labels)

        assert dense.edge_labels == labels
        assert labels[1, 2].flavor is None
        sparse = LatticeGraph.from_sparse_matrix(dense.sparse_adjacency_matrix(), edge_labels=labels)
        assert sparse.content_hash() == dense.content_hash()
        assert LatticeGraph.make_bidirectional(upper).edge_labels == labels
        assert LatticeGraph.from_json(dense.to_json()).content_hash() == dense.content_hash()
        for invalid in ({(0, 1): EdgeLabel(1)}, {**labels, (0, 1): EdgeLabel(0)}):
            with pytest.raises(ValueError, match="edge label"):
                LatticeGraph.from_dense_matrix(adjacency, edge_labels=invalid)

    def test_graph_permutation_preserves_edge_labels(self) -> None:
        """Valid permutations preserve edge labels; repeated indices are rejected."""
        graph = LatticeGraph.from_geometry(LatticeGeometry.chain(3), [1, 2], weight=2.5)
        permuted = LatticeGraph.permute(graph, [2, 0, 1])
        restored = LatticeGraph.from_json(permuted.to_json())
        assert restored.content_hash() == permuted.content_hash()
        assert {pair: label.shell for pair, label in permuted.edge_labels.items()} == {(0, 1): 2, (0, 2): 1, (1, 2): 1}
        with pytest.raises(ValueError, match="Permutation"):
            LatticeGraph.permute(graph, [0, 0])

    @pytest.mark.parametrize("empty", [False, True])
    @pytest.mark.parametrize("format_name", ["json", "hdf5", "pickle"])
    def test_round_trip_preserves_edge_labels(self, tmp_path: Path, empty: bool, format_name: str) -> None:
        """Serialization retains edge labels and weights."""
        geometry = LatticeGeometry.chain(1) if empty else LatticeGeometry.chain(3, periodic=True)
        definitions = [BondFlavorDefinition(1, np.array([1.0, 0.0]), 1000)]
        graph = LatticeGraph.from_geometry(geometry, [1], bond_flavors=definitions, weight=2.0)
        if format_name == "json":
            restored = LatticeGraph.from_json(graph.to_json())
        elif format_name == "hdf5":
            path = tmp_path / "graph.lattice_graph.h5"
            graph.to_hdf5_file(path)
            restored = LatticeGraph.from_hdf5_file(path)
        else:
            restored = pickle.loads(pickle.dumps(graph))

        assert restored.num_sites == graph.num_sites
        assert restored.edge_labels == graph.edge_labels
        assert len(graph.edge_labels) == (0 if empty else 3)
        np.testing.assert_array_equal(restored.adjacency_matrix(), graph.adjacency_matrix())
        assert restored.content_hash() == graph.content_hash()
