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

from qdk_chemistry.data import BondFlavorDefinition, LatticeGeometry, LatticeGraph, NeighborConnection

if TYPE_CHECKING:
    from pathlib import Path


def _assert_same_connections(actual: list[NeighborConnection], expected: list[NeighborConnection]) -> None:
    """Compare physical records without relying on Python wrapper identity."""
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert (left.site_i, left.site_j, left.flavor, left.weight) == (
            right.site_i,
            right.site_j,
            right.flavor,
            right.weight,
        )
        assert (left.bond_class.shell, left.bond_class.orientation) == (
            right.bond_class.shell,
            right.bond_class.orientation,
        )
        assert tuple(left.image_shift) == tuple(right.image_shift)
        np.testing.assert_allclose(left.bond_class.axis, right.bond_class.axis, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(left.displacement, right.displacement, atol=1e-12, rtol=0.0)


class TestLatticeGeometry:
    """Check neighbor discovery, geometry validation, and persistence."""

    def test_neighbor_queries_validate_shell_indices(self) -> None:
        """Neighbor queries reject zero, negative, and nonintegral shell indices."""
        geometry = LatticeGeometry.chain(3)
        with pytest.raises(ValueError, match="m must be > 0"):
            geometry.mth_nearest_neighbors(0)
        with pytest.raises(ValueError, match="m must be > 0"):
            geometry.nearest_neighbor_shells([0])
        with pytest.raises(ValueError, match="shell index must be > 0"):
            geometry.neighbor_connections([0])
        for shell in (-1, 1.5):
            with pytest.raises(TypeError):
                geometry.mth_nearest_neighbors(shell)
            with pytest.raises(TypeError):
                geometry.nearest_neighbor_shells([shell])
            with pytest.raises(TypeError):
                geometry.neighbor_connections([shell])

    def test_cartesian_geometry_copies_inputs_and_properties(self) -> None:
        """Mutating input or returned arrays leaves stored positions and periods unchanged."""
        positions = np.array([[0.0, 0.0], [1.0, 0.0]])
        periods = np.array([[2.0, 0.0], [0.0, 2.0]])
        geometry = LatticeGeometry(positions, periods=periods)
        positions[:] = 7.0
        periods[:] = 7.0
        copied_positions = geometry.positions
        copied_periods = geometry.periods
        assert copied_periods is not None
        copied_positions[:] = -7.0
        copied_periods[:] = -7.0

        np.testing.assert_array_equal(geometry.positions, [[0.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_equal(geometry.periods, [[2.0, 0.0], [0.0, 2.0]])

    @pytest.mark.parametrize("kind", ["open", "periodic", "empty"])
    @pytest.mark.parametrize("format_name", ["json", "hdf5", "pickle"])
    def test_round_trip_preserves_geometry(self, tmp_path: Path, kind: str, format_name: str) -> None:
        """Serialization preserves coordinates, periods, hashes, and neighbor connections."""
        geometry = (
            LatticeGeometry(np.empty((0, 2)))
            if kind == "empty"
            else LatticeGeometry.honeycomb_plaquettes(
                2, 2, periodic_x=kind == "periodic", periodic_y=kind == "periodic"
            )
        )
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

        assert restored.num_sites == geometry.num_sites
        np.testing.assert_array_equal(restored.positions, geometry.positions)
        if geometry.periods is None:
            assert restored.periods is None
        else:
            np.testing.assert_array_equal(restored.periods, geometry.periods)
        assert restored.content_hash() == geometry.content_hash()
        _assert_same_connections(restored.neighbor_connections([1, 2, 3]), geometry.neighbor_connections([1, 2, 3]))


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
        for connection in geometry.neighbor_connections([1, 2]):
            assert connection.flavor is None
            assert connection.weight == 1.0

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
