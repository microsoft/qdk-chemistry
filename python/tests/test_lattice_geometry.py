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

from qdk_chemistry.data import BondClass, BondFlavorDefinition, LatticeGeometry, LatticeGraph, NeighborConnection

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
    """Check selected connections, adjacency projection, and graph persistence."""

    def test_from_geometry_materializes_selected_union(self) -> None:
        """Selected shells yield a deduplicated weighted, flavored union without mutating geometry."""
        geometry = LatticeGeometry.chain(3)
        definitions = [BondFlavorDefinition(shell, np.array([1.0, 0.0]), 1000 + shell) for shell in (1, 2)]
        graph = LatticeGraph.from_geometry(
            geometry, [2, 1, 99, 2], bond_flavors=definitions, weight=2.5, tolerance=1e-9
        )

        assert graph.selected_shells == [1, 2, 99]
        assert len(graph.connections) == 3
        assert all(connection.weight == 2.5 for connection in graph.connections)
        assert all(connection.flavor == 1000 + connection.bond_class.shell for connection in graph.connections)
        np.testing.assert_array_equal(graph.adjacency_matrix(), 2.5 * (np.ones((3, 3)) - np.eye(3)))
        assert graph.geometry is not None
        np.testing.assert_array_equal(graph.geometry.positions, geometry.positions)
        for connection in geometry.neighbor_connections([1, 2]):
            assert connection.flavor is None
            assert connection.weight == 1.0

    def test_custom_record_constructors_and_canonicalization(self) -> None:
        """Canonicalizing reversed bonds preserves metadata and sums weights without mutating inputs."""
        bond_class = BondClass(shell=3, orientation=7, axis=np.array([1.0, 0.0]))
        default = NeighborConnection(0, 2, bond_class, np.array([2.0, 0.0]), (0, 0))
        reverse = NeighborConnection(2, 0, bond_class, np.array([-2.0, 0.0]), (-1, 0), flavor=1000, weight=-2.5)
        graph = LatticeGraph.from_connections(3, [reverse, default], geometry=None, selected_shells=[99, 1, 99])

        assert default.flavor is None
        assert default.weight == 1.0
        assert graph.geometry is None
        assert graph.selected_shells == [1, 3, 99]
        assert len(graph.connections) == 2
        first, second = graph.connections
        assert (first.site_i, first.site_j, first.flavor, first.weight) == (0, 2, None, 1.0)
        assert (second.site_i, second.site_j, second.flavor, second.weight) == (0, 2, 1000, -2.5)
        assert tuple(second.image_shift) == (1, 0)
        assert (second.bond_class.shell, second.bond_class.orientation) == (3, 7)
        np.testing.assert_array_equal(second.bond_class.axis, [1.0, 0.0])
        np.testing.assert_array_equal(second.displacement, [2.0, 0.0])
        assert graph.weight(0, 2) == -1.5
        assert graph.weight(2, 0) == -1.5
        assert reverse.site_i == 2

    def test_graph_permutation_preserves_resolved_records(self) -> None:
        """Valid permutations preserve resolved bonds and geometry; repeated indices are rejected."""
        graph = LatticeGraph.from_geometry(LatticeGeometry.chain(2, periodic=True), [1], weight=2.5)
        permuted = LatticeGraph.permute(graph, [1, 0])
        restored = LatticeGraph.from_json(permuted.to_json())
        assert restored.content_hash() == permuted.content_hash()
        _assert_same_connections(LatticeGraph.permute(permuted, [1, 0]).connections, graph.connections)
        np.testing.assert_array_equal(permuted.geometry.positions, graph.geometry.positions[[1, 0]])
        with pytest.raises(ValueError, match="Permutation"):
            LatticeGraph.permute(graph, [0, 0])

    @pytest.mark.parametrize("empty", [False, True])
    @pytest.mark.parametrize("format_name", ["json", "hdf5", "pickle"])
    def test_round_trip_preserves_selected_records(self, tmp_path: Path, empty: bool, format_name: str) -> None:
        """Serialization retains selected shells, physical images, weights, flavors, and optional geometry."""
        geometry = None if empty else LatticeGeometry.chain(2, periodic=True)
        bond_class = BondClass(1, 0, np.array([1.0, 0.0]))
        connections = (
            []
            if empty
            else [
                NeighborConnection(0, 1, bond_class, np.array([1.0, 0.0]), (0, 0), flavor=1000, weight=2.0),
                NeighborConnection(0, 1, bond_class, np.array([-1.0, 0.0]), (-1, 0), flavor=1001, weight=-3.0),
            ]
        )
        graph = LatticeGraph.from_connections(2, connections, geometry=geometry, selected_shells=[1, 99])
        if format_name == "json":
            restored = LatticeGraph.from_json(graph.to_json())
        elif format_name == "hdf5":
            path = tmp_path / "graph.lattice_graph.h5"
            graph.to_hdf5_file(path)
            restored = LatticeGraph.from_hdf5_file(path)
        else:
            restored = pickle.loads(pickle.dumps(graph))

        assert restored.num_sites == graph.num_sites
        assert restored.selected_shells == [1, 99]
        _assert_same_connections(restored.connections, graph.connections)
        np.testing.assert_array_equal(restored.adjacency_matrix(), graph.adjacency_matrix())
        assert restored.content_hash() == graph.content_hash()
        if geometry is None:
            assert restored.geometry is None
        else:
            assert restored.geometry is not None
            np.testing.assert_array_equal(restored.geometry.positions, geometry.positions)
            np.testing.assert_array_equal(restored.geometry.periods, geometry.periods)
