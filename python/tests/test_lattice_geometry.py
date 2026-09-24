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

from .reference_tolerances import float_comparison_absolute_tolerance

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

    def test_geometric_neighbor_shells(self) -> None:
        """Euclidean shells return sorted, unique pairs and omit unavailable neighbors."""
        geometry = LatticeGeometry.square(3, 3)
        shells = geometry.nearest_neighbor_shells([2, 1, 99, 1])

        assert geometry.periods is None
        assert geometry.mth_nearest_neighbors(99) == []
        assert shells[99] == []
        assert (0, 1) in shells[1]
        assert (0, 4) in shells[2]
        assert (0, 2) not in shells[2]
        assert len(shells[1]) == 12
        assert len(shells[2]) == 8
        for shell in (1, 2):
            assert shells[shell] == sorted(set(shells[shell]))
            assert shells[shell] == geometry.mth_nearest_neighbors(shell)

    def test_honeycomb_positions_define_one_cell_shells(self) -> None:
        """A honeycomb unit cell and hexagon have the expected distance-defined shells."""
        unit_cell = LatticeGraph.honeycomb(1, 1)
        assert unit_cell.num_sites == 2
        assert unit_cell.num_edges == 1
        assert unit_cell.geometry is not None
        assert unit_cell.geometry.mth_nearest_neighbors(1) == [(0, 1)]

        geometry = LatticeGeometry.honeycomb_plaquettes(1, 1)
        positions = geometry.positions
        assert positions.shape == (6, 2)
        shells = geometry.nearest_neighbor_shells([1, 2, 3])
        assert [len(shells[shell]) for shell in (1, 2, 3)] == [6, 6, 3]
        for shell, expected_distance in ((1, 1.0), (2, np.sqrt(3.0)), (3, 2.0)):
            distances = [np.linalg.norm(positions[site_j] - positions[site_i]) for site_i, site_j in shells[shell]]
            assert distances == pytest.approx(
                [expected_distance] * len(distances), abs=float_comparison_absolute_tolerance
            )

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

    @pytest.mark.parametrize("tolerance", [0.0, -1.0, np.inf, np.nan])
    def test_neighbor_queries_validate_tolerance(self, tolerance: float) -> None:
        """Neighbor queries reject nonpositive or nonfinite tolerances."""
        geometry = LatticeGeometry.chain(3)
        with pytest.raises(ValueError, match="tolerance must be positive"):
            geometry.mth_nearest_neighbors(1, tolerance=tolerance)
        with pytest.raises(ValueError, match="tolerance must be positive"):
            geometry.nearest_neighbor_shells([1], tolerance=tolerance)
        with pytest.raises(ValueError, match="tolerance must be positive"):
            geometry.neighbor_connections([1], tolerance=tolerance)

    def test_periodic_queries_retain_physical_images(self) -> None:
        """Periodic bonds retain distinct image shifts while pair-only queries are rejected."""
        geometry = LatticeGeometry.chain(2, periodic=True)
        np.testing.assert_array_equal(geometry.periods, [[2.0, 0.0]])
        connections = geometry.neighbor_connections([1])

        assert len(connections) == 2
        assert {(connection.site_i, connection.site_j) for connection in connections} == {(0, 1)}
        assert {tuple(connection.image_shift) for connection in connections} == {(-1, 0), (0, 0)}
        for connection in connections:
            assert connection.flavor is None
            assert connection.weight == 1.0
            np.testing.assert_array_equal(connection.bond_class.axis, [1.0, 0.0])
            np.testing.assert_array_equal(connection.displacement, [1.0 + 2.0 * connection.image_shift[0], 0.0])
        with pytest.raises(RuntimeError, match="support open lattices only"):
            geometry.mth_nearest_neighbors(1)
        with pytest.raises(RuntimeError, match="support open lattices only"):
            geometry.nearest_neighbor_shells([1, 2])

    def test_periodic_self_image_is_canonical(self) -> None:
        """A one-site periodic chain has one positively oriented self-image bond."""
        connections = LatticeGeometry.chain(1, periodic=True).neighbor_connections([1])
        assert len(connections) == 1
        connection = connections[0]
        assert (connection.site_i, connection.site_j) == (0, 0)
        assert tuple(connection.image_shift) == (1, 0)
        np.testing.assert_array_equal(connection.displacement, [1.0, 0.0])

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

    @pytest.mark.parametrize(
        ("positions", "periods", "message"),
        [
            (np.zeros((2, 0)), None, "positions"),
            (np.array([[np.nan, 0.0]]), None, "finite"),
            (np.zeros((1, 2)), np.zeros((1, 2)), "nonzero"),
            (np.zeros((1, 2)), np.ones((3, 2)), "Periodic vectors"),
            (np.zeros((1, 2)), np.array([[1.0, 0.0], [2.0, 0.0]]), "independent"),
        ],
    )
    def test_invalid_cartesian_geometry(self, positions: np.ndarray, periods: np.ndarray | None, message: str) -> None:
        """Geometry rejects invalid coordinates and malformed or degenerate periodic vectors."""
        with pytest.raises(ValueError, match=message):
            LatticeGeometry(positions, periods=periods)

    def test_non_planar_geometry_is_data_only(self) -> None:
        """Geometry and records accept any positive dimension, while neighbor searches stay two-dimensional."""
        axis = np.array([0.0, 0.0, 1.0])
        geometry = LatticeGeometry(np.array([[0.0, 0.0, 0.0], axis]), periods=np.array([[0.0, 0.0, 2.0]]))
        assert geometry.dimension == 3
        restored = LatticeGeometry.from_json(geometry.to_json())
        np.testing.assert_array_equal(restored.positions, geometry.positions)
        assert restored.content_hash() == geometry.content_hash()
        with pytest.raises(RuntimeError, match="two-dimensional"):
            geometry.neighbor_connections([1])

        record = NeighborConnection(0, 1, BondClass(1, 0, axis), axis, [0, 0, 0])
        graph = LatticeGraph.from_connections(2, [record], geometry=geometry)
        assert list(graph.connections[0].image_shift) == [0, 0, 0]
        np.testing.assert_array_equal(graph.connections[0].displacement, axis)
        with pytest.raises(ValueError, match="dimension"):
            LatticeGraph.from_connections(2, [record], geometry=LatticeGeometry(np.zeros((2, 2))))

    @pytest.mark.parametrize("periodic", [False, True])
    def test_permuted_builtin_and_cartesian_queries_agree(self, periodic: bool) -> None:
        """Permuted built-in geometry yields the same connections as reordered Cartesian data."""
        original = LatticeGeometry.honeycomb_plaquettes(2, 2, periodic_x=periodic, periodic_y=periodic)
        path = list(reversed(range(original.num_sites)))
        geometry = LatticeGeometry.permute(original, path)
        cartesian = LatticeGeometry(original.positions[path], periods=original.periods)

        np.testing.assert_array_equal(geometry.positions, original.positions[path])
        _assert_same_connections(geometry.neighbor_connections([3, 1, 2]), cartesian.neighbor_connections([1, 2, 3]))

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

    @pytest.mark.parametrize(
        ("factory", "dimensions"),
        [
            ("chain", (4,)),
            ("square", (3, 3)),
            ("triangular", (3, 3)),
            ("honeycomb", (2, 2)),
            ("honeycomb_plaquettes", (1, 1)),
            ("kagome", (2, 2)),
        ],
    )
    def test_legacy_factories_select_first_shell(self, factory: str, dimensions: tuple[int, ...]) -> None:
        """Legacy factories select unflavored first-shell bonds and project their weights to adjacency."""
        graph = getattr(LatticeGraph, factory)(*dimensions, t=2.5)
        assert isinstance(graph.geometry, LatticeGeometry)
        assert graph.geometry.num_sites == graph.num_sites
        assert graph.selected_shells == [1]
        assert {connection.bond_class.shell for connection in graph.connections} == {1}
        assert all(connection.flavor is None for connection in graph.connections)
        assert graph.edge_coloring is not None
        projected = np.zeros((graph.num_sites, graph.num_sites))
        for connection in graph.connections:
            projected[connection.site_i, connection.site_j] += connection.weight
            projected[connection.site_j, connection.site_i] += connection.weight
        np.testing.assert_array_equal(graph.adjacency_matrix(), projected)

    def test_legacy_two_site_periodic_chain_keeps_weight_and_coloring(self) -> None:
        """Two periodic images share one edge's weight and retain its legacy coloring."""
        graph = LatticeGraph.chain(2, periodic=True, t=2.5)
        assert graph.num_edges == 1
        assert graph.edge_coloring == {(0, 1): 0}
        assert graph.selected_shells == [1]
        assert len(graph.connections) == 2
        assert [connection.weight for connection in graph.connections] == [1.25, 1.25]
        np.testing.assert_array_equal(graph.adjacency_matrix(), [[0.0, 2.5], [2.5, 0.0]])

    def test_graph_exposes_geometry_not_geometric_queries(self) -> None:
        """Geometric queries live on geometry, while adjacency-only graphs lack connection metadata."""
        graph = LatticeGraph.chain(3)
        assert isinstance(graph.geometry, LatticeGeometry)
        assert graph.geometry.mth_nearest_neighbors(2) == [(0, 2)]
        for removed in (
            "positions",
            "mth_nearest_neighbors",
            "nearest_neighbor_shells",
            "neighbor_connections",
            "bond_flavor_definitions",
        ):
            assert not hasattr(graph, removed)
        adjacency_only = LatticeGraph.from_dense_matrix(graph.adjacency_matrix())
        assert adjacency_only.geometry is None
        assert adjacency_only.connections == []
        assert adjacency_only.selected_shells == []

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

    @pytest.mark.parametrize("shells", [[], [99], [99, 1, 99]])
    def test_empty_selections_remain_explicit(self, shells: list[int]) -> None:
        """Empty neighbor sets retain explicit shell selections and zero adjacency."""
        graph = LatticeGraph.from_geometry(LatticeGeometry.chain(1), shells)
        assert graph.selected_shells == sorted(set(shells))
        assert graph.connections == []
        np.testing.assert_array_equal(graph.adjacency_matrix(), np.zeros((1, 1)))

    def test_relabeling_does_not_select_or_discover_shells(self) -> None:
        """Flavor replacement clears unmatched labels without changing selected bonds or geometry."""
        graph = LatticeGraph.from_geometry(LatticeGeometry.chain(3), [1, 99], weight=-2.0)
        flavored = graph.with_bond_flavors([BondFlavorDefinition(1, np.array([1.0, 0.0]), 1000)])
        relabeled = flavored.with_bond_flavors([BondFlavorDefinition(2, np.array([1.0, 0.0]), 1001)])

        assert all(connection.flavor == 1000 for connection in flavored.connections)
        assert relabeled.selected_shells == graph.selected_shells
        np.testing.assert_array_equal(relabeled.adjacency_matrix(), graph.adjacency_matrix())
        _assert_same_connections(relabeled.connections, graph.connections)
        assert relabeled.geometry is not None
        assert graph.geometry is not None
        assert relabeled.geometry.content_hash() == graph.geometry.content_hash()

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

    @pytest.mark.parametrize("shell", [1, 2])
    def test_duplicate_canonical_records_are_rejected(self, shell: int) -> None:
        """Reversed copies of the same physical bond are rejected even with different shell labels."""
        first = NeighborConnection(0, 1, BondClass(1, 0, np.array([1.0, 0.0])), np.array([1.0, 0.0]), (0, 0))
        reverse = NeighborConnection(1, 0, BondClass(shell, 0, np.array([1.0, 0.0])), np.array([-1.0, 0.0]), (0, 0))
        with pytest.raises(ValueError, match="Duplicate canonical lattice connection"):
            LatticeGraph.from_connections(2, [first, reverse])

    def test_custom_geometry_site_count_must_match(self) -> None:
        """Connection graphs reject geometry with a different number of sites."""
        with pytest.raises(ValueError, match="site counts must match"):
            LatticeGraph.from_connections(2, [], geometry=LatticeGeometry.chain(3))

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
