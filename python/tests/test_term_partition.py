"""Tests for the ``TermPartition`` infrastructure and ``term_grouper`` algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.algorithms import registry
from qdk_chemistry.data import (
    FlatPartition,
    LatticeGraph,
    LayeredPartition,
    QubitHamiltonian,
    TermPartition,
)
from qdk_chemistry.utils.model_hamiltonians import (
    create_heisenberg_hamiltonian,
    create_ising_hamiltonian,
)

# ---------------------------------------------------------------------------
# TermPartition data classes
# ---------------------------------------------------------------------------


class TestFlatPartition:
    def test_construction_normalises_to_tuples_of_ints(self):
        """Construction normalises to tuples of ints."""
        p = FlatPartition(strategy="commuting", groups=[[0, 1, 2], [3, 4]])
        assert isinstance(p.groups, tuple)
        assert all(isinstance(g, tuple) for g in p.groups)
        assert all(isinstance(i, int) for g in p.groups for i in g)
        assert p.groups == ((0, 1, 2), (3, 4))

    def test_num_groups(self):
        """Num groups."""
        p = FlatPartition(strategy="x", groups=[[0], [1, 2], [3]])
        assert p.num_groups == 3

    def test_all_indices(self):
        """All indices."""
        p = FlatPartition(strategy="x", groups=[[2, 1], [0]])
        assert p.all_indices() == [2, 1, 0]

    def test_is_subclass_of_term_partition(self):
        """Is subclass of term partition."""
        assert issubclass(FlatPartition, TermPartition)


class TestLayeredPartition:
    def test_construction(self):
        """Construction."""
        p = LayeredPartition(
            strategy="geometry_coloring",
            groups=[[[0, 1], [2, 3]], [[4]]],
        )
        assert p.groups == (((0, 1), (2, 3)), ((4,),))

    def test_num_groups_and_layers(self):
        """Num groups and layers."""
        p = LayeredPartition(
            strategy="x",
            groups=[[[0]], [[1], [2]], [[3], [4], [5]]],
        )
        assert p.num_groups == 3
        assert p.num_layers(0) == 1
        assert p.num_layers(1) == 2
        assert p.num_layers(2) == 3

    def test_all_indices_flattens_in_order(self):
        """All indices flattens in order."""
        p = LayeredPartition(strategy="x", groups=[[[0, 1], [2]], [[3, 4]]])
        assert p.all_indices() == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# QubitHamiltonian.term_partition property
# ---------------------------------------------------------------------------


class TestQubitHamiltonianTermPartition:
    def test_default_is_none(self):
        """Default is none."""
        qh = QubitHamiltonian(["XX", "ZZ"], np.array([0.1, 0.2]))
        assert qh.term_partition is None

    def test_round_trip_flat(self):
        """Round trip flat."""
        partition = FlatPartition(strategy="commuting", groups=[[0], [1]])
        qh = QubitHamiltonian(["XX", "ZZ"], np.array([0.1, 0.2]), term_partition=partition)
        assert qh.term_partition is partition

    def test_round_trip_layered(self):
        """Round trip layered."""
        partition = LayeredPartition(strategy="geometry_coloring", groups=[[[0, 1]]])
        qh = QubitHamiltonian(["XX", "ZZ"], np.array([0.1, 0.2]), term_partition=partition)
        assert qh.term_partition is partition

    def test_to_interleaved_resets_partition(self):
        """To interleaved resets partition."""
        partition = FlatPartition(strategy="commuting", groups=[[0, 1, 2, 3]])
        qh = QubitHamiltonian(
            ["XXII", "YYII", "IIZZ", "IIXX"],
            np.array([0.1, 0.2, 0.3, 0.4]),
            term_partition=partition,
        )
        out = qh.to_interleaved(n_spatial=2)
        assert out.term_partition is None


# ---------------------------------------------------------------------------
# term_grouper algorithm registry integration
# ---------------------------------------------------------------------------


class TestTermGrouperRegistry:
    def test_available_strategies(self):
        """Available strategies."""
        names = registry.available("term_grouper")
        assert {"commuting", "qubit_wise_commuting", "identity"} <= set(names)

    def test_default_strategy_is_commuting(self):
        """Default strategy is commuting."""
        grouper = registry.create("term_grouper")
        assert grouper.name() == "commuting"

    @pytest.mark.parametrize("strategy", ["commuting", "qubit_wise_commuting", "identity"])
    def test_returns_new_hamiltonian_with_partition(self, strategy):
        """Returns new hamiltonian with partition."""
        qh = QubitHamiltonian(["XX", "YY", "ZZ"], np.array([1.0, 2.0, 3.0]))
        grouper = registry.create("term_grouper", strategy)
        out = grouper.run(qh)
        assert out is not qh
        assert isinstance(out.term_partition, FlatPartition)
        assert out.term_partition.strategy == strategy

    def test_partition_indices_cover_all_terms_exactly_once(self):
        """Partition indices cover all terms exactly once."""
        qh = QubitHamiltonian(
            ["XIII", "IXII", "IIXI", "IIIX", "ZIII", "IZII"],
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        )
        for strategy in ("commuting", "qubit_wise_commuting", "identity"):
            grouper = registry.create("term_grouper", strategy)
            out = grouper.run(qh)
            indices = out.term_partition.all_indices()
            assert sorted(indices) == list(range(len(qh.pauli_strings)))

    def test_identity_strategy_one_term_per_group(self):
        """Identity strategy one term per group."""
        qh = QubitHamiltonian(["XX", "YY", "ZZ"], np.array([1.0, 2.0, 3.0]))
        out = registry.create("term_grouper", "identity").run(qh)
        assert out.term_partition.num_groups == len(qh.pauli_strings)
        assert all(len(g) == 1 for g in out.term_partition.groups)

    def test_commuting_groups_globally_commute(self):
        # XX and YY commute (XY * YX = -ZZ * -ZZ = ZZ^2 = I; and YX * XY = ZZ),
        # ZZ commutes with both.  So all three should land in the same group.
        """Commuting groups globally commute."""
        qh = QubitHamiltonian(["XX", "YY", "ZZ"], np.array([1.0, 1.0, 1.0]))
        out = registry.create("term_grouper", "commuting").run(qh)
        assert out.term_partition.num_groups == 1

    def test_qwc_separates_paulis_that_only_globally_commute(self):
        # XX and YY are NOT qubit-wise commuting, even though they globally commute.
        """Qwc separates paulis that only globally commute."""
        qh = QubitHamiltonian(["XX", "YY"], np.array([1.0, 1.0]))
        out = registry.create("term_grouper", "qubit_wise_commuting").run(qh)
        assert out.term_partition.num_groups == 2


# ---------------------------------------------------------------------------
# LatticeGraph.edge_coloring overlay
# ---------------------------------------------------------------------------


class TestLatticeEdgeColoring:
    def test_chain_two_colors(self):
        """Chain two colors."""
        lat = LatticeGraph.chain(4, periodic=True)
        coloring = lat.edge_coloring
        assert coloring is not None
        assert len(set(coloring.values())) == 2

    def test_returns_dict_or_none(self):
        """Returns dict or none."""
        lat = LatticeGraph.chain(3, periodic=False)
        coloring = lat.edge_coloring
        assert isinstance(coloring, dict)


# ---------------------------------------------------------------------------
# create_*_hamiltonian populates term_partition
# ---------------------------------------------------------------------------


class TestModelHamiltonianTermPartition:
    def test_heisenberg_populates_layered_partition(self):
        """Heisenberg populates layered partition."""
        lat = LatticeGraph.chain(4, periodic=True)
        ham = create_heisenberg_hamiltonian(lat, jx=1.0, jy=1.0, jz=1.0)
        assert isinstance(ham.term_partition, LayeredPartition)
        assert ham.term_partition.strategy == "geometry_coloring"
        # Indices reach every term exactly once.
        assert sorted(ham.term_partition.all_indices()) == list(range(len(ham.pauli_strings)))

    def test_ising_populates_layered_partition(self):
        """Ising populates layered partition."""
        lat = LatticeGraph.chain(4, periodic=True)
        ham = create_ising_hamiltonian(lat, j=1.0, h=0.5)
        assert isinstance(ham.term_partition, LayeredPartition)
        assert ham.term_partition.strategy == "geometry_coloring"

    def test_include_term_groups_false_disables_partition(self):
        """Include term groups false disables partition."""
        lat = LatticeGraph.chain(4, periodic=True)
        ham = create_heisenberg_hamiltonian(lat, jx=1.0, jy=1.0, jz=1.0, include_term_groups=False)
        assert ham.term_partition is None


# ---------------------------------------------------------------------------
# Trotter consumes term_partition
# ---------------------------------------------------------------------------


class TestTrotterConsumesTermPartition:
    def test_trotter_runs_with_partitioned_hamiltonian(self):
        """Trotter runs with partitioned hamiltonian."""
        lat = LatticeGraph.chain(4, periodic=True)
        ham = create_heisenberg_hamiltonian(lat, jx=1.0, jy=1.0, jz=1.0)
        trotter = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter.settings().update({"order": 2, "time": 0.5})
        unitary = trotter.run(ham)
        assert unitary is not None

    def test_trotter_runs_without_partition(self):
        # Falls back to treating each term as its own group.
        """Trotter runs without partition."""
        ham = QubitHamiltonian(["XXII", "IXXI", "IIXX", "ZIII"], np.array([1.0, 1.0, 1.0, 0.5]))
        assert ham.term_partition is None
        trotter = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter.settings().update({"time": 0.5})
        unitary = trotter.run(ham)
        assert unitary is not None

    def test_partition_produces_smaller_or_equal_step_count_at_order_2(self):
        # With group sorting + schedule reduction, populating the partition
        # should never produce more step terms than the ungrouped fallback.
        """Partition produces smaller or equal step count at order 2."""
        lat = LatticeGraph.chain(4, periodic=True)
        with_groups = create_heisenberg_hamiltonian(lat, jx=1.0, jy=1.0, jz=1.0, include_term_groups=True)
        without_groups = create_heisenberg_hamiltonian(lat, jx=1.0, jy=1.0, jz=1.0, include_term_groups=False)
        assert with_groups.term_partition is not None
        assert without_groups.term_partition is None

        trotter = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter.settings().update({"order": 2, "num_divisions": 1, "time": 1.0})
        grouped_steps = len(trotter.run(with_groups).get_container().step_terms)

        trotter2 = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter2.settings().update({"order": 2, "num_divisions": 1, "time": 1.0})
        ungrouped_steps = len(trotter2.run(without_groups).get_container().step_terms)

        assert grouped_steps <= ungrouped_steps

    def test_trotter_runs_with_flat_partition(self):
        # Take a partitioned Hamiltonian and overwrite term_partition with a
        # FlatPartition (via the term_grouper algorithm), then drive Trotter.
        """Trotter runs with flat partition."""
        lat = LatticeGraph.chain(4, periodic=True)
        ham = create_heisenberg_hamiltonian(lat, jx=1.0, jy=1.0, jz=1.0)
        flat = registry.create("term_grouper", "commuting").run(ham)
        assert isinstance(flat.term_partition, FlatPartition)

        trotter = registry.create("hamiltonian_unitary_builder", "trotter")
        trotter.settings().update({"order": 2, "time": 0.5})
        unitary = trotter.run(flat)
        assert unitary is not None


# ---------------------------------------------------------------------------
# QubitHamiltonian round-trips term_partition through JSON / HDF5
# ---------------------------------------------------------------------------


class TestTermPartitionSerialisation:
    def test_flat_partition_to_json_round_trip(self):
        """Flat partition to json round trip."""
        partition = FlatPartition(strategy="commuting", groups=[[0, 2], [1]])
        data = partition.to_json()
        assert data["kind"] == "flat"
        restored = TermPartition.from_json(data)
        assert isinstance(restored, FlatPartition)
        assert restored == partition

    def test_layered_partition_to_json_round_trip(self):
        """Layered partition to json round trip."""
        partition = LayeredPartition(strategy="geometry_coloring", groups=[[[0, 1], [2]], [[3]]])
        data = partition.to_json()
        assert data["kind"] == "layered"
        restored = TermPartition.from_json(data)
        assert isinstance(restored, LayeredPartition)
        assert restored == partition

    def test_qubit_hamiltonian_json_round_trip_preserves_partition(self):
        """Qubit hamiltonian json round trip preserves partition."""
        partition = FlatPartition(strategy="commuting", groups=[[0, 1], [2]])
        ham = QubitHamiltonian(["XX", "YY", "ZZ"], np.array([0.1, 0.2, 0.3]), term_partition=partition)
        restored = QubitHamiltonian.from_json(ham.to_json())
        assert isinstance(restored.term_partition, FlatPartition)
        assert restored.term_partition == partition

    def test_qubit_hamiltonian_json_round_trip_with_no_partition(self):
        """Qubit hamiltonian json round trip with no partition."""
        ham = QubitHamiltonian(["XX", "ZZ"], np.array([0.1, 0.2]))
        restored = QubitHamiltonian.from_json(ham.to_json())
        assert restored.term_partition is None

    def test_qubit_hamiltonian_hdf5_round_trip_preserves_partition(self, tmp_path):
        """Qubit hamiltonian hdf5 round trip preserves partition."""
        h5py = pytest.importorskip("h5py")
        partition = LayeredPartition(strategy="geometry_coloring", groups=[[[0], [1]], [[2]]])
        ham = QubitHamiltonian(["XX", "YY", "ZZ"], np.array([0.1, 0.2, 0.3]), term_partition=partition)

        path = tmp_path / "ham.h5"
        with h5py.File(path, "w") as f:
            ham.to_hdf5(f)
        with h5py.File(path, "r") as f:
            restored = QubitHamiltonian.from_hdf5(f)

        assert isinstance(restored.term_partition, LayeredPartition)
        assert restored.term_partition == partition
