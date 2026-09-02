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
    QubitOperator,
    TaperingSpecification,
    TermPartition,
)
from qdk_chemistry.plugins.networkx import QDK_CHEMISTRY_HAS_NETWORKX
from qdk_chemistry.remote.serialization import deserialize_outputs, serialize_outputs
from qdk_chemistry.utils.model_hamiltonians import (
    create_heisenberg_hamiltonian,
    create_ising_hamiltonian,
)
from qdk_chemistry.utils.pauli_commutation import do_pauli_labels_commute, do_pauli_labels_qw_commute

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


@pytest.mark.parametrize(
    "partition",
    [
        FlatPartition(strategy="commuting", groups=((0, 1), (2,))),
        LayeredPartition(strategy="geometry_coloring", groups=(((0,), (1,)), ((2,),))),
    ],
)
def test_remote_round_trip_restores_partition_subtype(tmp_path, partition):
    """The term-partition loader restores the encoded concrete type."""
    serialize_outputs(tmp_path, partition)

    restored = deserialize_outputs(tmp_path)

    assert type(restored) is type(partition)
    assert restored == partition


# ---------------------------------------------------------------------------
# QubitOperator.term_partition property
# ---------------------------------------------------------------------------


class TestQubitHamiltonianTermPartition:
    def test_default_is_none(self):
        """Default is none."""
        qh = QubitOperator(["XX", "ZZ"], np.array([0.1, 0.2]))
        assert qh.term_partition is None

    def test_round_trip_flat(self):
        """Round trip flat."""
        partition = FlatPartition(strategy="commuting", groups=[[0], [1]])
        qh = QubitOperator(["XX", "ZZ"], np.array([0.1, 0.2]), term_partition=partition)
        assert qh.term_partition is partition

    def test_round_trip_layered(self):
        """Round trip layered."""
        partition = LayeredPartition(strategy="geometry_coloring", groups=[[[0, 1]]])
        qh = QubitOperator(["XX", "ZZ"], np.array([0.1, 0.2]), term_partition=partition)
        assert qh.term_partition is partition

    def test_to_interleaved_resets_partition(self):
        """To interleaved resets partition."""
        partition = FlatPartition(strategy="commuting", groups=[[0, 1, 2, 3]])
        qh = QubitOperator(
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
        assert {"commuting", "qubit_wise_commuting", "identity", "vacuum_annihilating"} <= set(names)

    def test_default_strategy_is_commuting(self):
        """Default strategy is commuting."""
        grouper = registry.create("term_grouper")
        assert grouper.name() == "commuting"

    @pytest.mark.parametrize("strategy", ["commuting", "qubit_wise_commuting", "identity", "vacuum_annihilating"])
    def test_returns_new_hamiltonian_with_partition(self, strategy):
        """Returns new hamiltonian with partition."""
        qh = QubitOperator(["XX", "YY", "ZZ"], np.array([1.0, 1.0, 3.0]))
        grouper = registry.create("term_grouper", strategy)
        out = grouper.run(qh)
        assert out is not qh
        assert isinstance(out.term_partition, FlatPartition)
        assert out.term_partition.strategy == strategy

    def test_partition_indices_cover_all_terms_exactly_once(self):
        """Partition indices cover all terms exactly once."""
        qh = QubitOperator(
            ["XXII", "YYII", "IIXX", "IIYY", "ZIII", "IZII"],
            np.array([0.5, 0.5, 0.25, 0.25, 0.5, 0.6]),
        )
        for strategy in ("commuting", "qubit_wise_commuting", "identity", "vacuum_annihilating"):
            grouper = registry.create("term_grouper", strategy)
            out = grouper.run(qh)
            indices = out.term_partition.all_indices()
            assert sorted(indices) == list(range(len(qh.pauli_strings)))

    def test_identity_strategy_one_term_per_group(self):
        """Identity strategy one term per group."""
        qh = QubitOperator(["XX", "YY", "ZZ"], np.array([1.0, 2.0, 3.0]))
        out = registry.create("term_grouper", "identity").run(qh)
        assert out.term_partition.num_groups == len(qh.pauli_strings)
        assert all(len(g) == 1 for g in out.term_partition.groups)

    @pytest.mark.parametrize("strategy", ["commuting", "qubit_wise_commuting", "identity", "vacuum_annihilating"])
    def test_operator_metadata_survives_grouping(self, strategy):
        """Grouping touches neither the qubits nor the mapped sector, so metadata must carry over."""
        qh = QubitOperator(
            ["XX", "YY", "ZZ"],
            np.array([1.0, 1.0, 3.0]),
            encoding="jordan-wigner",
            fermion_mode_order="blocked",
        )
        out = registry.create("term_grouper", strategy).run(qh)

        assert out.encoding == qh.encoding
        assert out.fermion_mode_order == qh.fermion_mode_order

    def test_vacuum_annihilating_preserves_tapering(self):
        """Grouping changes neither the qubits nor the mapped sector."""
        qh = QubitOperator(
            ["XX", "YY", "ZZ"],
            np.array([1.0, 1.0, 3.0]),
            tapering=TaperingSpecification(qubit_indices=(3, 1), eigenvalues=(1, -1)),
        )
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        assert out.tapering == qh.tapering

    def test_commuting_groups_globally_commute(self):
        # XX and YY commute (XY * YX = -ZZ * -ZZ = ZZ^2 = I; and YX * XY = ZZ),
        # ZZ commutes with both.  So all three should land in the same group.
        """Commuting groups globally commute."""
        qh = QubitOperator(["XX", "YY", "ZZ"], np.array([1.0, 1.0, 1.0]))
        out = registry.create("term_grouper", "commuting").run(qh)
        assert out.term_partition.num_groups == 1

    def test_qwc_separates_paulis_that_only_globally_commute(self):
        # XX and YY are NOT qubit-wise commuting, even though they globally commute.
        """Qwc separates paulis that only globally commute."""
        qh = QubitOperator(["XX", "YY"], np.array([1.0, 1.0]))
        out = registry.create("term_grouper", "qubit_wise_commuting").run(qh)
        assert out.term_partition.num_groups == 2


# ---------------------------------------------------------------------------
# Vacuum-annihilating term grouper
# ---------------------------------------------------------------------------


class TestVacuumAnnihilatingTermGrouper:
    """Tests for the ``vacuum_annihilating`` term grouper."""

    def test_cancellation_partners_are_grouped_together(self):
        """XX and YY flip the same qubits and must land in the same group."""
        # 0.5 (XX + YY) + 0.5 (I - Z0) is the JW image of a0^dag a1 + a1^dag a0 + a0^dag a0.
        qh = QubitOperator(["XX", "YY", "II", "IZ"], np.array([0.5, 0.5, 0.5, -0.5]))
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        groups = {frozenset(group) for group in out.term_partition.groups}
        assert frozenset({0, 1}) in groups
        # Pure I/Z strings are diagonal: they flip no qubits at all.
        assert frozenset({2, 3}) in groups

    def test_terms_flipping_different_qubits_are_separated(self):
        """Terms flipping different qubits cannot cancel each other on |0...0>."""
        qh = QubitOperator(["XXII", "YYII", "IIXX", "IIYY"], np.array([1.0, 1.0, 1.0, 1.0]))
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)
        assert out.term_partition.groups == ((0, 1), (2, 3))

    def test_same_support_with_odd_y_parity_is_separated(self):
        """X and Y flip the same qubit but anticommute, so they cannot share a group."""
        qh = QubitOperator(["IX", "IX", "IY", "IY"], np.array([1.0, -1.0, 1.0, -1.0]))
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)
        assert out.term_partition.groups == ((0, 1), (2, 3))

    def test_each_group_annihilates_the_zero_state(self):
        """Every group of a |0...0>-annihilating operator annihilates it too."""
        qh = QubitOperator(
            ["IIXX", "IIYY", "XXXX", "YYXX", "IIII", "IIIZ"],
            np.array([0.5, 0.5, 0.25, 0.25, 0.5, -0.5]),
        )
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        for group in out.term_partition.groups:
            amplitude = 0j
            for index in group:
                # P|0...0> = i^{n_Y} |b_F>; within a group every term hits the same |b_F>.
                phase = 1j ** out.pauli_strings[index].count("Y")
                amplitude += complex(out.coefficients[index]) * phase
            assert abs(amplitude) < 1e-12

    def test_group_members_pairwise_commute(self):
        """Members of a group flip the same qubits with the same Y parity, hence commute."""
        qh = QubitOperator(
            ["XXXX", "YYXX", "XXYY", "YYYY", "XYXY", "YXYX"],
            np.array([0.25, 0.25, 0.25, 0.25, 0.25, -0.25]),
        )
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        for group in out.term_partition.groups:
            for position, i in enumerate(group):
                for j in group[position + 1 :]:
                    assert do_pauli_labels_commute(out.pauli_strings[i], out.pauli_strings[j])

    def test_coefficients_split_a_flip_set_into_cancelling_groups(self):
        """Two independent XX/YY pairs cancel separately instead of forming one group."""
        qh = QubitOperator(["XX", "YY", "XX", "YY"], np.array([0.5, 0.5, 0.25, 0.25]))
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        assert out.term_partition.groups == ((0, 1), (2, 3))

    def test_terms_that_cannot_cancel_raise(self):
        """Unbalanced coefficients leave a remainder that no grouping can annihilate."""
        qh = QubitOperator(["XX", "YY", "XX"], np.array([0.5, 0.5, 0.25]))

        with pytest.raises(ValueError, match="uncancelled vacuum amplitude"):
            registry.create("term_grouper", "vacuum_annihilating").run(qh)

    def test_diagonal_group_may_leave_a_vacuum_phase(self):
        """Diagonal terms only phase the vacuum, which a consumer can correct for."""
        qh = QubitOperator(["ZI", "IZ", "II"], np.array([0.5, 0.5, 3.0]))
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        assert out.term_partition.groups == ((0, 1, 2),)

    @pytest.mark.parametrize("coefficients", [[0.5, 0.5j], [0.5, np.inf], [0.5, np.nan]])
    def test_non_real_or_non_finite_coefficients_raise(self, coefficients):
        """Vacuum amplitudes are compared against a real tolerance."""
        qh = QubitOperator(["XX", "YY"], np.array(coefficients))

        with pytest.raises(ValueError, match="finite, real coefficients"):
            registry.create("term_grouper", "vacuum_annihilating").run(qh)

    def test_tolerance_is_configurable(self):
        """``tolerance`` decides how exactly the coefficients have to cancel."""
        qh = QubitOperator(["XX", "YY", "XX", "YY"], np.array([0.5, 0.5 + 1e-7, 0.25, 0.25]))

        with pytest.raises(ValueError, match="uncancelled vacuum amplitude"):
            registry.create("term_grouper", "vacuum_annihilating").run(qh)

        # Certified as a whole, but no prefix cancels exactly, so the set stays in one group.
        tolerant = registry.create("term_grouper", "vacuum_annihilating", tolerance=1e-6)
        assert tolerant.run(qh).term_partition.groups == ((0, 1, 2, 3),)

    def test_tiny_prefixes_do_not_strand_a_remainder(self):
        """The flip set cancels exactly, so splitting on a sub-tolerance prefix must not reject it."""
        qh = QubitOperator(["XX", "XX", "XX"], np.array([0.75e-9, 0.75e-9, -1.5e-9]))
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        assert out.term_partition.groups == ((0, 1, 2),)

    def test_catastrophic_cancellation_is_accepted(self):
        """The set sums to zero exactly, which a running subtraction would lose."""
        qh = QubitOperator(["XX"] * 4, np.array([1e16, 1.0, -1e16, -1.0]))
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        assert sorted(index for group in out.term_partition.groups for index in group) == [0, 1, 2, 3]

    @pytest.mark.parametrize("tolerance", [-1e-9, np.inf, np.nan])
    def test_invalid_tolerance_raises(self, tolerance):
        """An infinite tolerance would certify every term on its own."""
        qh = QubitOperator(["XX", "YY"], np.array([0.5, 0.5]))

        with pytest.raises(ValueError, match="tolerance must be finite and non-negative"):
            registry.create("term_grouper", "vacuum_annihilating", tolerance=tolerance).run(qh)

    def test_group_order_is_deterministic_with_the_diagonal_group_first(self):
        """Group order drives the Trotter sequence, so it must not depend on dict ordering."""
        qh = QubitOperator(
            ["IIXI", "IZII", "XIII", "IIXI", "IIII", "XIII"],
            np.array([1.0, 2.0, 3.0, -1.0, 5.0, -3.0]),
        )
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        # Diagonal (I/Z-only) terms first, then the remaining groups by first member index.
        assert out.term_partition.groups == ((1, 4), (0, 3), (2, 5))

    def test_input_operator_is_not_mutated(self):
        """The grouper returns a new operator and leaves the input untouched."""
        qh = QubitOperator(["XX", "YY", "IZ"], np.array([0.5, 0.5, -0.5]))
        out = registry.create("term_grouper", "vacuum_annihilating").run(qh)

        assert qh.term_partition is None
        assert out.pauli_strings == qh.pauli_strings
        assert np.allclose(out.coefficients, qh.coefficients)


# ---------------------------------------------------------------------------
# NetworkX-backed term groupers (requires networkx)
# ---------------------------------------------------------------------------


class TestNxTermGroupers:
    """Tests for the NetworkX-backed term groupers."""

    pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_NETWORKX, reason="networkx not installed")

    NX_STRATEGIES = ("nx_commuting", "nx_qubit_wise_commuting")

    def test_nx_strategies_are_registered(self):
        """NetworkX plugin registers both nx_commuting and nx_qubit_wise_commuting."""
        names = registry.available("term_grouper")
        assert {"nx_commuting", "nx_qubit_wise_commuting"} <= set(names)

    @pytest.mark.parametrize("strategy", NX_STRATEGIES)
    def test_returns_new_hamiltonian_with_flat_partition(self, strategy):
        """Grouper returns a new QubitOperator with a FlatPartition."""
        qh = QubitOperator(["XX", "YY", "ZZ"], np.array([1.0, 2.0, 3.0]))
        grouper = registry.create("term_grouper", strategy)
        out = grouper.run(qh)
        assert out is not qh
        assert isinstance(out.term_partition, FlatPartition)
        assert out.term_partition.strategy == strategy

    @pytest.mark.parametrize("strategy", NX_STRATEGIES)
    def test_partition_indices_cover_all_terms_exactly_once(self, strategy):
        """Every term index appears exactly once across all partition groups."""
        qh = QubitOperator(
            ["XIII", "IXII", "IIXI", "IIIX", "ZIII", "IZII"],
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        )
        grouper = registry.create("term_grouper", strategy)
        out = grouper.run(qh)
        indices = out.term_partition.all_indices()
        assert sorted(indices) == list(range(len(qh.pauli_strings)))

    def test_nx_commuting_groups_all_commute(self):
        """Every pair of terms within an nx_commuting group globally commutes."""
        qh = QubitOperator(
            ["XIII", "IXII", "IIXI", "IIIX", "ZIII", "IZII", "XXII", "YYII"],
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        )
        out = registry.create("term_grouper", "nx_commuting").run(qh)
        labels = list(qh.pauli_strings)
        for group in out.term_partition.groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    assert do_pauli_labels_commute(labels[group[i]], labels[group[j]])

    def test_nx_qwc_groups_are_qubit_wise_commuting(self):
        """Every pair of terms within an nx_qubit_wise_commuting group is qubit-wise commuting."""
        qh = QubitOperator(
            ["XIII", "IXII", "IIXI", "IIIX", "ZIII", "IZII", "XXII", "YYII"],
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        )
        out = registry.create("term_grouper", "nx_qubit_wise_commuting").run(qh)
        labels = list(qh.pauli_strings)
        for group in out.term_partition.groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    assert do_pauli_labels_qw_commute(labels[group[i]], labels[group[j]])

    def test_nx_commuting_merges_globally_commuting_terms(self):
        """XX and YY globally commute and should be in the same nx_commuting group."""
        qh = QubitOperator(["XX", "YY", "ZZ"], np.array([1.0, 1.0, 1.0]))
        out = registry.create("term_grouper", "nx_commuting").run(qh)
        assert out.term_partition.num_groups == 1

    def test_nx_qwc_separates_globally_only_commuting_terms(self):
        """XX and YY globally commute but are NOT qubit-wise commuting — they must be separated."""
        qh = QubitOperator(["XX", "YY"], np.array([1.0, 1.0]))
        out = registry.create("term_grouper", "nx_qubit_wise_commuting").run(qh)
        assert out.term_partition.num_groups == 2

    def test_preserves_coefficients_and_metadata(self):
        """Grouper preserves coefficients, encoding, and fermion_mode_order."""
        qh = QubitOperator(
            ["XX", "ZZ"],
            np.array([0.1, 0.2]),
            encoding="jordan-wigner",
            fermion_mode_order="blocked",
        )
        out = registry.create("term_grouper", "nx_commuting").run(qh)
        np.testing.assert_array_equal(out.coefficients, qh.coefficients)
        assert out.encoding == qh.encoding
        assert out.fermion_mode_order == qh.fermion_mode_order

    def test_empty_hamiltonian(self):
        """Grouper handles a single-term Hamiltonian without error."""
        qh = QubitOperator(["X"], np.array([1.0]))
        out = registry.create("term_grouper", "nx_commuting").run(qh)
        assert isinstance(out.term_partition, FlatPartition)
        assert out.term_partition.num_groups == 1
        assert out.term_partition.all_indices() == [0]


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
        ham = QubitOperator(["XXII", "IXXI", "IIXX", "ZIII"], np.array([1.0, 1.0, 1.0, 0.5]))
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
# QubitOperator round-trips term_partition through JSON / HDF5
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
        ham = QubitOperator(["XX", "YY", "ZZ"], np.array([0.1, 0.2, 0.3]), term_partition=partition)
        restored = QubitOperator.from_json(ham.to_json())
        assert isinstance(restored.term_partition, FlatPartition)
        assert restored.term_partition == partition

    def test_qubit_hamiltonian_json_round_trip_with_no_partition(self):
        """Qubit hamiltonian json round trip with no partition."""
        ham = QubitOperator(["XX", "ZZ"], np.array([0.1, 0.2]))
        restored = QubitOperator.from_json(ham.to_json())
        assert restored.term_partition is None

    def test_qubit_hamiltonian_hdf5_round_trip_preserves_partition(self, tmp_path):
        """Qubit hamiltonian hdf5 round trip preserves partition."""
        h5py = pytest.importorskip("h5py")
        partition = LayeredPartition(strategy="geometry_coloring", groups=[[[0], [1]], [[2]]])
        ham = QubitOperator(["XX", "YY", "ZZ"], np.array([0.1, 0.2, 0.3]), term_partition=partition)

        path = tmp_path / "ham.h5"
        with h5py.File(path, "w") as f:
            ham.to_hdf5(f)
        with h5py.File(path, "r") as f:
            restored = QubitOperator.from_hdf5(f)

        assert isinstance(restored.term_partition, LayeredPartition)
        assert restored.term_partition == partition
