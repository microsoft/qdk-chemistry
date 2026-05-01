"""Tests for geometry-aware Trotter term grouping via QubitHamiltonian.term_partition."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter
from qdk_chemistry.data import LatticeGraph, LayeredPartition, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import (
    create_heisenberg_hamiltonian,
    create_ising_hamiltonian,
)


def _assert_disjoint_supports(layer: QubitHamiltonian) -> None:
    """Assert that all terms in a layer have disjoint qubit supports."""
    used_qubits: set[int] = set()
    for ps in layer.pauli_strings:
        support = {i for i, c in enumerate(reversed(ps)) if c != "I"}
        overlap = used_qubits & support
        assert not overlap, f"Qubit overlap {overlap} in layer"
        used_qubits.update(support)


def _materialise_layers(qh: QubitHamiltonian) -> list[list[QubitHamiltonian]]:
    """Materialise a Hamiltonian's layered term partition into ``[group][layer] QubitHamiltonian``."""
    partition = qh.term_partition
    assert isinstance(partition, LayeredPartition)
    return [
        [
            QubitHamiltonian(
                pauli_strings=[qh.pauli_strings[i] for i in layer],
                coefficients=np.asarray([qh.coefficients[i] for i in layer]),
                encoding=qh.encoding,
                fermion_mode_order=qh.fermion_mode_order,
            )
            for layer in group
            if layer
        ]
        for group in partition.groups
    ]


class TestIsingPartitionShape:
    """Tests for the geometry-coloring partition produced for Ising Hamiltonians."""

    def test_chain_ising(self):
        """Ising on a 4-site chain populates a partition with field + ZZ groups."""
        lattice = LatticeGraph.chain(4)
        qh = create_ising_hamiltonian(lattice, j=1.0, h=0.5)
        groups = _materialise_layers(qh)

        # At least 2 groups: X field + ZZ coupling
        assert len(groups) >= 2

        # Find the ZZ group (multi-layer, due to edge coloring of the chain).
        zz_groups = [g for g in groups if any(any("Z" in ps for ps in layer.pauli_strings) for layer in g)]
        assert zz_groups, "expected a ZZ-bearing group"
        for layer in zz_groups[0]:
            _assert_disjoint_supports(layer)

    def test_chain_ising_no_field(self):
        """Ising without a transverse field should have a single ZZ-only group with 2 colors."""
        lattice = LatticeGraph.chain(6)
        qh = create_ising_hamiltonian(lattice, j=1.0, h=0.0)
        groups = _materialise_layers(qh)

        assert len(groups) == 1
        assert len(groups[0]) == 2  # 2 colors for chain


class TestHeisenbergPartitionShape:
    """Tests for the geometry-coloring partition produced for Heisenberg Hamiltonians."""

    def test_chain_heisenberg(self):
        """Heisenberg on a 4-site chain produces 3 coupling groups with 2 layers each."""
        lattice = LatticeGraph.chain(4)
        qh = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0)
        groups = _materialise_layers(qh)

        # 3 coupling groups (XX, YY, ZZ); no field => no field groups.
        assert len(groups) == 3
        for group in groups:
            assert len(group) == 2  # 2 colors for chain
            for layer in group:
                _assert_disjoint_supports(layer)

    def test_patch_heisenberg(self):
        """Heisenberg on a 4x4 square lattice has 4 colors per coupling group."""
        lattice = LatticeGraph.square(4, 4)
        qh = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0)
        groups = _materialise_layers(qh)

        assert len(groups) == 3
        for group in groups:
            assert len(group) == 4
            for layer in group:
                _assert_disjoint_supports(layer)


class TestTrotterConsumesPartition:
    """Tests for the Trotter builder consuming QubitHamiltonian.term_partition."""

    @pytest.mark.parametrize("order", [1, 2, 4])
    def test_orders(self, order):
        """Trotter at orders 1/2/4 with a populated partition produces a valid container."""
        lattice = LatticeGraph.chain(4)
        qh = create_ising_hamiltonian(lattice, j=1.0, h=0.5)

        builder = Trotter(order=order)
        result = builder.run(qh, time=1.0)

        assert result is not None
        container = result.get_container()
        assert container.step_reps >= 1
        assert len(container.step_terms) > 0

    def test_partition_vs_no_partition_same_qubit_count(self):
        """Trotter with and without a partition produces unitaries on the same number of qubits."""
        lattice = LatticeGraph.chain(3)
        qh_grouped = create_ising_hamiltonian(lattice, j=1.0, h=0.5)
        qh_plain = create_ising_hamiltonian(lattice, j=1.0, h=0.5, include_term_groups=False)

        builder = Trotter(order=1, num_divisions=10)
        result_grouped = builder.run(qh_grouped, time=0.1)
        result_plain = builder.run(qh_plain, time=0.1)

        assert result_grouped.get_container().num_qubits == result_plain.get_container().num_qubits

    def test_geometry_partition_yields_no_more_layers_than_generic(self):
        """A geometry-coloring partition uses at most as many parallel layers as generic grouping."""
        lattice = LatticeGraph.chain(6)
        qh = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0)
        qh_plain = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0, include_term_groups=False)

        groups_geometry = _materialise_layers(qh)
        total_geometry_layers = sum(len(g) for g in groups_geometry)

        builder = Trotter(order=1, optimize_term_ordering=True)
        generic_groups = builder._group_terms(qh_plain, optimize_term_ordering=True)
        total_generic_layers = sum(len(g) for g in generic_groups)

        assert total_geometry_layers <= total_generic_layers
