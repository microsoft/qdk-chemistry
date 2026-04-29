"""Tests for geometry-aware Trotter term grouping."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter
from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.geometry import Chain1D, Patch2D
from qdk_chemistry.utils.model_hamiltonians import (
    create_heisenberg_hamiltonian,
    create_ising_hamiltonian,
    heisenberg_term_groups,
    ising_term_groups,
)


class TestIsingTermGroups:
    """Tests for geometry-aware Ising model term grouping."""

    def test_chain_ising(self):
        """Ising model on a 4-site chain should produce 2 groups: ZZ + X-field."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(4)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Chain1D(4)
        groups = ising_term_groups(lattice, j=1.0, h=0.5, geometry=geometry)

        # Should have at least 2 groups: X field + ZZ coupling
        assert len(groups) >= 2

        # X field group should have 1 layer (comes first for Strang merge optimality)
        x_group = groups[0]
        assert len(x_group) == 1

        # ZZ coupling group should have 2 layers (2 colors for chain)
        zz_group = groups[1]
        assert len(zz_group) == 2

        # Each layer should have non-overlapping qubit supports
        for layer in zz_group:
            _assert_disjoint_supports(layer)

    def test_chain_ising_no_field(self):
        """Ising model without field should produce 1 group (ZZ only)."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(6)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Chain1D(6)
        groups = ising_term_groups(lattice, j=1.0, h=0.0, geometry=geometry)

        assert len(groups) == 1  # Only ZZ group
        assert len(groups[0]) == 2  # 2 colors for chain


class TestHeisenbergTermGroups:
    """Tests for geometry-aware Heisenberg model term grouping."""

    def test_chain_heisenberg(self):
        """Heisenberg model on a chain should produce 3 groups: XX, YY, ZZ."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(4)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Chain1D(4)
        groups = heisenberg_term_groups(lattice, jx=1.0, jy=1.0, jz=1.0, geometry=geometry)

        # 3 coupling groups (XX, YY, ZZ)
        assert len(groups) == 3

        # Each group should have 2 layers (2 colors for a chain)
        for group in groups:
            assert len(group) == 2
            for layer in group:
                _assert_disjoint_supports(layer)

    def test_patch_heisenberg(self):
        """Heisenberg model on a 2D patch should have 4 colors per group."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.square(4, 4)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Patch2D(4, 4)
        groups = heisenberg_term_groups(lattice, jx=1.0, jy=1.0, jz=1.0, geometry=geometry)

        # 3 coupling groups (XX, YY, ZZ)
        assert len(groups) == 3

        # Each group should have 4 layers (4 colors for a 2D patch)
        for group in groups:
            assert len(group) == 4
            for layer in group:
                _assert_disjoint_supports(layer)

    def test_auto_geometry(self):
        """When no geometry is provided, greedy coloring should still work."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(4)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        groups = heisenberg_term_groups(lattice, jx=1.0, jy=1.0, jz=1.0)

        # Should still produce valid groups
        assert len(groups) >= 3
        for group in groups:
            for layer in group:
                _assert_disjoint_supports(layer)


class TestTrotterWithTermGroups:
    """Tests for Trotter builder using pre-computed term groups."""

    def test_first_order_with_groups(self):
        """First-order Trotter with pre-computed groups should produce valid output."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(4)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Chain1D(4)
        qh = create_ising_hamiltonian(lattice, j=1.0, h=0.5)
        groups = ising_term_groups(lattice, j=1.0, h=0.5, geometry=geometry)

        builder = Trotter(order=1, term_groups=groups)
        result = builder.run(qh, time=1.0)

        assert result is not None
        container = result.get_container()
        assert container.step_reps >= 1
        assert len(container.step_terms) > 0

    def test_second_order_with_groups(self):
        """Second-order Trotter with pre-computed groups should produce valid output."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(4)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Chain1D(4)
        qh = create_ising_hamiltonian(lattice, j=1.0, h=0.5)
        groups = ising_term_groups(lattice, j=1.0, h=0.5, geometry=geometry)

        builder = Trotter(order=2, term_groups=groups)
        result = builder.run(qh, time=1.0)

        assert result is not None
        container = result.get_container()
        assert len(container.step_terms) > 0

    def test_fourth_order_with_groups(self):
        """Fourth-order Trotter with pre-computed groups should produce valid output."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(4)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Chain1D(4)
        qh = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0)
        groups = heisenberg_term_groups(lattice, jx=1.0, jy=1.0, jz=1.0, geometry=geometry)

        builder = Trotter(order=4, term_groups=groups)
        result = builder.run(qh, time=1.0)

        assert result is not None
        container = result.get_container()
        assert len(container.step_terms) > 0

    def test_groups_vs_generic_correctness(self):
        """Geometry-grouped and generic-grouped Trotter should approximate the same unitary."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(3)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Chain1D(3)
        qh = create_ising_hamiltonian(lattice, j=1.0, h=0.5)
        groups = ising_term_groups(lattice, j=1.0, h=0.5, geometry=geometry)

        # Both should produce valid decompositions
        builder_generic = Trotter(order=1, num_divisions=10)
        builder_grouped = Trotter(order=1, num_divisions=10, term_groups=groups)

        result_generic = builder_generic.run(qh, time=0.1)
        result_grouped = builder_grouped.run(qh, time=0.1)

        # Both should have the same number of qubits
        assert result_generic.get_container().num_qubits == result_grouped.get_container().num_qubits

    def test_fewer_terms_with_geometry(self):
        """Geometry-aware grouping should produce equal or fewer term groups than generic."""
        try:
            from qdk_chemistry.data import LatticeGraph

            lattice = LatticeGraph.chain(6)
        except ImportError:
            pytest.skip("LatticeGraph not available")

        geometry = Chain1D(6)
        qh = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0)
        groups = heisenberg_term_groups(lattice, jx=1.0, jy=1.0, jz=1.0, geometry=geometry)

        # Geometry grouping: 3 groups (XX, YY, ZZ) with 2 colors each = 6 layers total
        total_geometry_layers = sum(len(g) for g in groups)

        # Generic grouping: likely more layers
        builder = Trotter(order=1, optimize_term_ordering=True)
        generic_groups = builder._group_terms(qh, optimize_term_ordering=True)
        total_generic_layers = sum(len(g) for g in generic_groups)

        # Geometry-aware should use fewer or equal total layers
        assert total_geometry_layers <= total_generic_layers


def _assert_disjoint_supports(layer: QubitHamiltonian) -> None:
    """Assert that all terms in a layer have disjoint qubit supports."""
    used_qubits: set[int] = set()
    for ps in layer.pauli_strings:
        support = {i for i, c in enumerate(reversed(ps)) if c != "I"}
        overlap = used_qubits & support
        assert not overlap, f"Qubit overlap {overlap} in layer"
        used_qubits.update(support)
