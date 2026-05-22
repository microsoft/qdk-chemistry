"""Tests for DrivenQubitHamiltonian construction validation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qdk_chemistry.data import (
    DrivenQubitHamiltonian,
    FlatPartition,
    LatticeGraph,
    LayeredPartition,
    QubitHamiltonian,
)
from qdk_chemistry.utils.model_hamiltonians import create_ising_hamiltonian


class TestDrivenQubitHamiltonianValidation:
    """Tests for DrivenQubitHamiltonian construction validation."""

    def _make_hamiltonian(self, num_qubits: int = 2) -> QubitHamiltonian:
        labels = ["Z" + "I" * (num_qubits - 1)]
        return QubitHamiltonian(labels, np.array([1.0]))

    def _constant_drive(self, _t: float) -> float:
        return 1.0

    def test_mismatched_num_qubits_raises(self):
        """h0 and h1 with different num_qubits should raise ValueError."""
        h2 = self._make_hamiltonian(num_qubits=2)
        h3 = self._make_hamiltonian(num_qubits=3)
        with pytest.raises(ValueError, match="same number of qubits"):
            DrivenQubitHamiltonian(h2, h3, drive=self._constant_drive)

    def test_valid_construction(self):
        """Valid inputs should construct without error."""
        h = self._make_hamiltonian()
        td = DrivenQubitHamiltonian(h, h, drive=self._constant_drive)
        assert td.num_qubits == 2

    def test_evaluate_applies_drive(self):
        """evaluate() should return H0 + f(t) * H1."""
        h0 = QubitHamiltonian(["ZI"], np.array([1.0]))
        h1 = QubitHamiltonian(["IZ"], np.array([2.0]))
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda t: t)
        snap = td.evaluate(0.5)
        assert snap.pauli_strings == ["ZI", "IZ"]
        np.testing.assert_allclose(snap.coefficients, [1.0, 1.0])


class TestDrivenQubitHamiltonianDriveFunctions:
    """Tests for DrivenQubitHamiltonian with various drive functions."""

    def _h0(self) -> QubitHamiltonian:
        return QubitHamiltonian(["ZI", "IZ"], np.array([1.0, 0.5]))

    def _h1(self) -> QubitHamiltonian:
        return QubitHamiltonian(["XX", "YY"], np.array([1.0, 1.0]))

    def test_sinusoidal_drive(self):
        """Sinusoidal drive f(t) = sin(t) should modulate H1 coefficients."""
        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=np.sin)
        for t in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            snap = td.evaluate(t)
            np.testing.assert_allclose(
                snap.coefficients,
                [1.0, 0.5, np.sin(t) * 1.0, np.sin(t) * 1.0],
            )

    def test_exponential_decay_drive(self):
        """Exponential decay f(t) = exp(-t) should dampen H1 over time."""
        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=lambda t: np.exp(-t))
        times = [0.0, 1.0, 2.0, 5.0]
        snaps = [td.evaluate(t) for t in times]
        h1_norms = [np.sum(np.abs(s.coefficients[2:])) for s in snaps]
        assert all(a > b for a, b in itertools.pairwise(h1_norms))

    def test_linear_ramp_drive(self):
        """Linear ramp f(t) = t should scale H1 linearly."""
        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=lambda t: t)
        np.testing.assert_allclose(td.evaluate(0.5).coefficients[2:], [0.5, 0.5])
        np.testing.assert_allclose(td.evaluate(1.0).coefficients[2:], [1.0, 1.0])
        np.testing.assert_allclose(td.evaluate(2.0).coefficients[2:], [2.0, 2.0])

    def test_zero_drive_returns_h0_only(self):
        """Zero drive f(t) = 0 should yield only H0 contributions for H1 terms."""
        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=lambda _t: 0.0)
        snap = td.evaluate(1.0)
        np.testing.assert_allclose(snap.coefficients[:2], [1.0, 0.5])
        np.testing.assert_allclose(snap.coefficients[2:], [0.0, 0.0])

    def test_step_function_drive(self):
        """Step function drive should switch H1 on/off at threshold."""

        def step_drive(t: float) -> float:
            return 1.0 if t >= 1.5 else 0.0

        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=step_drive)
        np.testing.assert_allclose(td.evaluate(1.0).coefficients[2:], [0.0, 0.0])
        np.testing.assert_allclose(td.evaluate(2.0).coefficients[2:], [1.0, 1.0])

    def test_cosine_drive_symmetry(self):
        """Cosine drive should produce identical magnitudes at symmetric time points."""
        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=np.cos)
        snap_a = td.evaluate(np.pi / 3)
        snap_b = td.evaluate(2 * np.pi / 3)
        np.testing.assert_allclose(np.abs(snap_a.coefficients[2:]), np.abs(snap_b.coefficients[2:]))

    def test_negative_drive(self):
        """Negative drive should flip the sign of H1 coefficients."""
        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=lambda _t: -2.0)
        snap = td.evaluate(1.0)
        np.testing.assert_allclose(snap.coefficients[2:], [-2.0, -2.0])

    def test_container_properties_accessible(self):
        """Container properties h0, h1, drive should be accessible through the wrapper."""

        def square_drive(t: float) -> float:
            return t**2

        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=square_drive)
        assert td.h0.pauli_strings == ["ZI", "IZ"]
        assert td.h1.pauli_strings == ["XX", "YY"]
        assert td.drive is square_drive

    def test_repr(self):
        """Repr should show the class name and num_qubits."""
        td = DrivenQubitHamiltonian(self._h0(), self._h1(), drive=lambda t: t)
        assert "DrivenQubitHamiltonian" in repr(td)
        assert "num_qubits=2" in repr(td)


class TestDrivenQubitHamiltonianPartition:
    """Tests for partition preservation through evaluate()."""

    def test_evaluate_preserves_flat_partition(self):
        """evaluate() should carry through a merged FlatPartition when both h0 and h1 have one."""
        h0 = QubitHamiltonian(
            ["ZI", "IZ"], np.array([1.0, 1.0]), term_partition=FlatPartition(strategy="s", groups=((0, 1),))
        )
        h1 = QubitHamiltonian(["XX"], np.array([0.5]), term_partition=FlatPartition(strategy="s", groups=((0,),)))
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda t: t)
        snap = td.evaluate(2.0)
        assert snap.term_partition is not None
        assert isinstance(snap.term_partition, FlatPartition)
        assert snap.term_partition.groups == ((0, 1), (2,))

    def test_evaluate_preserves_layered_partition(self):
        """evaluate() should carry through a merged LayeredPartition when both h0 and h1 have one."""
        h0 = QubitHamiltonian(
            ["ZI", "IZ"], np.array([1.0, 1.0]), term_partition=LayeredPartition(strategy="s", groups=(((0,), (1,)),))
        )
        h1 = QubitHamiltonian(["XX"], np.array([0.5]), term_partition=LayeredPartition(strategy="s", groups=(((0,),),)))
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 1.0)
        snap = td.evaluate(0.0)
        assert snap.term_partition is not None
        assert isinstance(snap.term_partition, LayeredPartition)
        # h1 index 0 is offset by 2 (len(h0.pauli_strings))
        assert snap.term_partition.groups == (((0,), (1,)), ((2,),))

    def test_evaluate_no_partition_when_missing(self):
        """evaluate() should return no partition when h0 or h1 lacks one."""
        h0 = QubitHamiltonian(["ZI"], np.array([1.0]))
        h1 = QubitHamiltonian(["IX"], np.array([0.5]))
        td = DrivenQubitHamiltonian(h0, h1, drive=lambda _t: 1.0)
        snap = td.evaluate(0.0)
        assert snap.term_partition is None

    def test_evaluate_preserves_partition_from_ising(self):
        """Partition from create_ising_hamiltonian should survive through evaluate()."""
        lattice = LatticeGraph.chain(2)
        h0 = create_ising_hamiltonian(lattice, j=1.0, h=0.0)
        h1 = create_ising_hamiltonian(lattice, j=0.0, h=0.5)
        td = DrivenQubitHamiltonian(h0, h1, drive=np.sin)
        snap = td.evaluate(1.0)
        assert snap.term_partition is not None
        assert snap.term_partition.num_groups == h0.term_partition.num_groups + h1.term_partition.num_groups
