"""Tests for PiecewiseConstantQubitHamiltonian construction validation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.data import PiecewiseConstantQubitHamiltonian, QubitHamiltonian


class TestPiecewiseConstantQubitHamiltonianValidation:
    """Tests for PiecewiseConstantQubitHamiltonian construction validation."""

    def _make_hamiltonian(self, num_qubits: int = 2) -> QubitHamiltonian:
        labels = ["Z" + "I" * (num_qubits - 1)]
        return QubitHamiltonian(labels, np.array([1.0]))

    def test_empty_hamiltonians_raises(self):
        """Empty hamiltonians list should raise ValueError."""
        with pytest.raises(ValueError, match="hamiltonians must not be empty"):
            PiecewiseConstantQubitHamiltonian([], [1.0])

    def test_empty_times_raises(self):
        """Empty times list should raise ValueError."""
        with pytest.raises(ValueError, match="times must not be empty"):
            PiecewiseConstantQubitHamiltonian([self._make_hamiltonian()], [])

    def test_length_mismatch_raises(self):
        """Mismatched lengths should raise ValueError."""
        h = self._make_hamiltonian()
        with pytest.raises(ValueError, match="same length"):
            PiecewiseConstantQubitHamiltonian([h, h], [1.0])

    def test_non_monotonic_times_raises(self):
        """Non-monotonically-increasing times should raise ValueError."""
        h = self._make_hamiltonian()
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            PiecewiseConstantQubitHamiltonian([h, h], [2.0, 1.0])

    def test_duplicate_times_raises(self):
        """Duplicate times should raise ValueError (strict monotonicity)."""
        h = self._make_hamiltonian()
        with pytest.raises(ValueError, match="strictly monotonically increasing"):
            PiecewiseConstantQubitHamiltonian([h, h], [1.0, 1.0])

    def test_mismatched_num_qubits_raises(self):
        """Hamiltonians with different num_qubits should raise ValueError."""
        h2 = self._make_hamiltonian(num_qubits=2)
        h3 = self._make_hamiltonian(num_qubits=3)
        with pytest.raises(ValueError, match="same number of qubits"):
            PiecewiseConstantQubitHamiltonian([h2, h3], [1.0, 2.0])

    def test_valid_construction(self):
        """Valid inputs should construct without error."""
        h = self._make_hamiltonian()
        td = PiecewiseConstantQubitHamiltonian([h, h], [1.0, 2.0])
        assert len(td) == 2
        assert td.num_qubits == 2
