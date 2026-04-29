"""Tests for Trotter error bound estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import pytest

from qdk_chemistry.algorithms.hamiltonian_unitary.builder.trotter_error import (
    trotter_steps_commutator,
    trotter_steps_naive,
)
from qdk_chemistry.data import QubitHamiltonian


class TestTrotterStepsNaive:
    """Tests for the trotter_steps_naive function."""

    def test_basic(self):
        """Test basic naive bound computation."""
        # one_norm = 2, N = ceil((2^2 * 2 * 1^2) / 0.1) = ceil(80) = 80
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        assert trotter_steps_naive(h, 1.0, 0.1, order=1) == 80

    def test_minimum_one(self):
        """Test that result is at least 1."""
        # Very large epsilon should still give 1
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert trotter_steps_naive(h, 0.001, 1e6, order=1) == 1

    def test_small_accuracy(self):
        """Test with very small accuracy."""
        # one_norm = 1, N = ceil((1^2 * 2 * 1^2) / 0.001) = ceil(2000) = 2000
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert trotter_steps_naive(h, 1.0, 0.001, order=1) == 2000

    def test_large_time(self):
        """Test with large evolution time."""
        # one_norm = 1, N = ceil((1^2 * 2 * 10^2) / 1.0) = ceil(200) = 200
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert trotter_steps_naive(h, 10.0, 1.0, order=1) == 200

    def test_zero_accuracy_raises(self):
        """Test that zero accuracy raises ValueError."""
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        with pytest.raises(ValueError, match="positive"):
            trotter_steps_naive(h, 1.0, 0.0, order=1)

    def test_negative_accuracy_raises(self):
        """Test that negative accuracy raises ValueError."""
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        with pytest.raises(ValueError, match="positive"):
            trotter_steps_naive(h, 1.0, -0.1, order=1)

    # Second-order Trotter tests.
    def test_basic_second_order(self):
        """Test basic second-order naive bound computation."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        assert trotter_steps_naive(h, 1.0, 0.1, order=2) == 18

    def test_small_accuracy_second_order(self):
        """Test with very small accuracy."""
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert trotter_steps_naive(h, 1.0, 0.0001, order=2) == 200

    def test_large_time_second_order(self):
        """Test with large evolution time."""
        # one_norm = 1, N = ceil((2 * 1^1.5 * 10^1.5) / 1.0^0.5) = 64
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert trotter_steps_naive(h, 10.0, 1.0, order=2) == 64

    # Higher-order Trotter tests.

    def test_order_3_raises(self):
        """Test that odd order > 2 raises NotImplementedError."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        with pytest.raises(NotImplementedError, match="order 3"):
            trotter_steps_naive(h, 1.0, 0.1, order=3)

    def test_basic_higher_order(self):
        """Test basic higher-order naive bound computation."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        assert trotter_steps_naive(h, 1.0, 0.1, order=6) == 7

    def test_small_accuracy_higher_order(self):
        """Test with very small accuracy."""
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert trotter_steps_naive(h, 1.0, 0.0001, order=6) == 10

    def test_large_time_higher_order(self):
        """Test with large evolution time."""
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert trotter_steps_naive(h, 10.0, 1.0, order=6) == 30


class TestTrotterStepsCommutator:
    """Tests for the trotter_steps_commutator function."""

    def test_anticommuting_pair(self):
        """Test with an anticommuting pair."""
        # X and Z anticommute: commutator bound = 2
        # N = ceil(2 * 1^2 / (2 * 0.1)) = ceil(10) = 10
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        assert trotter_steps_commutator(h, 1.0, 0.1, order=1) == 10

    def test_all_commuting(self):
        """Test with all commuting terms."""
        # XI and IX commute: commutator bound = 0, N = 1
        h = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        assert trotter_steps_commutator(h, 1.0, 0.1, order=1) == 1

    def test_tighter_than_naive(self):
        """Test that commutator bound is never looser than naive."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        eps = 0.01
        time = 1.0
        n_naive = trotter_steps_naive(h, time, eps, order=1)
        n_comm = trotter_steps_commutator(h, time, eps, order=1)
        assert n_comm <= n_naive

    def test_minimum_one(self):
        """Test that result is at least 1 for commuting Hamiltonian."""
        h = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        assert trotter_steps_commutator(h, 1.0, 100.0, order=1) == 1

    def test_time_scaling(self):
        """Test that step count scales with t^2."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        n1 = trotter_steps_commutator(h, 1.0, 0.1, order=1)
        n2 = trotter_steps_commutator(h, 2.0, 0.1, order=1)
        # n2 should be approximately 4 * n1 (t^2 scaling)
        assert n2 == math.ceil(4 * (2.0 / 0.2))  # 40
        assert abs(n2 - 4 * n1) <= 1  # Allow for ceiling effects

    def test_zero_accuracy_raises(self):
        """Test that zero accuracy raises ValueError."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        with pytest.raises(ValueError, match="positive"):
            trotter_steps_commutator(h, 1.0, 0.0, order=1)

    # Second-order Trotter tests.

    def test_all_commuting_second_order(self):
        """Test with all commuting terms."""
        # XI and IX commute: commutator bound = 0, N = 1
        h = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        assert trotter_steps_commutator(h, 1.0, 0.1, order=2) == 1

    def test_tighter_than_naive_second_order(self):
        """Test that commutator bound is never looser than naive."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        eps = 0.01
        time = 1.0
        n_naive = trotter_steps_naive(h, time, eps, order=2)
        n_comm = trotter_steps_commutator(h, time, eps, order=2)
        assert n_comm <= n_naive

    def test_minimum_one_second_order(self):
        """Test that result is at least 1 for commuting Hamiltonian."""
        h = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        assert trotter_steps_commutator(h, 1.0, 100.0, order=2) == 1

    def test_time_scaling_second_order(self):
        """Test that step count scales with t^(3/2)."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        n1 = trotter_steps_commutator(h, 1.0, 0.1, order=2)
        n2 = trotter_steps_commutator(h, 2.0, 0.1, order=2)
        # n2 should be approximately 2**1.5 * n1 (t^1.5 scaling)
        assert (n1 * math.ceil(2**1.5)) // n2 == 1  # Allow for ceiling effects

    # Higher-order Trotter tests.

    def test_all_commuting_higher_order(self):
        """Test with all commuting terms."""
        # XI and IX commute: commutator bound = 0, N = 1
        h = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        assert trotter_steps_commutator(h, 1.0, 0.1, order=4) == 1

    def test_tighter_than_naive_higher_order(self):
        """Test that commutator bound is never looser than naive."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        eps = 0.01
        time = 1.0
        n_naive = trotter_steps_naive(h, time, eps, order=4)
        n_comm = trotter_steps_commutator(h, time, eps, order=4)
        assert n_comm <= n_naive

    def test_minimum_one_higher_order(self):
        """Test that result is at least 1 for commuting Hamiltonian."""
        h = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        assert trotter_steps_commutator(h, 1.0, 100.0, order=4) == 1

    def test_time_scaling_higher_order(self):
        """Test that step count scales with t^(1+1/4)."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        n1 = trotter_steps_commutator(h, 1.0, 0.1, order=4)
        n2 = trotter_steps_commutator(h, 2.0, 0.1, order=4)
        # n2 should be approximately 2**1.25 * n1 (t^1.25 scaling)
        assert (n1 * math.ceil(2**1.25)) // n2 == 1  # Allow for ceiling effects

    def test_order_3_raises(self):
        """Test that order > 2 raises NotImplementedError."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        with pytest.raises(NotImplementedError, match="order 3"):
            trotter_steps_commutator(h, 1.0, 0.1, order=3)
