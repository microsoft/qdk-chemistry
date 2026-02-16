"""Tests for Trotter error bound estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import pytest

from qdk_chemistry.algorithms.time_evolution.builder.trotter_error import (
    trotter_steps_commutator,
    trotter_steps_naive,
)


class TestTrotterStepsNaive:
    """Tests for the trotter_steps_naive function."""

    def test_basic(self):
        """Test basic naive bound computation."""
        # N = ceil((2^2 * 1^2) / 0.1) = ceil(40) = 40
        assert trotter_steps_naive(2.0, 1.0, 0.1) == 40

    def test_minimum_one(self):
        """Test that result is at least 1."""
        # Very large epsilon should still give 1
        assert trotter_steps_naive(1.0, 0.001, 1e6) == 1

    def test_small_accuracy(self):
        """Test with very small accuracy."""
        # N = ceil((1^2 * 1^2) / 0.001) = ceil(1000) = 1000
        assert trotter_steps_naive(1.0, 1.0, 0.001) == 1000

    def test_large_time(self):
        """Test with large evolution time."""
        # N = ceil((1^2 * 10^2) / 1.0) = ceil(100) = 100
        assert trotter_steps_naive(1.0, 10.0, 1.0) == 100

    def test_zero_accuracy_raises(self):
        """Test that zero accuracy raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            trotter_steps_naive(1.0, 1.0, 0.0)

    def test_negative_accuracy_raises(self):
        """Test that negative accuracy raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            trotter_steps_naive(1.0, 1.0, -0.1)


class TestTrotterStepsCommutator:
    """Tests for the trotter_steps_commutator function."""

    def test_anticommuting_pair(self):
        """Test with an anticommuting pair."""
        # X and Z anticommute: commutator bound = 2
        # N = ceil(2 * 1^2 / (2 * 0.1)) = ceil(10) = 10
        assert trotter_steps_commutator(["X", "Z"], [1.0, 1.0], 1.0, 0.1) == 10

    def test_all_commuting(self):
        """Test with all commuting terms."""
        # XI and IX commute: commutator bound = 0, N = 1
        assert trotter_steps_commutator(["XI", "IX"], [1.0, 1.0], 1.0, 0.1) == 1

    def test_tighter_than_naive(self):
        """Test that commutator bound is never looser than naive."""
        labels = ["X", "Z"]
        coeffs = [1.0, 1.0]
        time = 1.0
        eps = 0.01
        one_norm = 2.0
        n_naive = trotter_steps_naive(one_norm, time, eps)
        n_comm = trotter_steps_commutator(labels, coeffs, time, eps)
        assert n_comm <= n_naive

    def test_minimum_one(self):
        """Test that result is at least 1 for commuting Hamiltonian."""
        assert trotter_steps_commutator(["XI", "IX"], [1.0, 1.0], 1.0, 100.0) == 1

    def test_time_scaling(self):
        """Test that step count scales with t^2."""
        n1 = trotter_steps_commutator(["X", "Z"], [1.0, 1.0], 1.0, 0.1)
        n2 = trotter_steps_commutator(["X", "Z"], [1.0, 1.0], 2.0, 0.1)
        # n2 should be approximately 4 * n1 (t^2 scaling)
        assert n2 == math.ceil(4 * (2.0 / 0.2))  # 40
        assert n2 >= 4 * n1 - 1  # Allow for ceiling effects

    def test_zero_accuracy_raises(self):
        """Test that zero accuracy raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            trotter_steps_commutator(["X", "Z"], [1.0, 1.0], 1.0, 0.0)
