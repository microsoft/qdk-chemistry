"""Tests for Zassenhaus error bound estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from fractions import Fraction

import pytest

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus_error import (
    zassenhaus_steps_naive,
)
from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.utils.zassenhaus_generation import zassenhaus_commutator_plan


class TestZassenhausStepsNaive:
    """Tests for the zassenhaus_steps_naive function."""

    def test_first_order_uses_omitted_coefficient_sum(self):
        """Test first-order naive bound includes the C2 coefficient sum."""
        # For two leaves, C2 has coefficient sum 1/2.
        # N = ceil((1/2 * 2 * 2^2) * 1^2 / 0.1) = 40
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        assert zassenhaus_steps_naive(h, 1.0, 0.1, order=1) == 40

    def test_second_order_uses_omitted_coefficient_sum(self):
        """Test second-order naive bound includes the C3 coefficient sum."""
        # For two leaves, C3 has coefficient sum 1/6 + 1/3 = 1/2.
        # N = ceil(sqrt(1/2 * 4 * 2^3) / sqrt(0.1)) = 13
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        assert zassenhaus_steps_naive(h, 1.0, 0.1, order=2) == 13

    def test_accepts_precomputed_commutator_exponents(self):
        """Test callers can pass precomputed exponents through order p + 1."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        planned_exponents, _ = zassenhaus_commutator_plan((0, 1), max_order=2)

        assert (
            zassenhaus_steps_naive(
                h,
                1.0,
                0.1,
                order=1,
                commutator_exponents=planned_exponents,
            )
            == 40
        )

    def test_uses_supplied_coefficients_without_regenerating(self):
        """Test supplied exponent coefficients control the naive prefactor."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        commutator_exponents = {2: {"custom": Fraction(3, 2)}}

        assert (
            zassenhaus_steps_naive(
                h,
                1.0,
                0.1,
                order=1,
                commutator_exponents=commutator_exponents,
            )
            == 120
        )

    def test_single_term_is_exact(self):
        """Test one Hamiltonian term needs only one exact Zassenhaus step."""
        h = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        assert zassenhaus_steps_naive(h, 1.0, 0.1, order=1) == 1

    def test_missing_omitted_exponent_raises(self):
        """Test precomputed exponents must include the first omitted order."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])

        with pytest.raises(ValueError, match="C_3"):
            zassenhaus_steps_naive(h, 1.0, 0.1, order=2, commutator_exponents={2: {}})

