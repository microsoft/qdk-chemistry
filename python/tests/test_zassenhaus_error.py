"""Tests for Zassenhaus error bound estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from fractions import Fraction

import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus_error import (
    zassenhaus_steps_commutator,
    zassenhaus_steps_naive,
)
from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.utils.zassenhaus_generation import zassenhaus_commutator_plan


def _pauli_product_matrix(label: str) -> np.ndarray:
    mapping = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    result = np.array([[1]], dtype=complex)
    for char in label:
        result = np.kron(result, mapping[char])
    return result


def _first_order_product_formula_error(hamiltonian: QubitHamiltonian, *, time: float, steps: int) -> float:
    step_unitary = np.eye(2**hamiltonian.num_qubits, dtype=complex)
    for label, coeff in zip(hamiltonian.pauli_strings, hamiltonian.coefficients, strict=True):
        step_unitary = scipy.linalg.expm(
            -1j * time / steps * complex(coeff).real * _pauli_product_matrix(label)
        ) @ step_unitary

    approximate = np.linalg.matrix_power(step_unitary, steps)
    exact = scipy.linalg.expm(-1j * time * hamiltonian.to_matrix())
    return float(np.linalg.norm(approximate - exact, ord=2))


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


class TestZassenhausStepsCommutator:
    """Tests for the zassenhaus_steps_commutator function."""

    def test_first_order_two_anticommuting_terms_uses_actual_commutator_norm(self):
        """Test first-order commutator bound evaluates the omitted C2 exponent."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])

        assert zassenhaus_steps_commutator(h, 1.0, 0.1, order=1) == 10

    def test_commuting_terms_are_exact(self):
        """Test mutually commuting Pauli strings need only one Zassenhaus step."""
        h = QubitHamiltonian(pauli_strings=["ZI", "IZ", "ZZ"], coefficients=[0.5, -0.25, 0.125])

        assert zassenhaus_steps_commutator(h, 4.0, 1e-6, order=1) == 1
        assert zassenhaus_steps_commutator(h, 4.0, 1e-6, order=2) == 1

    def test_second_order_bound_uses_nested_commutator_cancellations(self):
        """Test the second-order commutator bound is tighter than the naive bound."""
        h = QubitHamiltonian(pauli_strings=["XI", "ZI", "IX", "IZ", "ZZ"], coefficients=[0.7, -0.2, 0.5, 0.3, -0.4])

        assert zassenhaus_steps_commutator(h, 1.0, 0.01, order=2) < zassenhaus_steps_naive(
            h,
            1.0,
            0.01,
            order=2,
        )

    def test_first_order_bound_controls_two_term_actual_error(self):
        """Test the returned step count bounds a non-commuting two-term evolution."""
        h = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[0.8, -0.35])
        target_accuracy = 0.01
        steps = zassenhaus_steps_commutator(h, 0.7, target_accuracy, order=1)

        assert _first_order_product_formula_error(h, time=0.7, steps=steps) <= target_accuracy

    def test_first_order_bound_controls_meaningful_multi_qubit_error(self):
        """Test the returned step count bounds a small chemistry-like Pauli Hamiltonian."""
        h = QubitHamiltonian(
            pauli_strings=["ZI", "IZ", "XX", "YY", "ZZ"],
            coefficients=[-0.6, 0.25, 0.18, -0.12, 0.08],
        )
        target_accuracy = 0.02
        steps = zassenhaus_steps_commutator(h, 0.5, target_accuracy, order=1)

        assert _first_order_product_formula_error(h, time=0.5, steps=steps) <= target_accuracy
