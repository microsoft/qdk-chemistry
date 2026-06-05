"""Tests for Zassenhaus Constructor and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus import Zassenhaus
from qdk_chemistry.data import QubitHamiltonian, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def _pauli_matrix(label: str) -> np.ndarray:
    """Helper to get Pauli matrix from label."""
    if label == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if label == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError(f"Unsupported Pauli label: {label}")


def _pauli_product_matrix(label: str) -> np.ndarray:
    """Helper to build a two-qubit Pauli matrix from a label string."""
    mapping = {
        "I": np.eye(2, dtype=complex),
        "X": _pauli_matrix("X"),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": _pauli_matrix("Z"),
    }
    result = np.array([[1]], dtype=complex)
    for char in label:
        result = np.kron(result, mapping[char])
    return result


class TestZassenhaus:
    """Tests for the Zassenhaus class."""

    def test_name(self):
        """Test the name method of Zassenhaus."""
        builder = Zassenhaus()
        assert builder.name() == "zassenhaus"

    def test_single_step_construction(self):
        """Test construction of time evolution unitary with a single Zassenhaus step."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = Zassenhaus(num_divisions=1, time=0.2)
        unitary = builder.run(hamiltonian)

        assert isinstance(unitary, UnitaryRepresentation)
        container = unitary.get_container()

        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == 1
        assert container.step_reps == 1
        assert len(container.step_terms) == 2

    def test_multiple_zassenhaus_steps(self):
        """Test construction of time evolution unitary with multiple Zassenhaus steps."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "ZZ"],
            coefficients=[2.0, 1.0],
        )

        builder = Zassenhaus(num_divisions=4, time=0.2)
        unitary = builder.run(hamiltonian)

        container = unitary.get_container()

        # dt = 0.2 / 4 = 0.05
        assert container.step_reps == 4
        assert np.isclose(
            container.step_terms[0].angle,
            0.1,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(
            container.step_terms[1].angle,
            0.05,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_basic_decomposition(self):
        """Test basic decomposition of a qubit Hamiltonian."""
        builder = Zassenhaus()
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])

        terms = builder._decompose_zassenhaus_step(hamiltonian, time=2.0)

        assert len(terms) == 2
        assert terms[0] == ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=2.0)
        assert terms[1] == ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=1.0)

    def test_filters_small_coefficients(self):
        """Test that terms with small coefficients are filtered out."""
        builder = Zassenhaus()
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[1e-15, 1.0],
        )

        terms = builder._decompose_zassenhaus_step(hamiltonian, time=1.0, atol=1e-12)

        assert len(terms) == 1
        assert terms[0].pauli_term == {0: "Z"}

    def test_rejects_non_hermitian(self):
        """Test that non-Hermitian Hamiltonians raise a ValueError."""
        builder = Zassenhaus()
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X"],
            coefficients=[1.0 + 1.0j],
        )

        with pytest.raises(ValueError, match="Non-Hermitian"):
            builder._decompose_zassenhaus_step(hamiltonian, time=1.0)

    def test_zassenhaus_x_z_example(self):
        """Correctness check for first-order Zassenhaus decomposition."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.5, 0.5])

        t = 0.1
        builder = Zassenhaus(num_divisions=1, order=1, time=t)
        unitary = builder.run(hamiltonian)
        container = unitary.get_container()

        # Expected first-order expansion:
        #   exp(-i t (1.5 X + 0.5 Z)) ≈ exp(-i 1.5 t X) exp(-i 0.5 t Z)
        u_zassenhaus = np.eye(2, dtype=complex)
        for term in container.step_terms:
            pauli_label = next(iter(term.pauli_term.values()))
            pauli_matrix = _pauli_matrix(pauli_label)
            u_zassenhaus = scipy.linalg.expm(-1j * term.angle * pauli_matrix) @ u_zassenhaus

        hamiltonian_matrix = 1.5 * _pauli_matrix("X") + 0.5 * _pauli_matrix("Z")
        u_exact = scipy.linalg.expm(-1j * t * hamiltonian_matrix)

        assert container.step_terms[0].pauli_term == {0: "X"}
        assert container.step_terms[1].pauli_term == {0: "Z"}
        assert container.step_terms[0].angle == 1.5 * t
        assert container.step_terms[1].angle == 0.5 * t
        error_actual = np.linalg.norm(u_zassenhaus - u_exact, ord=2)
        assert error_actual < 0.05

    def test_single_step_construction_second_order(self):
        """Test construction of time evolution unitary with a single Zassenhaus step."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.5, 0.5])
        builder = Zassenhaus(num_divisions=1, order=2, time=0.2)
        unitary = builder.run(hamiltonian)

        assert isinstance(unitary, UnitaryRepresentation)
        container = unitary.get_container()

        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == 1
        assert container.step_reps == 1
        assert len(container.step_terms) == 3

    def test_zassenhaus_x_z_example_second_order(self):
        """Correctness check for second-order Zassenhaus decomposition."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.5, 0.5])

        t = 0.1
        builder = Zassenhaus(num_divisions=1, order=2, time=t)
        unitary = builder.run(hamiltonian)
        container = unitary.get_container()

        # Expected second-order expansion:
        #   exp(-i t (1.5 X + 0.5 Z)) => exp(-i 0.75 t X) exp(-i 0.5 t Z) exp(-i 0.75 t X)
        u_zassenhaus = np.eye(2, dtype=complex)
        for term in container.step_terms:
            pauli_label = next(iter(term.pauli_term.values()))
            pauli_matrix = _pauli_matrix(pauli_label)
            u_zassenhaus = scipy.linalg.expm(-1j * term.angle * pauli_matrix) @ u_zassenhaus

        hamiltonian_matrix = 1.5 * _pauli_matrix("X") + 0.5 * _pauli_matrix("Z")
        u_exact = scipy.linalg.expm(-1j * t * hamiltonian_matrix)

        assert container.step_terms[0].pauli_term == {0: "X"}
        assert container.step_terms[1].pauli_term == {0: "Z"}
        assert container.step_terms[2].pauli_term == {0: "X"}
        assert container.step_terms[0].angle == 0.75 * t
        assert container.step_terms[1].angle == 0.5 * t
        assert container.step_terms[2].angle == 0.75 * t
        error_actual = np.linalg.norm(u_zassenhaus - u_exact, ord=2)
        assert error_actual < 0.01

    def test_zassenhaus_two_qubit_fourth_order_example(self):
        """Correctness check for fourth-order Zassenhaus decomposition on a two-qubit Hamiltonian."""
        # This pair closes under commutators:
        #   [XI, ZZ] ~ YZ, [XI, YZ] ~ ZZ, [ZZ, YZ] ~ XI up to proportionality.
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "ZZ"], coefficients=[0.75, 0.25])

        t = 0.1
        builder = Zassenhaus(num_divisions=1, order=4, time=t)
        unitary = builder.run(hamiltonian)
        container = unitary.get_container()

        # Expected fourth-order expansion:
        #   exp(-i t (0.75 XI + 0.25 ZZ)) => product of 11 exponentials
        #   with nested commutator corrections through O(t^4).
        u_zassenhaus = np.eye(4, dtype=complex)
        for term in container.step_terms:
            pauli_label = "".join(term.pauli_term.get(i, "I") for i in range(2))
            pauli_matrix = _pauli_product_matrix(pauli_label)
            u_zassenhaus = scipy.linalg.expm(-1j * term.angle * pauli_matrix) @ u_zassenhaus

        hamiltonian_matrix = 0.75 * _pauli_product_matrix("XI") + 0.25 * _pauli_product_matrix("ZZ")
        u_exact = scipy.linalg.expm(-1j * t * hamiltonian_matrix)

        assert container.step_reps == 1
        assert len(container.step_terms) == 11
        assert container.step_terms[0].pauli_term == {1: "X"}
        assert container.step_terms[1].pauli_term == {1: "Z", 0: "Z"}
        error_actual = np.linalg.norm(u_zassenhaus - u_exact, ord=2)
        assert error_actual < 0.02
