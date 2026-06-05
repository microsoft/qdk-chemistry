"""Tests for Zassenhaus Constructor and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus import Zassenhaus
from qdk_chemistry.data import FlatPartition, QubitHamiltonian, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def _pauli_matrix(label: str) -> np.ndarray:
    """Helper to get Pauli matrix from label."""
    if label == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if label == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if label == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError(f"Unsupported Pauli label: {label}")


def _pauli_product_matrix(label: str) -> np.ndarray:
    """Helper to build a two-qubit Pauli matrix from a label string."""
    mapping = {
        "I": np.eye(2, dtype=complex),
        "X": _pauli_matrix("X"),
        "Y": _pauli_matrix("Y"),
        "Z": _pauli_matrix("Z"),
    }
    result = np.array([[1]], dtype=complex)
    for char in label:
        result = np.kron(result, mapping[char])
    return result


def _pauli_label_from_map(pauli_term: dict[int, str], num_qubits: int) -> str:
    """Helper to convert sparse qubit-index maps to QubitHamiltonian label order."""
    return "".join(pauli_term.get(i, "I") for i in reversed(range(num_qubits)))


class TestZassenhaus:
    """Tests for the Zassenhaus class."""

    def test_name(self):
        """Test the name method of Zassenhaus."""
        builder = Zassenhaus()
        assert builder.name() == "zassenhaus"

    def test_type_name(self):
        """Test the type_name method of Zassenhaus."""
        builder = Zassenhaus()
        assert builder.type_name() == "hamiltonian_unitary_builder"

    def test_can_create_via_registry(self):
        """Test that Zassenhaus can be created via the algorithm registry."""
        builder = create("hamiltonian_unitary_builder", "zassenhaus")
        assert isinstance(builder, Zassenhaus)

    def test_can_create_with_settings(self):
        """Test that Zassenhaus can be created with custom settings."""
        builder = create(
            "hamiltonian_unitary_builder",
            "zassenhaus",
            order=4,
            num_divisions=3,
            time=0.2,
            weight_threshold=1e-10,
        )

        assert builder.settings().get("order") == 4
        assert builder.settings().get("num_divisions") == 3
        assert builder.settings().get("time") == 0.2
        assert builder.settings().get("weight_threshold") == 1e-10

    def test_registry_builder_returns_pauli_product_formula_container(self):
        """Test registry-created Zassenhaus returns the standard product-formula container shape."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "ZZ"],
            coefficients=[2.0, 1.0],
        )

        builder = create("hamiltonian_unitary_builder", "zassenhaus", order=2, num_divisions=4, time=0.2)
        unitary = builder.run(hamiltonian)

        assert isinstance(unitary, UnitaryRepresentation)
        container = unitary.get_container()
        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == hamiltonian.num_qubits
        assert container.step_reps == 4
        assert len(container.step_terms) > 0

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

        # Expected second-order Zassenhaus expansion:
        #   exp(A + B) => exp(A) exp(B) exp(-[A, B]/2), emitted in execution order.
        u_zassenhaus = np.eye(2, dtype=complex)
        for term in container.step_terms:
            pauli_label = next(iter(term.pauli_term.values()))
            pauli_matrix = _pauli_matrix(pauli_label)
            u_zassenhaus = scipy.linalg.expm(-1j * term.angle * pauli_matrix) @ u_zassenhaus

        hamiltonian_matrix = 1.5 * _pauli_matrix("X") + 0.5 * _pauli_matrix("Z")
        u_exact = scipy.linalg.expm(-1j * t * hamiltonian_matrix)

        assert container.step_terms[0].pauli_term == {0: "Y"}
        assert container.step_terms[1].pauli_term == {0: "Z"}
        assert container.step_terms[2].pauli_term == {0: "X"}
        assert np.isclose(
            container.step_terms[0].angle,
            1.5 * 0.5 * t**2,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert container.step_terms[1].angle == 0.5 * t
        assert container.step_terms[2].angle == 1.5 * t
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

        # Expected fourth-order Zassenhaus expansion:
        #   exp(A + B) => exp(A) exp(B) exp(C2) exp(C3) exp(C4),
        # emitted in execution order. This two-term Pauli algebra closes on
        # XI, ZZ, and YZ, so the fourth-order product emits seven exponentials.
        u_zassenhaus = np.eye(4, dtype=complex)
        for term in container.step_terms:
            pauli_label = _pauli_label_from_map(term.pauli_term, num_qubits=2)
            pauli_matrix = _pauli_product_matrix(pauli_label)
            u_zassenhaus = scipy.linalg.expm(-1j * term.angle * pauli_matrix) @ u_zassenhaus

        hamiltonian_matrix = 0.75 * _pauli_product_matrix("XI") + 0.25 * _pauli_product_matrix("ZZ")
        u_exact = scipy.linalg.expm(-1j * t * hamiltonian_matrix)

        assert container.step_reps == 1
        assert len(container.step_terms) == 7
        expected_terms = [
            ({0: "Z", 1: "Y"}, -(3 / 256) * t**4),
            ({0: "Z", 1: "Y"}, -(9 / 256) * t**4),
            ({1: "X"}, (1 / 16) * t**3),
            ({0: "Z", 1: "Z"}, -(3 / 32) * t**3),
            ({0: "Z", 1: "Y"}, (3 / 16) * t**2),
            ({0: "Z", 1: "Z"}, 0.25 * t),
            ({1: "X"}, 0.75 * t),
        ]
        for term, (expected_pauli_term, expected_angle) in zip(container.step_terms, expected_terms, strict=True):
            assert term.pauli_term == expected_pauli_term
            assert np.isclose(
                term.angle,
                expected_angle,
                atol=float_comparison_absolute_tolerance,
                rtol=float_comparison_relative_tolerance,
            )
        error_actual = np.linalg.norm(u_zassenhaus - u_exact, ord=2)
        assert error_actual < 0.02

    def test_zassenhaus_four_term_two_group_fourth_order_example(self):
        """Correctness check for fourth-order Zassenhaus decomposition with two commuting groups."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "IX", "ZZ", "YY"],
            coefficients=[0.7, -0.2, 0.3, 0.11],
            term_partition=FlatPartition(strategy="commuting", groups=((0, 1), (2, 3))),
        )

        t = 0.1
        builder = Zassenhaus(num_divisions=1, order=4, time=t)
        unitary = builder.run(hamiltonian)
        container = unitary.get_container()

        u_zassenhaus = np.eye(4, dtype=complex)
        for term in container.step_terms:
            pauli_label = _pauli_label_from_map(term.pauli_term, num_qubits=2)
            pauli_matrix = _pauli_product_matrix(pauli_label)
            u_zassenhaus = scipy.linalg.expm(-1j * term.angle * pauli_matrix) @ u_zassenhaus

        hamiltonian_matrix = (
            0.7 * _pauli_product_matrix("XI")
            - 0.2 * _pauli_product_matrix("IX")
            + 0.3 * _pauli_product_matrix("ZZ")
            + 0.11 * _pauli_product_matrix("YY")
        )
        u_exact = scipy.linalg.expm(-1j * t * hamiltonian_matrix)

        assert container.step_reps == 1
        assert len(container.step_terms) == 14
        expected_terms = [
            ({0: "Z", 1: "Y"}, -(81823 / 2500000) * t**4),
            ({0: "Y", 1: "Z"}, (292997 / 10000000) * t**4),
            ({0: "Z", 1: "Y"}, -(4033 / 75_000) * t**4),
            ({0: "Y", 1: "Z"}, (13757 / 300_000) * t**4),
            ({0: "X"}, -(3331 / 37_500) * t**3),
            ({1: "X"}, (8467 / 75_000) * t**3),
            ({0: "Y", 1: "Y"}, -(1423 / 15_000) * t**3),
            ({0: "Z", 1: "Z"}, -(949 / 7_500) * t**3),
            ({0: "Z", 1: "Y"}, (29 / 125) * t**2),
            ({0: "Y", 1: "Z"}, -(137 / 1000) * t**2),
            ({0: "Y", 1: "Y"}, 0.11 * t),
            ({0: "Z", 1: "Z"}, 0.3 * t),
            ({0: "X"}, -0.2 * t),
            ({1: "X"}, 0.7 * t),
        ]
        for term, (expected_pauli_term, expected_angle) in zip(container.step_terms, expected_terms, strict=True):
            assert term.pauli_term == expected_pauli_term
            assert np.isclose(
                term.angle,
                expected_angle,
                atol=float_comparison_absolute_tolerance,
                rtol=float_comparison_relative_tolerance,
            )
        error_actual = np.linalg.norm(u_zassenhaus - u_exact, ord=2)
        assert error_actual < 1e-5
