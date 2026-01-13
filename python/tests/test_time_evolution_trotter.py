"""Tests for FirstOrderTrotterConstructor and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms.time_evolution.constructor.trotter import Trotter
from qdk_chemistry.data import QubitHamiltonian, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestPauliLabelToMap:
    """Tests for the _pauli_label_to_map helper function."""

    def test_identity_only(self):
        """Test that identity-only labels return an empty mapping."""
        constructor = Trotter()
        assert constructor._pauli_label_to_map("III") == {}

    def test_single_pauli(self):
        """Test labels with a single non-identity Pauli."""
        constructor = Trotter()
        assert constructor._pauli_label_to_map("X") == {0: "X"}
        assert constructor._pauli_label_to_map("IZ") == {0: "Z"}

    def test_multiple_paulis(self):
        """Test labels with multiple non-identity Paulis."""
        # label is little-endian: rightmost char -> qubit 0
        constructor = Trotter()
        mapping = constructor._pauli_label_to_map("XYZ")
        assert mapping == {0: "Z", 1: "Y", 2: "X"}


class TestTrotter:
    """Tests for the Trotter class."""

    def test_name(self):
        """Test the name method of Trotter."""
        ctor = Trotter()
        assert ctor.name() == "trotter"

    def test_single_step_construction(self):
        """Test construction of time evolution unitary with a single Trotter step."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        ctor = Trotter(num_trotter_steps=1)
        unitary = ctor.run(hamiltonian, time=0.2)

        assert isinstance(unitary, TimeEvolutionUnitary)
        container = unitary.get_container()

        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == 1
        assert container.step_reps == 1
        assert len(container.step_terms) == 2

    def test_multiple_trotter_steps(self):
        """Test construction of time evolution unitary with multiple Trotter steps."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "ZZ"],
            coefficients=[2.0, 1.0],
        )

        ctor = Trotter(num_trotter_steps=4)
        unitary = ctor.run(hamiltonian, time=0.2)

        container = unitary.get_container()

        # dt = 0.2 / 4 = 0.05
        assert container.step_reps == 4
        assert np.isclose(
            container.step_terms[0].angle,
            0.1,  # angle = dt * coeff = 0.05 * 2.0
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(
            container.step_terms[1].angle,
            0.05,  # angle = dt * coeff = 0.05 * 1.0
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_basic_decomposition(self):
        """Test basic decomposition of a qubit Hamiltonian."""
        ctor = Trotter()
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])

        terms = ctor._decompose_trotter_step(hamiltonian, time=2.0)

        assert len(terms) == 2

        assert terms[0] == ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=2.0)
        assert terms[1] == ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=1.0)

    def test_filters_small_coefficients(self):
        """Test that terms with small coefficients are filtered out."""
        ctor = Trotter()
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[1e-15, 1.0],
        )

        terms = ctor._decompose_trotter_step(hamiltonian, time=1.0, atol=1e-12)

        assert len(terms) == 1
        assert terms[0].pauli_term == {0: "Z"}

    def test_rejects_non_hermitian(self):
        """Test that non-Hermitian Hamiltonians raise a ValueError."""
        ctor = Trotter()
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X"],
            coefficients=[1.0 + 1.0j],
        )

        with pytest.raises(ValueError, match="Non-Hermitian"):
            ctor._decompose_trotter_step(hamiltonian, time=1.0)

    def test_not_implemented_order(self):
        """Test that unsupported Trotter orders raise NotImplementedError."""
        ctor = Trotter(order=2)
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])

        with pytest.raises(NotImplementedError, match="Only first-order Trotter decomposition is currently supported."):
            ctor.run(hamiltonian, time=1.0)

    def test_trotter_x_z_example(self):
        """Correctness check for first-order Trotter decomposition."""
        # Hamiltonian H = X + Z
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])

        ctor = Trotter(num_trotter_steps=1)
        t = 0.1
        unitary = ctor.run(hamiltonian, time=t)
        container = unitary.get_container()

        def _pauli_matrix(label: str) -> np.ndarray:
            """Helper to get Pauli matrix from label."""
            if label == "X":
                return np.array([[0, 1], [1, 0]], dtype=complex)
            if label == "Z":
                return np.array([[1, 0], [0, -1]], dtype=complex)
            raise ValueError(f"Unsupported Pauli label: {label}")

        # Build Trotter unitary matrix
        u_trot = np.eye(2, dtype=complex)
        for term in container.step_terms:
            pauli_label = next(iter(term.pauli_term.values()))
            pauli_matrix = _pauli_matrix(pauli_label)
            u_trot = scipy.linalg.expm(-1j * term.angle * pauli_matrix) @ u_trot

        # Exact unitary
        hamiltonian_matrix = _pauli_matrix("X") + _pauli_matrix("Z")
        u_exact = scipy.linalg.expm(-1j * t * hamiltonian_matrix)

        assert container.step_terms[0].pauli_term == {0: "X"}
        assert container.step_terms[1].pauli_term == {0: "Z"}
        assert container.step_terms[0].angle == t  # angle for X term
        assert container.step_terms[1].angle == t  # angle for Z term
        # Compare: the error should scales as O(t^2) for first-order Trotter
        assert np.allclose(u_trot, u_exact, atol=t**2)
