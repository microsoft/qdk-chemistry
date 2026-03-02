"""Tests for Trotter Constructor and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter
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
        builder = Trotter()
        assert builder._pauli_label_to_map("III") == {}

    def test_single_pauli(self):
        """Test labels with a single non-identity Pauli."""
        builder = Trotter()
        assert builder._pauli_label_to_map("X") == {0: "X"}
        assert builder._pauli_label_to_map("IZ") == {0: "Z"}

    def test_multiple_paulis(self):
        """Test labels with multiple non-identity Paulis."""
        # label is little-endian: rightmost char -> qubit 0
        builder = Trotter()
        mapping = builder._pauli_label_to_map("XYZ")
        assert mapping == {0: "Z", 1: "Y", 2: "X"}


class TestTrotter:
    """Tests for the Trotter class."""

    def test_name(self):
        """Test the name method of Trotter."""
        builder = Trotter()
        assert builder.name() == "trotter"

    def test_single_step_construction(self):
        """Test construction of time evolution unitary with a single Trotter step."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = Trotter(num_divisions=1)
        unitary = builder.run(hamiltonian, time=0.2)

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

        builder = Trotter(num_divisions=4)
        unitary = builder.run(hamiltonian, time=0.2)

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
        builder = Trotter()
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])

        terms = builder._decompose_trotter_step(hamiltonian, time=2.0)

        assert len(terms) == 2

        assert terms[0] == ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=2.0)
        assert terms[1] == ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=1.0)

    def test_filters_small_coefficients(self):
        """Test that terms with small coefficients are filtered out."""
        builder = Trotter()
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[1e-15, 1.0],
        )

        terms = builder._decompose_trotter_step(hamiltonian, time=1.0, atol=1e-12)

        assert len(terms) == 1
        assert terms[0].pauli_term == {0: "Z"}

    def test_rejects_non_hermitian(self):
        """Test that non-Hermitian Hamiltonians raise a ValueError."""
        builder = Trotter()
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X"],
            coefficients=[1.0 + 1.0j],
        )

        with pytest.raises(ValueError, match="Non-Hermitian"):
            builder._decompose_trotter_step(hamiltonian, time=1.0)

    def test_not_implemented_order(self):
        """Test that unsupported Trotter orders raise NotImplementedError."""
        builder = Trotter(order=3)
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])

        with pytest.raises(NotImplementedError, match="Only orders 1 or 2 are currently supported."):
            builder.run(hamiltonian, time=1.0)

    def test_trotter_x_z_example(self):
        """Correctness check for first-order Trotter decomposition."""
        # Hamiltonian H = X + Z
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])

        builder = Trotter(num_divisions=1)
        t = 0.1
        unitary = builder.run(hamiltonian, time=t)
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
        # Compare: the error should scale as O(t^2) for first-order Trotter
        assert np.allclose(u_trot, u_exact, atol=t**2)

    # Second-order Trotter tests.
    def test_single_step_construction_second_order(self):
        """Test construction of time evolution unitary with a single Trotter step."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = Trotter(num_divisions=1, order=2)
        unitary = builder.run(hamiltonian, time=0.2)

        assert isinstance(unitary, TimeEvolutionUnitary)
        container = unitary.get_container()

        assert isinstance(container, PauliProductFormulaContainer)
        assert container.num_qubits == 1
        assert container.step_reps == 1
        assert len(container.step_terms) == 3

    def test_multiple_trotter_steps_second_order(self):
        """Test construction of time evolution unitary with multiple Trotter steps."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "ZZ"],
            coefficients=[2.0, 3.0],
        )

        builder = Trotter(num_divisions=4, order=2)
        unitary = builder.run(hamiltonian, time=0.2)

        container = unitary.get_container()

        # dt = 0.2 / 4 = 0.05
        assert container.step_reps == 4
        assert np.isclose(
            container.step_terms[0].angle,
            0.05,  # angle = dt/2 * coeff = 0.025 * 2.0
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(
            container.step_terms[1].angle,
            0.15,  # angle = dt * coeff = 0.05 * 3.0
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        assert np.isclose(
            container.step_terms[2].angle,
            0.05,  # angle = dt/2 * coeff = 0.025 * 2.0
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_basic_decomposition_second_order(self):
        """Test basic decomposition of a qubit Hamiltonian."""
        builder = Trotter(order=2)
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[3.0, 0.5])

        terms = builder._decompose_trotter_step(hamiltonian, time=2.0)

        assert len(terms) == 3

        assert terms[0] == ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=3.0)
        assert terms[1] == ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=1.0)
        assert terms[2] == ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=3.0)

    def test_filters_small_coefficients_second_order(self):
        """Test that terms with small coefficients are filtered out."""
        builder = Trotter(order=2)
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[1e-15, 1.0],
        )

        terms = builder._decompose_trotter_step(hamiltonian, time=1.0, atol=1e-12)

        assert len(terms) == 1
        assert terms[0].pauli_term == {0: "Z"}

    def test_trotter_x_z_example_second_order(self):
        """Correctness check for second-order Trotter decomposition."""
        # Hamiltonian H = X + Z
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])

        builder = Trotter(num_divisions=1, order=2)
        t = 0.1
        unitary = builder.run(hamiltonian, time=t)
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
        assert container.step_terms[2].pauli_term == {0: "X"}
        assert container.step_terms[0].angle == t / 2  # angle for X term
        assert container.step_terms[1].angle == t  # angle for Z term
        assert container.step_terms[2].angle == t / 2  # angle for X term
        # Compare: the error should scale as O(t^2) for second-order Trotter
        assert np.allclose(u_trot, u_exact, atol=t**3)


class TestTrotterAccuracyAware:
    """Tests for accuracy-aware Trotter parameterization."""

    def test_target_accuracy_commutator_bound(self):
        """Test that target_accuracy with commutator bound computes correct step count."""
        # H = X + Z, X and Z anticommute -> commutator bound = 2
        # N = ceil(2 * t^2 / (2 * eps)) = ceil(t^2 / eps)
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Trotter(target_accuracy=0.01)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        # N = ceil(2 * 1 / (2 * 0.01)) = 100
        assert container.step_reps == 100

    def test_target_accuracy_naive_bound(self):
        """Test that target_accuracy with naive bound computes correct step count."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Trotter(target_accuracy=0.01, error_bound="naive")
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        # one_norm = 2, N = ceil(4 / 0.01) = 400
        assert container.step_reps == 400

    def test_commutator_tighter_than_naive(self):
        """Test that commutator bound gives fewer steps than naive bound."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        eps = 0.01
        time = 1.0
        builder_comm = Trotter(target_accuracy=eps, error_bound="commutator")
        builder_naive = Trotter(target_accuracy=eps, error_bound="naive")
        n_comm = builder_comm.run(hamiltonian, time=time).get_container().step_reps
        n_naive = builder_naive.run(hamiltonian, time=time).get_container().step_reps
        assert n_comm <= n_naive

    def test_commuting_hamiltonian_needs_one_step(self):
        """Test that a fully commuting Hamiltonian needs only 1 Trotter step."""
        # XI and IX commute -> commutator bound = 0 -> N = 1
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        builder = Trotter(target_accuracy=0.01)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        assert container.step_reps == 1

    def test_manual_steps_as_lower_bound(self):
        """Test that manual num_divisions acts as a lower bound."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        builder = Trotter(num_divisions=10, target_accuracy=0.01)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        # Commutator bound gives 1, but manual gives 10 -> max(1, 10) = 10
        assert container.step_reps == 10

    def test_no_target_accuracy_backward_compatible(self):
        """Test that the builder is backward compatible when target_accuracy is disabled (0.0)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Trotter(num_divisions=3)
        unitary = builder.run(hamiltonian, time=0.5)
        container = unitary.get_container()
        assert container.step_reps == 3

    def test_angle_scaling_with_auto_steps(self):
        """Test that angles are correctly scaled when steps are auto-computed."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[2.0, 1.0])
        builder = Trotter(target_accuracy=0.01)
        t = 1.0
        unitary = builder.run(hamiltonian, time=t)
        container = unitary.get_container()
        n = container.step_reps
        dt = t / n
        # First term angle = coeff * dt = 2.0 * dt
        assert np.isclose(
            container.step_terms[0].angle,
            2.0 * dt,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        # Second term angle = coeff * dt = 1.0 * dt
        assert np.isclose(
            container.step_terms[1].angle,
            1.0 * dt,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_zero_target_accuracy_means_disabled(self):
        """Test that target_accuracy=0.0 (default) disables auto step computation."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Trotter(target_accuracy=0.0, num_divisions=3)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        # target_accuracy=0.0 means disabled, so num_divisions=3 is used directly
        assert container.step_reps == 3

    def test_invalid_error_bound_raises(self):
        """Test that an invalid error_bound raises an exception via Settings constraint."""
        with pytest.raises(ValueError, match="allowed options"):
            Trotter(error_bound="invalid")

    # Second-order Trotter tests.

    def test_target_accuracy_commutator_bound_second_order(self):
        """Test that target_accuracy with commutator bound computes correct step count."""
        # H = X + Z, X and Z anticommute -> commutator bound = 4
        # N = ceil(sqrt(4) * t^(3/2) / (sqrt(3! * eps))) = 17
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Trotter(target_accuracy=0.01, order=2)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        assert container.step_reps == 12

    def test_target_accuracy_naive_bound_second_order(self):
        """Test that target_accuracy with naive bound computes correct step count."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Trotter(target_accuracy=0.01, error_bound="naive", order=2)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        # one_norm = 2, N = ceil(2^1/2 / 0.01) = 29
        assert container.step_reps == 29

    def test_commutator_tighter_than_naive_second_order(self):
        """Test that commutator bound gives fewer steps than naive bound."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        eps = 0.01
        time = 1.0
        builder_comm = Trotter(target_accuracy=eps, error_bound="commutator", order=2)
        builder_naive = Trotter(target_accuracy=eps, error_bound="naive", order=2)
        n_comm = builder_comm.run(hamiltonian, time=time).get_container().step_reps
        n_naive = builder_naive.run(hamiltonian, time=time).get_container().step_reps
        assert n_comm <= n_naive

    def test_commuting_hamiltonian_needs_one_step_second_order(self):
        """Test that a fully commuting Hamiltonian needs only 1 Trotter step."""
        # XI and IX commute -> commutator bound = 0 -> N = 1
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        builder = Trotter(target_accuracy=0.01, order=2)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        assert container.step_reps == 1

    def test_manual_steps_as_lower_bound_second_order(self):
        """Test that manual num_divisions acts as a lower bound."""
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "IX"], coefficients=[1.0, 1.0])
        builder = Trotter(num_divisions=10, target_accuracy=0.01, order=2)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        # Commutator bound gives 1, but manual gives 10 -> max(1, 10) = 10
        assert container.step_reps == 10

    def test_no_target_accuracy_backward_compatible_second_order(self):
        """Test that the builder is backward compatible when target_accuracy is disabled (0.0)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Trotter(num_divisions=3, order=2)
        unitary = builder.run(hamiltonian, time=0.5)
        container = unitary.get_container()
        assert container.step_reps == 3

    def test_angle_scaling_with_auto_steps_second_order(self):
        """Test that angles are correctly scaled when steps are auto-computed."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[3.0, 1.0])
        builder = Trotter(target_accuracy=0.01, order=2)
        t = 1.0
        unitary = builder.run(hamiltonian, time=t)
        container = unitary.get_container()
        n = container.step_reps
        dt = t / n
        # First term angle = coeff * dt/2 = 1.5 * dt
        assert np.isclose(
            container.step_terms[0].angle,
            1.5 * dt,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )
        # Second term angle = coeff * dt = 1.0 * dt
        assert np.isclose(
            container.step_terms[1].angle,
            1.0 * dt,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

        # Third term angle = coeff * dt/2 = 1.5 * dt
        assert np.isclose(
            container.step_terms[2].angle,
            1.5 * dt,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_zero_target_accuracy_means_disabled_second_order(self):
        """Test that target_accuracy=0.0 (default) disables auto step computation."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Trotter(target_accuracy=0.0, num_divisions=3, order=2)
        unitary = builder.run(hamiltonian, time=1.0)
        container = unitary.get_container()
        # target_accuracy=0.0 means disabled, so num_divisions=3 is used directly
        assert container.step_reps == 3
