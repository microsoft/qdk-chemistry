"""Tests for qDRIFT randomized time evolution builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.time_evolution.builder.qdrift import QDrift, QDriftSettings
from qdk_chemistry.data import QubitHamiltonian, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import PauliProductFormulaContainer

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestQDriftSettings:
    """Tests for QDriftSettings configuration."""

    def test_default_settings(self):
        """Verify default settings are properly initialized."""
        settings = QDriftSettings()
        assert settings.get("num_samples") == 100
        assert settings.get("seed") == -1
        assert settings.get("tolerance") == pytest.approx(1e-12)

    def test_settings_can_be_updated(self):
        """Verify settings can be modified."""
        settings = QDriftSettings()
        settings.set("num_samples", 500)
        settings.set("seed", 42)
        settings.set("tolerance", 1e-10)

        assert settings.get("num_samples") == 500
        assert settings.get("seed") == 42
        assert settings.get("tolerance") == 1e-10


class TestQDriftBasics:
    """Basic tests for the QDrift class."""

    def test_name(self):
        """Test the name method of QDrift."""
        builder = QDrift()
        assert builder.name() == "qdrift"

    def test_type_name(self):
        """Test the type_name method of QDrift."""
        builder = QDrift()
        assert builder.type_name() == "time_evolution_builder"

    def test_can_create_via_registry(self):
        """Test that QDrift can be created via the algorithm registry."""
        builder = create("time_evolution_builder", "qdrift")
        assert isinstance(builder, QDrift)

    def test_can_create_with_settings(self):
        """Test that QDrift can be created with custom settings."""
        builder = create("time_evolution_builder", "qdrift", num_samples=200, seed=42)
        assert builder.settings().get("num_samples") == 200
        assert builder.settings().get("seed") == 42


class TestQDriftConstruction:
    """Tests for QDrift time evolution construction."""

    def test_returns_time_evolution_unitary(self):
        """Test that run returns a TimeEvolutionUnitary."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = QDrift(num_samples=10, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        assert isinstance(unitary, TimeEvolutionUnitary)
        container = unitary.get_container()
        assert isinstance(container, PauliProductFormulaContainer)

    def test_correct_number_of_samples(self):
        """Test that the container has the correct number of sampled terms."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        num_samples = 50
        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        container = unitary.get_container()
        assert len(container.step_terms) == num_samples

    def test_reproducible_with_seed(self):
        """Test that results are reproducible when using a seed."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Y", "Z"], coefficients=[1.0, 0.5, 0.25])

        builder1 = QDrift(num_samples=20, seed=12345)
        builder2 = QDrift(num_samples=20, seed=12345)

        unitary1 = builder1.run(hamiltonian, time=0.1)
        unitary2 = builder2.run(hamiltonian, time=0.1)

        terms1 = unitary1.get_container().step_terms
        terms2 = unitary2.get_container().step_terms

        assert len(terms1) == len(terms2)
        for t1, t2 in zip(terms1, terms2, strict=True):
            assert t1.pauli_term == t2.pauli_term
            assert t1.angle == t2.angle

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different sampling."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "YI", "ZI", "XX", "ZZ"],
            coefficients=[1.0, 0.5, 0.25, 0.1, 0.05],
        )

        builder1 = QDrift(num_samples=30, seed=42)
        builder2 = QDrift(num_samples=30, seed=123)

        unitary1 = builder1.run(hamiltonian, time=0.1)
        unitary2 = builder2.run(hamiltonian, time=0.1)

        terms1 = [(t.pauli_term, t.angle) for t in unitary1.get_container().step_terms]
        terms2 = [(t.pauli_term, t.angle) for t in unitary2.get_container().step_terms]

        # Results should be different with high probability
        assert terms1 != terms2


class TestQDriftSampling:
    """Tests for qDRIFT sampling behavior."""

    def test_samples_proportional_to_weights(self):
        """Test that terms are sampled approximately proportional to |h_j|."""
        # Create a Hamiltonian with unequal weights
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[0.9, 0.1],  # X should be sampled ~90% of the time
        )

        num_samples = 1000
        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        terms = unitary.get_container().step_terms

        # Count X vs Z samples
        x_count = sum(1 for t in terms if t.pauli_term == {0: "X"})
        z_count = sum(1 for t in terms if t.pauli_term == {0: "Z"})

        # X should be ~90%, Z should be ~10%
        assert x_count + z_count == num_samples
        assert x_count > 0.8 * num_samples  # Allow some variance
        assert z_count > 0.02 * num_samples

    def test_angle_calculation(self):
        """Test that rotation angles are calculated correctly."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[0.6, 0.4])
        time = 0.5
        num_samples = 10

        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=time)

        # λ = |0.6| + |0.4| = 1.0
        # angle_magnitude = λ * t / N = 1.0 * 0.5 / 10 = 0.05
        # All terms should have angle = ±0.05
        expected_magnitude = 0.05

        for term in unitary.get_container().step_terms:
            assert np.isclose(
                abs(term.angle),
                expected_magnitude,
                atol=float_comparison_absolute_tolerance,
                rtol=float_comparison_relative_tolerance,
            )


class TestQDriftEdgeCases:
    """Edge case tests for QDrift."""

    def test_empty_hamiltonian_after_filtering(self):
        """Test handling of Hamiltonian with only negligible terms."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1e-15])
        builder = QDrift(num_samples=10, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        container = unitary.get_container()
        assert len(container.step_terms) == 0
        assert container.step_reps == 1

    def test_single_term_hamiltonian(self):
        """Test QDrift with a single-term Hamiltonian."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        num_samples = 20
        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        terms = unitary.get_container().step_terms
        assert len(terms) == num_samples

        # All terms should be X rotations
        for term in terms:
            assert term.pauli_term == {0: "X"}

    def test_rejects_non_hermitian_hamiltonian(self):
        """Test that non-Hermitian Hamiltonians raise an error."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X"],
            coefficients=[1.0 + 0.5j],
        )
        builder = QDrift(num_samples=10, seed=42)

        with pytest.raises(ValueError, match="Non-Hermitian"):
            builder.run(hamiltonian, time=0.1)

    def test_negative_coefficients(self):
        """Test that negative coefficients are handled correctly."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[-1.0, 0.5],
        )
        time = 0.1
        num_samples = 20
        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=time)

        terms = unitary.get_container().step_terms

        # λ = |-1.0| + |0.5| = 1.5
        # angle magnitude = λ * t / N = 1.5 * 0.1 / 20 = 0.0075
        expected_magnitude = 1.5 * time / num_samples

        # Check that X terms have negative angles, Z terms have positive,
        # and all angles have the correct magnitude.
        for term in terms:
            assert np.isclose(
                abs(term.angle),
                expected_magnitude,
                atol=float_comparison_absolute_tolerance,
                rtol=float_comparison_relative_tolerance,
            )
            if term.pauli_term == {0: "X"}:
                assert term.angle < 0
            elif term.pauli_term == {0: "Z"}:
                assert term.angle > 0

    def test_multi_qubit_hamiltonian(self):
        """Test QDrift with multi-qubit Pauli strings."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "IZ", "XX", "ZZ"],
            coefficients=[1.0, 0.5, 0.3, 0.2],
        )
        builder = QDrift(num_samples=50, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        container = unitary.get_container()
        assert container.num_qubits == 2
        assert len(container.step_terms) == 50

    def test_num_qubits_preserved_four_qubits(self):
        """Verify the output preserves the correct number of qubits for larger systems."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XXII", "IIZZ", "IYIY"],
            coefficients=[0.3, 0.3, 0.4],
        )
        builder = QDrift(num_samples=10, seed=42)
        unitary = builder.run(hamiltonian, time=1.0)

        assert unitary.get_container().num_qubits == 4


class TestQDriftPauliLabelToMap:
    """Tests for the _pauli_label_to_map helper function."""

    def test_identity_only(self):
        """Test that identity-only labels return an empty mapping."""
        builder = QDrift()
        assert builder._pauli_label_to_map("III") == {}

    def test_single_pauli(self):
        """Test labels with a single non-identity Pauli."""
        builder = QDrift()
        assert builder._pauli_label_to_map("X") == {0: "X"}
        assert builder._pauli_label_to_map("IZ") == {0: "Z"}

    def test_multiple_paulis(self):
        """Test labels with multiple non-identity Paulis."""
        builder = QDrift()
        mapping = builder._pauli_label_to_map("XYZ")
        # Little-endian: rightmost char -> qubit 0
        assert mapping == {0: "Z", 1: "Y", 2: "X"}
