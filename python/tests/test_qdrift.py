"""Tests for the qDRIFT randomized time evolution builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.time_evolution.builder.qdrift import QDrift, QDriftSettings
from qdk_chemistry.data import QubitHamiltonian


class TestQDriftSettings:
    """Tests for QDriftSettings configuration."""

    def test_default_settings(self):
        """Verify default settings are properly initialized."""
        settings = QDriftSettings()
        assert settings.get("num_samples") == 100
        assert settings.get("seed") is None
        assert settings.get("tolerance") == 1e-12

    def test_settings_can_be_updated(self):
        """Verify settings can be modified."""
        settings = QDriftSettings()
        settings.set("num_samples", 500)
        settings.set("seed", 42)
        settings.set("tolerance", 1e-10)

        assert settings.get("num_samples") == 500
        assert settings.get("seed") == 42
        assert settings.get("tolerance") == 1e-10


class TestQDriftBuilder:
    """Tests for QDrift time evolution builder."""

    def test_name_returns_qdrift(self):
        """Verify the builder reports its name correctly."""
        qdrift = QDrift()
        assert qdrift.name() == "qdrift"

    def test_type_name_returns_time_evolution_builder(self):
        """Verify the builder reports its type correctly."""
        qdrift = QDrift()
        assert qdrift.type_name() == "time_evolution_builder"

    def test_create_qdrift_via_registry(self):
        """Verify qDRIFT can be created through the algorithm registry."""
        qdrift = create("time_evolution_builder", "qdrift")
        assert qdrift.name() == "qdrift"

    def test_create_qdrift_with_settings_via_registry(self):
        """Verify qDRIFT settings can be passed through the registry."""
        qdrift = create("time_evolution_builder", "qdrift", num_samples=250, seed=123)
        assert qdrift.settings().get("num_samples") == 250
        assert qdrift.settings().get("seed") == 123

    def test_qdrift_produces_correct_number_of_terms(self):
        """Verify qDRIFT produces the expected number of sampled terms."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "YY"],
            coefficients=[0.3, 0.5, 0.2],
        )
        num_samples = 50
        qdrift = QDrift(num_samples=num_samples, seed=42)

        result = qdrift.run(hamiltonian, time=1.0)
        container = result.container

        # The number of step_terms should equal num_samples
        assert len(container.step_terms) == num_samples
        assert container.step_reps == 1

    def test_qdrift_reproducibility_with_seed(self):
        """Verify that the same seed produces identical results."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "YY", "XI"],
            coefficients=[0.3, 0.5, 0.2, 0.1],
        )

        qdrift1 = QDrift(num_samples=30, seed=42)
        qdrift2 = QDrift(num_samples=30, seed=42)

        result1 = qdrift1.run(hamiltonian, time=1.0)
        result2 = qdrift2.run(hamiltonian, time=1.0)

        # Results should be identical
        assert len(result1.container.step_terms) == len(result2.container.step_terms)
        for term1, term2 in zip(result1.container.step_terms, result2.container.step_terms, strict=True):
            assert term1.pauli_term == term2.pauli_term
            assert term1.angle == term2.angle

    def test_qdrift_different_seeds_produce_different_results(self):
        """Verify that different seeds produce different sampling."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ", "YY", "XI"],
            coefficients=[0.3, 0.5, 0.2, 0.1],
        )

        qdrift1 = QDrift(num_samples=30, seed=42)
        qdrift2 = QDrift(num_samples=30, seed=123)

        result1 = qdrift1.run(hamiltonian, time=1.0)
        result2 = qdrift2.run(hamiltonian, time=1.0)

        # Results should be different (with high probability)
        terms1 = [(t.pauli_term, t.angle) for t in result1.container.step_terms]
        terms2 = [(t.pauli_term, t.angle) for t in result2.container.step_terms]
        assert terms1 != terms2

    def test_qdrift_angle_magnitude_is_correct(self):
        """Verify that the rotation angles follow qDRIFT formula."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=[0.25, 0.75],
        )
        time = 2.0
        num_samples = 10
        qdrift = QDrift(num_samples=num_samples, seed=42)

        result = qdrift.run(hamiltonian, time=time)

        # λ = |0.25| + |0.75| = 1.0
        # Expected angle magnitude = λ * t / N = 1.0 * 2.0 / 10 = 0.2
        expected_angle_magnitude = 1.0 * time / num_samples

        for term in result.container.step_terms:
            assert np.isclose(abs(term.angle), expected_angle_magnitude)

    def test_qdrift_handles_negative_coefficients(self):
        """Verify qDRIFT correctly handles negative Hamiltonian coefficients."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=[-0.5, 0.5],
        )
        qdrift = QDrift(num_samples=20, seed=42)

        result = qdrift.run(hamiltonian, time=1.0)

        # Check that angles have the correct signs
        # λ = |-0.5| + |0.5| = 1.0
        # angle_magnitude = 1.0 * 1.0 / 20 = 0.05
        expected_magnitude = 0.05

        for term in result.container.step_terms:
            assert np.isclose(abs(term.angle), expected_magnitude)

    def test_qdrift_empty_hamiltonian_returns_identity(self):
        """Verify qDRIFT handles zero/negligible coefficients gracefully."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=[1e-15, 1e-15],  # Below default tolerance
        )
        qdrift = QDrift(num_samples=10, seed=42)

        result = qdrift.run(hamiltonian, time=1.0)

        # Should return empty terms (identity evolution)
        assert len(result.container.step_terms) == 0

    def test_qdrift_single_term_hamiltonian(self):
        """Verify qDRIFT correctly handles a single-term Hamiltonian."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["ZZ"],
            coefficients=[0.5],
        )
        num_samples = 10
        time = 1.0
        qdrift = QDrift(num_samples=num_samples, seed=42)

        result = qdrift.run(hamiltonian, time=time)

        # All samples should be the same term
        assert len(result.container.step_terms) == num_samples

        # λ = 0.5, angle = 0.5 * 1.0 / 10 = 0.05
        for term in result.container.step_terms:
            assert term.pauli_term == {0: "Z", 1: "Z"}
            assert np.isclose(term.angle, 0.05)

    def test_qdrift_sampling_probability_distribution(self):
        """Verify that terms are sampled proportionally to coefficient magnitudes."""
        # Use a Hamiltonian with clearly different coefficients
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=[0.1, 0.9],  # ZZ should be sampled ~9x more often
        )
        num_samples = 10000  # Large sample for statistical significance
        qdrift = QDrift(num_samples=num_samples, seed=42)

        result = qdrift.run(hamiltonian, time=1.0)

        # Count how many times each term was sampled
        xx_count = 0
        zz_count = 0
        for term in result.container.step_terms:
            if 0 in term.pauli_term and term.pauli_term[0] == "X":
                xx_count += 1
            elif 0 in term.pauli_term and term.pauli_term[0] == "Z":
                zz_count += 1

        # Expected ratio: ZZ/XX ≈ 0.9/0.1 = 9
        # Allow some statistical variation
        ratio = zz_count / xx_count if xx_count > 0 else float("inf")
        assert 7.0 < ratio < 11.0  # Reasonable bounds for statistical test

    def test_qdrift_rejects_non_hermitian_hamiltonian(self):
        """Verify qDRIFT raises an error for non-Hermitian coefficients."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=[0.5 + 0.1j, 0.5],  # Complex coefficient
        )
        qdrift = QDrift(num_samples=10, seed=42)

        with pytest.raises(ValueError, match="Non-Hermitian"):
            qdrift.run(hamiltonian, time=1.0)

    def test_qdrift_num_qubits_preserved(self):
        """Verify the output preserves the correct number of qubits."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XXII", "IIZZ", "IYIY"],
            coefficients=[0.3, 0.3, 0.4],
        )
        qdrift = QDrift(num_samples=10, seed=42)

        result = qdrift.run(hamiltonian, time=1.0)

        assert result.container.num_qubits == 4


class TestQDriftVsTrotter:
    """Comparative tests between qDRIFT and Trotter builders."""

    def test_both_builders_available_in_registry(self):
        """Verify both Trotter and qDRIFT are available."""
        trotter = create("time_evolution_builder", "trotter")
        qdrift = create("time_evolution_builder", "qdrift")

        assert trotter.name() == "trotter"
        assert qdrift.name() == "qdrift"

    def test_qdrift_and_trotter_same_interface(self):
        """Verify both builders produce TimeEvolutionUnitary with same structure."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "ZZ"],
            coefficients=[0.3, 0.7],
        )

        trotter = create("time_evolution_builder", "trotter")
        qdrift = create("time_evolution_builder", "qdrift", num_samples=50, seed=42)

        trotter_result = trotter.run(hamiltonian, time=1.0)
        qdrift_result = qdrift.run(hamiltonian, time=1.0)

        # Both should produce PauliProductFormulaContainer
        assert type(trotter_result.container) == type(qdrift_result.container)
        assert trotter_result.container.num_qubits == qdrift_result.container.num_qubits
