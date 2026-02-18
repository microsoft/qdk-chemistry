"""Tests for qDRIFT randomized time evolution builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.time_evolution.builder.pauli_commutation import do_pauli_terms_qw_commute
from qdk_chemistry.algorithms.time_evolution.builder.qdrift import QDrift, QDriftSettings
from qdk_chemistry.data import QubitHamiltonian, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestQDriftSettings:
    """Tests for QDriftSettings configuration."""

    def test_default_settings(self):
        """Verify default settings are properly initialized."""
        settings = QDriftSettings()
        assert settings.get("num_samples") == 100
        assert settings.get("seed") == -1
        assert settings.get("merge_duplicate_terms") is True

    def test_settings_can_be_updated(self):
        """Verify settings can be modified."""
        settings = QDriftSettings()
        settings.set("num_samples", 500)
        settings.set("seed", 42)
        settings.set("merge_duplicate_terms", False)

        assert settings.get("num_samples") == 500
        assert settings.get("seed") == 42
        assert settings.get("merge_duplicate_terms") is False


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
        """Test that the container has at most num_samples terms (fewer after merging)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        num_samples = 50
        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        container = unitary.get_container()
        # Duplicate-term merging may reduce the count below num_samples
        assert len(container.step_terms) <= num_samples
        assert len(container.step_terms) >= 1

    def test_exact_sample_count_without_merging(self):
        """Disabling merge_duplicate_terms gives exactly num_samples terms."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        num_samples = 50
        builder = QDrift(num_samples=num_samples, seed=42, merge_duplicate_terms=False)
        unitary = builder.run(hamiltonian, time=0.1)

        assert len(unitary.get_container().step_terms) == num_samples

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

        num_samples = 10000
        time = 1.0
        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=time)

        terms = unitary.get_container().step_terms

        # After merging, extract the total angle assigned to each operator.
        # The total angle is proportional to the number of times that term
        # was sampled, so the ratio of angles reflects the sampling ratio.
        angle_by_op: dict[str, float] = {}
        for t in terms:
            key = str(sorted(t.pauli_term.items()))
            angle_by_op[key] = angle_by_op.get(key, 0.0) + abs(t.angle)

        x_key = str(sorted({0: "X"}.items()))
        z_key = str(sorted({0: "Z"}.items()))

        total_angle = angle_by_op.get(x_key, 0.0) + angle_by_op.get(z_key, 0.0)
        x_fraction = angle_by_op.get(x_key, 0.0) / total_angle

        # X should receive ~90% of the total angle
        assert x_fraction > 0.85
        assert x_fraction < 0.95

    def test_angle_calculation(self):
        """Test that the total rotation angle equals λ·t after merging."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[0.6, 0.4])
        time = 0.5
        num_samples = 10

        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=time)

        # λ = |0.6| + |0.4| = 1.0
        # Sum of |merged_angle| across all terms should equal λ * t = 0.5,
        # since each of the N samples contributes λ*t/N to exactly one term.
        total_angle = sum(abs(t.angle) for t in unitary.get_container().step_terms)
        assert np.isclose(
            total_angle,
            1.0 * time,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )


class TestQDriftEdgeCases:
    """Edge case tests for QDrift."""

    def test_tiny_coefficients_still_sampled(self):
        """Test that even very small coefficients are included (no filtering)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1e-10])
        builder = QDrift(num_samples=10, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        container = unitary.get_container()
        # Single term: all 10 identical samples merge into one rotation
        assert len(container.step_terms) == 1
        assert container.step_terms[0].pauli_term == {0: "X"}

    def test_single_term_hamiltonian(self):
        """Test QDrift with a single-term Hamiltonian."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        num_samples = 20
        time = 0.1
        builder = QDrift(num_samples=num_samples, seed=42)
        unitary = builder.run(hamiltonian, time=time)

        terms = unitary.get_container().step_terms
        # All 20 identical samples merge into a single X rotation
        assert len(terms) == 1
        assert terms[0].pauli_term == {0: "X"}

        # Merged angle = num_samples * (λ * t / N) = λ * t = 1.0 * 0.1
        assert np.isclose(
            terms[0].angle,
            1.0 * time,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

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
        # Total |angle| across all merged terms should equal λ * t = 0.15
        total_angle = sum(abs(t.angle) for t in terms)
        assert np.isclose(
            total_angle,
            1.5 * time,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

        # X terms should have negative angles, Z terms positive
        for term in terms:
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
        assert len(container.step_terms) <= 50
        assert len(container.step_terms) >= 1

    def test_num_qubits_preserved_four_qubits(self):
        """Verify the output preserves the correct number of qubits for larger systems."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XXII", "IIZZ", "IYIY"],
            coefficients=[0.3, 0.3, 0.4],
        )
        builder = QDrift(num_samples=10, seed=42)
        unitary = builder.run(hamiltonian, time=1.0)

        assert unitary.get_container().num_qubits == 4


class TestQDriftDuplicateTermFusion:
    """Tests for the duplicate-term fusion optimisation."""

    def test_identical_terms_merged(self):
        """Consecutive identical Pauli terms are fused into one."""
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
        ]
        merged = QDrift._merge_duplicate_terms(terms)
        assert len(merged) == 1
        assert merged[0].pauli_term == {0: "X"}
        assert np.isclose(merged[0].angle, 0.3)

    def test_non_commuting_boundary_preserved(self):
        """Non-commuting terms are not merged across boundaries."""
        # X and Y on the same qubit anti-commute
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
            ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=0.2),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
        ]
        merged = QDrift._merge_duplicate_terms(terms)
        # Cannot merge across the Y boundary → 3 separate terms
        assert len(merged) == 3

    def test_commuting_different_terms_kept_separate(self):
        """Commuting but distinct terms within a run are kept (not fused)."""
        # XI and IZ act on different qubits → commute
        terms = [
            ExponentiatedPauliTerm(pauli_term={1: "X"}, angle=0.1),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.2),
        ]
        merged = QDrift._merge_duplicate_terms(terms)
        assert len(merged) == 2

    def test_commuting_duplicates_fused(self):
        """Duplicate terms within a commuting run are fused."""
        # XI, IZ, XI — all commute, two XI terms merge
        terms = [
            ExponentiatedPauliTerm(pauli_term={1: "X"}, angle=0.1),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.2),
            ExponentiatedPauliTerm(pauli_term={1: "X"}, angle=0.1),
        ]
        merged = QDrift._merge_duplicate_terms(terms)
        assert len(merged) == 2
        angles = {str(sorted(t.pauli_term.items())): t.angle for t in merged}
        assert np.isclose(angles[str(sorted({1: "X"}.items()))], 0.2)
        assert np.isclose(angles[str(sorted({0: "Z"}.items()))], 0.2)

    def test_cancelling_angles_dropped(self):
        """Terms whose fused angle is zero are dropped."""
        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=-0.5),
        ]
        merged = QDrift._merge_duplicate_terms(terms)
        assert len(merged) == 0

    def test_empty_input(self):
        """Empty term list returns empty."""
        assert QDrift._merge_duplicate_terms([]) == []

    def test_pauli_terms_qw_commute(self):
        """Verify qubit-wise commutation checks for known cases."""
        # Same qubit, same Pauli → qw-commute
        assert do_pauli_terms_qw_commute({0: "X"}, {0: "X"}) is True
        # Same qubit, different Pauli → do not qw-commute
        assert do_pauli_terms_qw_commute({0: "X"}, {0: "Y"}) is False
        # Different qubits → qw-commute
        assert do_pauli_terms_qw_commute({0: "X"}, {1: "Y"}) is True
        # XY vs YX: commute globally but do NOT qubit-wise commute
        assert do_pauli_terms_qw_commute({0: "X", 1: "Y"}, {0: "Y", 1: "X"}) is False
        # Same Paulis on overlapping qubits → qw-commute
        assert do_pauli_terms_qw_commute({0: "X", 1: "Z"}, {0: "X", 1: "Z"}) is True
        # One differing, one matching → do not qw-commute
        assert do_pauli_terms_qw_commute({0: "X", 1: "Z"}, {0: "Y", 1: "Z"}) is False


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
