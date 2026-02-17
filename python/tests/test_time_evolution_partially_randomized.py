"""Tests for Partially Randomized time evolution builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.time_evolution.builder.partially_randomized import PartiallyRandomized
from qdk_chemistry.data import QubitHamiltonian, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import PauliProductFormulaContainer

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestPartiallyRandomizedBasics:
    """Basic tests for the PartiallyRandomized class."""

    def test_name(self):
        """Test the name method of PartiallyRandomized."""
        builder = PartiallyRandomized()
        assert builder.name() == "partially_randomized"

    def test_type_name(self):
        """Test the type_name method of PartiallyRandomized."""
        builder = PartiallyRandomized()
        assert builder.type_name() == "time_evolution_builder"

    def test_can_create_via_registry(self):
        """Test that PartiallyRandomized can be created via the algorithm registry."""
        builder = create("time_evolution_builder", "partially_randomized")
        assert isinstance(builder, PartiallyRandomized)

    def test_can_create_with_settings(self):
        """Test that PartiallyRandomized can be created with custom settings."""
        builder = create(
            "time_evolution_builder",
            "partially_randomized",
            weight_threshold=0.5,
            num_random_samples=200,
            trotter_order=2,
            seed=42,
        )
        assert builder.settings().get("weight_threshold") == 0.5
        assert builder.settings().get("num_random_samples") == 200
        assert builder.settings().get("trotter_order") == 2
        assert builder.settings().get("seed") == 42


class TestPartiallyRandomizedConstruction:
    """Tests for PartiallyRandomized time evolution construction."""

    def test_returns_time_evolution_unitary(self):
        """Test that run returns a TimeEvolutionUnitary."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "YI", "ZI", "XX", "ZZ"],
            coefficients=[1.0, 0.5, 0.3, 0.1, 0.05],
        )
        builder = PartiallyRandomized(weight_threshold=0.4, num_random_samples=10, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        assert isinstance(unitary, TimeEvolutionUnitary)
        container = unitary.get_container()
        assert isinstance(container, PauliProductFormulaContainer)

    def test_second_order_structure(self):
        """Test that second-order Trotter has symmetric structure."""
        # Create Hamiltonian with clear weight separation
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z", "Y"],
            coefficients=[1.0, 0.1, 0.05],  # X is deterministic, Y and Z are random
        )
        builder = PartiallyRandomized(
            weight_threshold=0.5,
            num_random_samples=5,
            trotter_order=2,
            seed=42,
            merge_duplicate_terms=False,
        )
        unitary = builder.run(hamiltonian, time=0.2)
        terms = unitary.get_container().step_terms

        # Structure should be: X(half) -> random samples -> X(half)
        # First term should be X (deterministic, half angle)
        assert terms[0].pauli_term == {0: "X"}
        assert np.isclose(terms[0].angle, 0.1, atol=float_comparison_absolute_tolerance)

        # Last term should also be X (deterministic, half angle)
        assert terms[-1].pauli_term == {0: "X"}
        assert np.isclose(terms[-1].angle, 0.1, atol=float_comparison_absolute_tolerance)

        # Middle terms should be random samples
        assert len(terms) == 1 + 5 + 1  # det + random + det

    def test_first_order_structure(self):
        """Test that first-order Trotter has non-symmetric structure."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z", "Y"],
            coefficients=[1.0, 0.1, 0.05],
        )
        builder = PartiallyRandomized(
            weight_threshold=0.5,
            num_random_samples=5,
            trotter_order=1,
            seed=42,
            merge_duplicate_terms=False,
        )
        unitary = builder.run(hamiltonian, time=0.2)
        terms = unitary.get_container().step_terms

        # Structure should be: X(full) -> random samples
        # First term should be X (deterministic, full angle)
        assert terms[0].pauli_term == {0: "X"}
        assert np.isclose(terms[0].angle, 0.2, atol=float_comparison_absolute_tolerance)

        # Remaining terms should be random samples
        assert len(terms) == 1 + 5  # det + random

    def test_reproducible_with_seed(self):
        """Test that results are reproducible when using a seed."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "YI", "ZI", "XX", "ZZ"],
            coefficients=[1.0, 0.5, 0.3, 0.1, 0.05],
        )

        builder1 = PartiallyRandomized(weight_threshold=0.4, num_random_samples=20, seed=12345)
        builder2 = PartiallyRandomized(weight_threshold=0.4, num_random_samples=20, seed=12345)

        unitary1 = builder1.run(hamiltonian, time=0.1)
        unitary2 = builder2.run(hamiltonian, time=0.1)

        terms1 = unitary1.get_container().step_terms
        terms2 = unitary2.get_container().step_terms

        assert len(terms1) == len(terms2)
        for t1, t2 in zip(terms1, terms2, strict=True):
            assert t1.pauli_term == t2.pauli_term
            assert t1.angle == t2.angle


class TestPartiallyRandomizedSplitting:
    """Tests for the deterministic/random term splitting."""

    def test_split_by_weight_threshold_explicit(self):
        """Test splitting with explicit weight_threshold."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "YI", "ZI", "XX", "ZZ"],
            coefficients=[1.0, 0.5, 0.3, 0.1, 0.05],
        )
        # Threshold 0.4: XI (1.0) and YI (0.5) are deterministic
        builder = PartiallyRandomized(
            weight_threshold=0.4,
            num_random_samples=10,
            trotter_order=2,
            seed=42,
            merge_duplicate_terms=False,
        )
        unitary = builder.run(hamiltonian, time=0.1)
        terms = unitary.get_container().step_terms

        # 2nd order: 2 det forward + 10 random + 2 det backward = 14 total
        assert len(terms) == 14

        # First 2 terms should be XI and YI (largest coefficients)
        # Since it's second order, angles are half
        # XI maps to {1: "X"}, YI maps to {1: "Y"} (little-endian)
        first_det_paulis = [terms[0].pauli_term, terms[1].pauli_term]
        assert {1: "X"} in first_det_paulis
        assert {1: "Y"} in first_det_paulis

    def test_split_by_weight_threshold_first_order(self):
        """Test splitting with weight_threshold using first-order Trotter."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "YI", "ZI", "XX", "ZZ"],
            coefficients=[1.0, 0.5, 0.3, 0.1, 0.05],
        )
        # Threshold 0.4: XI (1.0) and YI (0.5) are deterministic
        builder = PartiallyRandomized(
            weight_threshold=0.4,
            num_random_samples=10,
            trotter_order=1,
            seed=42,
            merge_duplicate_terms=False,
        )
        unitary = builder.run(hamiltonian, time=0.1)
        terms = unitary.get_container().step_terms

        # 1st order: 2 deterministic + 10 random = 12 total
        assert len(terms) == 12

    def test_default_splitting(self):
        """Test default splitting (top 10% by weight)."""
        # Create Hamiltonian with 10 terms - all 2-qubit strings
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XI", "YI", "ZI", "XX", "YY", "ZZ", "XY", "XZ", "YZ", "IZ"],
            coefficients=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        )
        builder = PartiallyRandomized(
            num_random_samples=10,
            trotter_order=1,
            seed=42,
            merge_duplicate_terms=False,
        )
        unitary = builder.run(hamiltonian, time=0.1)
        terms = unitary.get_container().step_terms

        # 10 terms, 10% = 1 deterministic, 9 random
        # 1st order: 1 det + 10 random samples = 11 total
        # (default is max(1, 10//10) = 1 deterministic term)
        assert len(terms) == 11


class TestPartiallyRandomizedRandomPart:
    """Tests for the random sampling part of partially randomized."""

    def test_random_samples_use_lambda_r(self):
        """Test that random samples use λ_R (weight of random terms only)."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[0.8, 0.2],  # X deterministic, Z random
        )
        time = 0.5
        num_samples = 10

        builder = PartiallyRandomized(
            weight_threshold=0.5,
            num_random_samples=num_samples,
            trotter_order=1,
            seed=42,
            merge_duplicate_terms=False,
        )
        unitary = builder.run(hamiltonian, time=time)
        terms = unitary.get_container().step_terms

        # λ_R = 0.2 (only Z term)
        # angle_magnitude for random = λ_R * t / N = 0.2 * 0.5 / 10 = 0.01
        expected_random_angle = 0.2 * 0.5 / 10

        # Skip first term (deterministic X)
        for term in terms[1:]:
            assert np.isclose(
                abs(term.angle),
                expected_random_angle,
                atol=float_comparison_absolute_tolerance,
                rtol=float_comparison_relative_tolerance,
            )

    def test_all_terms_deterministic(self):
        """Test when all terms are treated deterministically."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[1.0, 0.5],
        )
        # Threshold low enough to capture all terms as deterministic
        builder = PartiallyRandomized(
            weight_threshold=0.1,
            num_random_samples=20,
            trotter_order=2,
            seed=42,
        )
        unitary = builder.run(hamiltonian, time=0.1)
        terms = unitary.get_container().step_terms

        # All 2 terms treated deterministically, no random samples
        # 2nd order: 2 forward + 2 backward = 4 total
        assert len(terms) == 4


class TestPartiallyRandomizedEdgeCases:
    """Edge case tests for PartiallyRandomized."""

    def test_empty_hamiltonian_after_filtering(self):
        """Test handling of Hamiltonian with only negligible terms."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1e-15])
        builder = PartiallyRandomized(weight_threshold=0.5, num_random_samples=10, seed=42)
        unitary = builder.run(hamiltonian, time=0.1)

        container = unitary.get_container()
        assert len(container.step_terms) == 0
        assert container.step_reps == 1

    def test_single_term_hamiltonian(self):
        """Test with a single-term Hamiltonian (all deterministic)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        builder = PartiallyRandomized(
            weight_threshold=0.5,
            num_random_samples=10,
            trotter_order=2,
            seed=42,
        )
        unitary = builder.run(hamiltonian, time=0.2)
        terms = unitary.get_container().step_terms

        # 2nd order with 1 deterministic term: X(half) + X(half) = 2 terms
        assert len(terms) == 2
        for term in terms:
            assert term.pauli_term == {0: "X"}
            assert np.isclose(term.angle, 0.1, atol=float_comparison_absolute_tolerance)

    def test_rejects_non_hermitian_hamiltonian(self):
        """Test that non-Hermitian Hamiltonians raise an error."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X"],
            coefficients=[1.0 + 0.5j],
        )
        builder = PartiallyRandomized(weight_threshold=0.5, seed=42)

        with pytest.raises(ValueError, match="Non-Hermitian"):
            builder.run(hamiltonian, time=0.1)

    def test_negative_coefficients(self):
        """Test that negative coefficients are handled correctly."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z", "Y"],
            coefficients=[-1.0, 0.1, -0.05],
        )
        builder = PartiallyRandomized(
            weight_threshold=0.5,
            num_random_samples=20,
            trotter_order=1,
            seed=42,
        )
        unitary = builder.run(hamiltonian, time=0.1)
        terms = unitary.get_container().step_terms

        # First term is X (deterministic) with negative angle
        assert terms[0].pauli_term == {0: "X"}
        assert terms[0].angle < 0

    def test_multi_qubit_hamiltonian(self):
        """Test with multi-qubit Pauli strings."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XII", "IZI", "IIY", "XXI", "ZZI", "IXX"],
            coefficients=[1.0, 0.5, 0.3, 0.1, 0.05, 0.01],
        )
        builder = PartiallyRandomized(
            weight_threshold=0.4,
            num_random_samples=20,
            trotter_order=2,
            seed=42,
        )
        unitary = builder.run(hamiltonian, time=0.1)

        container = unitary.get_container()
        assert container.num_qubits == 3


class TestPartiallyRandomizedPauliLabelToMap:
    """Tests for the _pauli_label_to_map helper function."""

    def test_identity_only(self):
        """Test that identity-only labels return an empty mapping."""
        builder = PartiallyRandomized()
        assert builder._pauli_label_to_map("III") == {}

    def test_single_pauli(self):
        """Test labels with a single non-identity Pauli."""
        builder = PartiallyRandomized()
        assert builder._pauli_label_to_map("X") == {0: "X"}
        assert builder._pauli_label_to_map("IZ") == {0: "Z"}

    def test_multiple_paulis(self):
        """Test labels with multiple non-identity Paulis."""
        builder = PartiallyRandomized()
        mapping = builder._pauli_label_to_map("XYZ")
        # Little-endian: rightmost char -> qubit 0
        assert mapping == {0: "Z", 1: "Y", 2: "X"}
