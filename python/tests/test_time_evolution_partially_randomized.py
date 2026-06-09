"""Tests for Partially Randomized time evolution builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest
from scipy.linalg import expm

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.partially_randomized import PartiallyRandomized
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.qdrift_error import qdrift_samples_campbell
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter_error import (
    trotter_steps_commutator,
    trotter_steps_naive,
)
from qdk_chemistry.data import LatticeGraph, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import (
    create_heisenberg_hamiltonian,
    create_ising_hamiltonian,
)
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def _container_to_unitary(container) -> np.ndarray:
    """Materialise a PauliProductFormulaContainer as a dense unitary matrix."""
    n = container.num_qubits
    unitary = np.eye(2**n, dtype=complex)
    for term in container.step_terms:
        s = ["I"] * n
        for q, op in term.pauli_term.items():
            s[n - 1 - q] = op  # little-endian
        pauli = pauli_to_dense_matrix(["".join(s)], np.array([1.0]))
        unitary = expm(-1j * term.angle * pauli) @ unitary
    return np.linalg.matrix_power(unitary, container.step_reps)


def _partial_state_trace_error(
    hamiltonian, *, eps, t, weight_threshold, seeds, accuracy_split=0.5, trotter_order=2, num_random_samples=1
):
    r"""Trace distance between the seed-averaged output state and the exact state.

    For a fixed input :math:`|0\dots0\rangle`, builds the partially randomized
    unitary :math:`U^{(k)} = U_D U_R^{(k)} U_D` for each seed, averages the
    resulting output density matrices, and compares to the exact evolution
    :math:`U = e^{-iHt}`.

    IMPORTANT: we average the *conjugated states* ``U_k rho U_k^dagger`` (the
    quantum-channel / diamond-norm quantity), NOT the bare unitaries
    ``E[<psi|U_k|psi>]``.  qDRIFT is a *biased* estimator of the time-evolution
    signal (it carries a ``(1 + tau^2)^(-r/2)`` damping factor), so an unbiased
    LCU-style check ``E[<psi|U_k|psi>] == <psi|e^{-iHt}|psi>`` would *correctly*
    fail.  Do not "fix" this into an expectation-of-unitary comparison.
    """
    h_matrix = hamiltonian.to_matrix()
    u_exact = expm(-1j * h_matrix * t)
    dim = u_exact.shape[0]
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    rho_exact = np.outer(u_exact @ psi, (u_exact @ psi).conj())

    rho_avg = np.zeros_like(rho_exact)
    for seed in seeds:
        builder = PartiallyRandomized(
            target_accuracy=eps,
            accuracy_split=accuracy_split,
            trotter_order=trotter_order,
            weight_threshold=weight_threshold,
            num_random_samples=num_random_samples,
            seed=seed,
            time=t,
            merge_duplicate_terms=False,
        )
        u_k = _container_to_unitary(builder.run(hamiltonian).get_container())
        v = u_k @ psi
        rho_avg = rho_avg + np.outer(v, v.conj())
    rho_avg = rho_avg / len(seeds)

    return 0.5 * float(np.linalg.svd(rho_avg - rho_exact, compute_uv=False).sum())


class TestPartiallyRandomizedBasics:
    """Basic tests for the PartiallyRandomized class."""

    def test_name(self):
        """Test the name method of PartiallyRandomized."""
        builder = PartiallyRandomized()
        assert builder.name() == "partially_randomized"

    def test_type_name(self):
        """Test the type_name method of PartiallyRandomized."""
        builder = PartiallyRandomized()
        assert builder.type_name() == "hamiltonian_unitary_builder"

    def test_can_create_via_registry(self):
        """Test that PartiallyRandomized can be created via the algorithm registry."""
        builder = create("hamiltonian_unitary_builder", "partially_randomized")
        assert isinstance(builder, PartiallyRandomized)

    def test_can_create_with_settings(self):
        """Test that PartiallyRandomized can be created with custom settings."""
        builder = create(
            "hamiltonian_unitary_builder",
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
            time=0.2,
        )
        unitary = builder.run(hamiltonian)
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
            time=0.2,
        )
        unitary = builder.run(hamiltonian)
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

        builder1 = PartiallyRandomized(weight_threshold=0.4, num_random_samples=20, seed=12345, time=0.1)
        builder2 = PartiallyRandomized(weight_threshold=0.4, num_random_samples=20, seed=12345, time=0.1)

        unitary1 = builder1.run(hamiltonian)
        unitary2 = builder2.run(hamiltonian)

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
            time=0.1,
        )
        unitary = builder.run(hamiltonian)
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
            time=0.1,
        )
        unitary = builder.run(hamiltonian)
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
            time=0.1,
        )
        unitary = builder.run(hamiltonian)
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
            time=time,
        )
        unitary = builder.run(hamiltonian)
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
            time=0.1,
        )
        unitary = builder.run(hamiltonian)
        terms = unitary.get_container().step_terms

        # All 2 terms treated deterministically, no random samples
        # 2nd order: 2 forward + 2 backward = 4 total
        assert len(terms) == 4


class TestPartiallyRandomizedEdgeCases:
    """Edge case tests for PartiallyRandomized."""

    def test_empty_hamiltonian_after_filtering(self):
        """Test handling of Hamiltonian with only negligible terms."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1e-15])
        builder = PartiallyRandomized(weight_threshold=0.5, num_random_samples=10, seed=42, time=0.1)
        unitary = builder.run(hamiltonian)

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
            time=0.2,
        )
        unitary = builder.run(hamiltonian)
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
        builder = PartiallyRandomized(weight_threshold=0.5, seed=42, time=0.1)

        with pytest.raises(ValueError, match="Non-Hermitian"):
            builder.run(hamiltonian)

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
            time=0.1,
        )
        unitary = builder.run(hamiltonian)
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
            time=0.1,
        )
        unitary = builder.run(hamiltonian)

        container = unitary.get_container()
        assert container.num_qubits == 3


class TestPartiallyRandomizedAccuracyAwareStructure:
    """Structural tests for ε-aware (target_accuracy) parameterization."""

    def test_epsilon_zero_preserves_single_step(self):
        """With target_accuracy=0 the builder uses a single sandwich (r=1)."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z", "Y"],
            coefficients=[1.0, 0.1, 0.05],
        )
        builder = PartiallyRandomized(
            weight_threshold=0.5,
            num_random_samples=5,
            trotter_order=2,
            seed=42,
            merge_duplicate_terms=False,
            time=0.2,
        )
        terms = builder.run(hamiltonian).get_container().step_terms
        # Single sandwich: 1 det forward + 5 random + 1 det backward
        assert len(terms) == 1 + 5 + 1

    def test_resolve_num_divisions_matches_commutator_bound(self):
        """The outer step count r equals the commutator Trotter bound for ε_D."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "YY", "ZZ", "XI", "IZ"],
            coefficients=[1.0, 0.8, 0.6, 0.4, 0.2],
        )
        time = 1.0
        eps = 0.01
        order = 2
        builder = PartiallyRandomized(
            target_accuracy=eps,
            accuracy_split=0.5,
            trotter_order=order,
            trotter_error_bound="commutator",
            weight_threshold=0.5,
            time=time,
        )
        eps_d = math.sqrt(0.5) * eps
        expected_r = trotter_steps_commutator(hamiltonian, time, eps_d, order=order, weight_threshold=1e-12)
        assert builder._resolve_num_divisions(hamiltonian, time) == expected_r

    def test_resolve_num_divisions_matches_naive_bound(self):
        """The outer step count r equals the naive Trotter bound when selected."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "YY", "ZZ", "XI", "IZ"],
            coefficients=[1.0, 0.8, 0.6, 0.4, 0.2],
        )
        time = 1.0
        eps = 0.01
        order = 2
        builder = PartiallyRandomized(
            target_accuracy=eps,
            accuracy_split=0.5,
            trotter_order=order,
            trotter_error_bound="naive",
            weight_threshold=0.5,
            time=time,
        )
        eps_d = math.sqrt(0.5) * eps
        expected_r = trotter_steps_naive(hamiltonian, time, eps_d, order=order, weight_threshold=1e-12)
        assert builder._resolve_num_divisions(hamiltonian, time) == expected_r

    def test_smaller_epsilon_increases_divisions(self):
        """Tightening ε increases (never decreases) the outer step count r."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "YY", "ZZ", "XI", "IZ"],
            coefficients=[1.0, 0.8, 0.6, 0.4, 0.2],
        )
        time = 1.0
        r_loose = PartiallyRandomized(
            target_accuracy=0.1, trotter_order=2, weight_threshold=0.5, time=time
        )._resolve_num_divisions(hamiltonian, time)
        r_tight = PartiallyRandomized(
            target_accuracy=0.001, trotter_order=2, weight_threshold=0.5, time=time
        )._resolve_num_divisions(hamiltonian, time)
        assert r_tight > r_loose

    def test_time_zero_single_division(self):
        """With ε set but time=0 the builder degenerates to a single step."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = PartiallyRandomized(target_accuracy=0.01, weight_threshold=0.6, time=0.0)
        assert builder._resolve_num_divisions(hamiltonian, 0.0) == 1

    def test_all_random_single_division(self):
        """With no deterministic terms there is no Trotter bias, so a single block is built."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        # weight_threshold huge -> all terms random
        builder = PartiallyRandomized(
            target_accuracy=0.01,
            weight_threshold=10.0,
            num_random_samples=3,
            trotter_order=2,
            seed=1,
            time=0.5,
            merge_duplicate_terms=False,
        )
        container = builder.run(hamiltonian).get_container()
        terms = container.step_terms
        # No deterministic terms -> a single qDRIFT block (r=1). With merge off,
        # the total term count equals one block's sample count; if r were > 1 we
        # would instead see r identical blocks.
        random_terms = hamiltonian.get_real_coefficients(tolerance=1e-12, sort_by_magnitude=True)
        n_block = builder._resolve_block_samples(random_terms, 0.5, 1)
        assert len(terms) == n_block

    def test_total_random_samples_match_r_times_block(self):
        """Total qDRIFT rotations equal r times the per-step block size (merge disabled)."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "YY", "ZZ", "XI", "IZ"],
            coefficients=[1.0, 0.8, 0.6, 0.4, 0.2],
        )
        time = 1.0
        eps = 0.05
        weight_threshold = 0.5  # XX, YY, ZZ deterministic; XI, IZ random
        builder = PartiallyRandomized(
            target_accuracy=eps,
            accuracy_split=0.5,
            trotter_order=2,
            weight_threshold=weight_threshold,
            num_random_samples=1,
            seed=7,
            time=time,
            merge_duplicate_terms=False,
        )
        container = builder.run(hamiltonian).get_container()
        terms = container.step_terms

        num_det = 3  # XX, YY, ZZ
        r = builder._resolve_num_divisions(hamiltonian, time)
        random_terms = [("XI", 0.4), ("IZ", 0.2)]
        n_block = builder._resolve_block_samples(random_terms, time, r)

        # Order-2: each step has 2*num_det deterministic + n_block random terms.
        expected_total = r * (2 * num_det + n_block)
        assert len(terms) == expected_total

    def test_block_samples_match_campbell_bound(self):
        """Per-step block size equals ceil(N_total / r) with N_total from Campbell."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "YY", "ZZ", "XI", "IZ"],
            coefficients=[1.0, 0.8, 0.6, 0.4, 0.2],
        )
        time = 1.0
        eps = 0.05
        builder = PartiallyRandomized(
            target_accuracy=eps,
            accuracy_split=0.5,
            trotter_order=2,
            weight_threshold=0.5,
            num_random_samples=1,
            time=time,
        )
        random_terms = [("XI", 0.4), ("IZ", 0.2)]
        r = builder._resolve_num_divisions(hamiltonian, time)
        h_random = QubitHamiltonian(pauli_strings=["XI", "IZ"], coefficients=np.array([0.4, 0.2]))
        eps_r = math.sqrt(0.5) * eps
        n_total = qdrift_samples_campbell(h_random, time, eps_r, weight_threshold=1e-12)
        expected_block = max(1, math.ceil(n_total / r))
        assert builder._resolve_block_samples(random_terms, time, r) == expected_block

    def test_num_random_samples_acts_as_floor(self):
        """A large num_random_samples floor wins over the Campbell-derived value."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "YY", "ZZ", "XI", "IZ"],
            coefficients=[1.0, 0.8, 0.6, 0.4, 0.2],
        )
        time = 1.0
        builder = PartiallyRandomized(
            target_accuracy=0.05,
            trotter_order=2,
            weight_threshold=0.5,
            num_random_samples=1000,
            time=time,
        )
        random_terms = [("XI", 0.4), ("IZ", 0.2)]
        r = builder._resolve_num_divisions(hamiltonian, time)
        assert builder._resolve_block_samples(random_terms, time, r) == 1000

    def test_accuracy_split_clamped(self):
        """accuracy_split is clamped to (0, 1) so both budgets stay positive."""
        builder_hi = PartiallyRandomized(target_accuracy=0.05, accuracy_split=5.0)
        eps_d, eps_r = builder_hi._split_accuracy()
        assert eps_d > 0.0
        assert eps_r > 0.0
        builder_lo = PartiallyRandomized(target_accuracy=0.05, accuracy_split=-3.0)
        eps_d2, eps_r2 = builder_lo._split_accuracy()
        assert eps_d2 > 0.0
        assert eps_r2 > 0.0

    def test_accuracy_split_quadrature(self):
        """ε_D² + ε_R² = ε² for the quadrature split."""
        eps = 0.05
        builder = PartiallyRandomized(target_accuracy=eps, accuracy_split=0.3)
        eps_d, eps_r = builder._split_accuracy()
        assert np.isclose(eps_d**2 + eps_r**2, eps**2, atol=1e-15)

    def test_larger_split_reduces_divisions(self):
        """A larger accuracy_split (looser ε_D) does not increase r."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "YY", "ZZ", "XI", "IZ"],
            coefficients=[1.0, 0.8, 0.6, 0.4, 0.2],
        )
        time = 1.0
        eps = 0.005
        r_small_split = PartiallyRandomized(
            target_accuracy=eps, accuracy_split=0.1, trotter_order=2, weight_threshold=0.5, time=time
        )._resolve_num_divisions(hamiltonian, time)
        r_large_split = PartiallyRandomized(
            target_accuracy=eps, accuracy_split=0.9, trotter_order=2, weight_threshold=0.5, time=time
        )._resolve_num_divisions(hamiltonian, time)
        assert r_large_split <= r_small_split

    def test_reproducible_with_seed_accuracy_aware(self):
        """Two ε-aware builders with the same seed produce identical circuits."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z", "Y"],
            coefficients=[1.0, 0.5, 0.4],
        )
        kwargs = {
            "target_accuracy": 0.05,
            "trotter_order": 2,
            "weight_threshold": 0.6,
            "time": 0.5,
            "merge_duplicate_terms": False,
        }
        terms1 = PartiallyRandomized(seed=2024, **kwargs).run(hamiltonian).get_container().step_terms
        terms2 = PartiallyRandomized(seed=2024, **kwargs).run(hamiltonian).get_container().step_terms
        assert len(terms1) == len(terms2)
        for t1, t2 in zip(terms1, terms2, strict=True):
            assert t1.pauli_term == t2.pauli_term
            assert t1.angle == t2.angle

    def test_cost_optimal_split_selects_dominant_terms(self):
        """Cost-optimal L_D assigns the dominant terms to H_D and the tail to H_R."""
        # Three dominant terms + a long tail of tiny terms.
        pauli_strings = ["XX", "YY", "ZZ"] + ["XI", "IX", "IY", "YI", "ZI"] * 4
        coefficients = [10.0, 9.0, 8.0] + [0.001] * 20
        hamiltonian = QubitHamiltonian(pauli_strings=pauli_strings, coefficients=np.array(coefficients))
        time = 1.0
        builder = PartiallyRandomized(
            target_accuracy=0.01,
            accuracy_split=0.5,
            trotter_order=2,
            weight_threshold=-1.0,  # automatic -> cost-optimal
            time=time,
        )
        terms = hamiltonian.get_real_coefficients(tolerance=1e-12, sort_by_magnitude=True)
        num_det = builder._determine_num_deterministic_cost_optimal(hamiltonian, terms, time)
        # The three large terms belong in H_D; the tiny tail should be randomized.
        assert num_det == 3

    def test_explicit_threshold_overrides_cost_optimal(self):
        """An explicit weight_threshold takes precedence over cost-optimal split."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["XX", "YY", "ZZ", "XI", "IZ"],
            coefficients=[1.0, 0.8, 0.6, 0.4, 0.2],
        )
        time = 1.0
        builder = PartiallyRandomized(
            target_accuracy=0.01,
            trotter_order=2,
            weight_threshold=0.5,  # explicit -> count |c| >= 0.5
            time=time,
        )
        terms = hamiltonian.get_real_coefficients(tolerance=1e-12, sort_by_magnitude=True)
        num_det = builder._determine_num_deterministic(hamiltonian, terms, time)
        assert num_det == 3  # XX, YY, ZZ


class TestPartiallyRandomizedOutputAccuracy:
    """Output-unitary accuracy tests for the ε-aware builder.

    The qDRIFT block is random, so a single built unitary cannot be compared
    directly to ``exp(-iHt)`` — its spectral error is ``O(λ_R t / sqrt(N))`` and
    routinely exceeds ε.  These tests therefore use two complementary tiers:

    * Deterministic limit (λ_R = 0): the builder reduces to a pure Trotter
      product, which is deterministic and can be compared straight to the exact
      unitary in operator norm.
    * Partial case: the seed-averaged output state is compared to the exact
      state (the channel-level guarantee), mirroring the qDRIFT accuracy tests.
    """

    # Combined Trotter bias (≤ ε_D) and averaged qDRIFT error (≤ ε_R) add up to
    # at most ε_D + ε_R = sqrt(2)·ε; allow extra headroom for Monte-Carlo noise.
    TOLERANCE_FACTOR = 3

    def test_deterministic_limit_within_epsilon(self):
        """All-deterministic split (λ_R=0): exact Trotter stays within ε."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.7])
        time = 0.8
        eps = 0.05
        builder = PartiallyRandomized(
            target_accuracy=eps,
            trotter_order=2,
            weight_threshold=0.0,  # all terms deterministic -> λ_R = 0
            num_random_samples=1,
            seed=0,
            time=time,
            merge_duplicate_terms=False,
        )
        u_built = _container_to_unitary(builder.run(hamiltonian).get_container())
        u_exact = expm(-1j * hamiltonian.to_matrix() * time)
        err = float(np.linalg.norm(u_built - u_exact, 2))
        assert err <= eps

    def test_deterministic_limit_error_decreases(self):
        """Tightening ε reduces the deterministic-limit operator-norm error."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.7])
        time = 0.8

        def err_at(eps: float) -> float:
            builder = PartiallyRandomized(
                target_accuracy=eps,
                trotter_order=2,
                weight_threshold=0.0,
                num_random_samples=1,
                seed=0,
                time=time,
                merge_duplicate_terms=False,
            )
            u_built = _container_to_unitary(builder.run(hamiltonian).get_container())
            u_exact = expm(-1j * hamiltonian.to_matrix() * time)
            return float(np.linalg.norm(u_built - u_exact, 2))

        assert err_at(0.01) < err_at(0.1)

    def test_partial_state_trace_within_tolerance(self):
        """Seed-averaged output state stays within a small multiple of ε."""
        # X deterministic; Z and Y random (non-commuting -> genuine qDRIFT noise).
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z", "Y"], coefficients=[1.0, 0.5, 0.4])
        eps = 0.05
        time = 0.5
        err = _partial_state_trace_error(hamiltonian, eps=eps, t=time, weight_threshold=0.6, seeds=range(400))
        assert err <= self.TOLERANCE_FACTOR * eps

    def test_partial_error_scales_with_epsilon(self):
        """Tightening ε reduces the seed-averaged output-state error."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z", "Y"], coefficients=[1.0, 0.5, 0.4])
        time = 0.5
        seeds = range(400)
        err_loose = _partial_state_trace_error(hamiltonian, eps=0.1, t=time, weight_threshold=0.6, seeds=seeds)
        err_tight = _partial_state_trace_error(hamiltonian, eps=0.02, t=time, weight_threshold=0.6, seeds=seeds)
        assert err_tight < err_loose

    def test_partial_accuracy_multi_step(self):
        """Multi-step (r > 1) partial evolution stays within tolerance.

        This is the key accuracy test for the outer Trotter loop: it forces
        several independent sandwiches, each with its own freshly sampled qDRIFT
        block, and checks the seed-averaged output state against the exact
        evolution.  The ``r >= 3`` assertion is self-validating — if a change
        ever collapses the loop back to a single step, this test fails loudly
        rather than silently degrading to single-sandwich coverage.
        """
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z", "Y"], coefficients=[1.0, 0.5, 0.4])
        eps = 0.01
        time = 1.0
        weight_threshold = 0.6  # X deterministic; Z, Y randomized

        probe = PartiallyRandomized(target_accuracy=eps, trotter_order=2, weight_threshold=weight_threshold, time=time)
        assert probe._resolve_num_divisions(hamiltonian, time) >= 3

        err = _partial_state_trace_error(
            hamiltonian, eps=eps, t=time, weight_threshold=weight_threshold, seeds=range(400)
        )
        assert err <= self.TOLERANCE_FACTOR * eps

    def test_partial_two_qubit_accuracy(self):
        """Two-qubit partial evolution with a multi-term deterministic part.

        A single-qubit / single-deterministic-term test cannot exercise the
        order-2 sandwich reversal (``reversed(deterministic_terms)``) or the
        little-endian multi-term ordering.  Here ``H_D = {XI, IZ}`` and
        ``H_R = {XX, YY}`` on two qubits, and the parameters also yield
        ``r > 1`` so the multi-step loop is exercised in the multi-qubit case.
        """
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "IZ", "XX", "YY"], coefficients=[1.0, 0.8, 0.2, 0.15])
        eps = 0.02
        time = 0.8
        weight_threshold = 0.5  # H_D = {XI, IZ}, H_R = {XX, YY}

        probe = PartiallyRandomized(target_accuracy=eps, trotter_order=2, weight_threshold=weight_threshold, time=time)
        assert probe._resolve_num_divisions(hamiltonian, time) >= 2

        err = _partial_state_trace_error(
            hamiltonian, eps=eps, t=time, weight_threshold=weight_threshold, seeds=range(400)
        )
        assert err <= self.TOLERANCE_FACTOR * eps

    def test_partial_state_trace_strict(self):
        """Strict tier: seed-averaged output state stays within ε (no slack).

        Campbell (2019) bounds the diamond norm of the channel difference by ε,
        and the state trace distance is upper-bounded by the diamond norm, so
        ``err <= ε`` is achievable (unlike the looser ``3·ε`` channel tests
        elsewhere, which only leave Monte-Carlo headroom).  Mirrors the strict
        qDRIFT accuracy tier; uses more seeds to suppress sampling noise.
        """
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z", "Y"], coefficients=[1.0, 0.5, 0.4])
        eps = 0.05
        time = 0.5
        err = _partial_state_trace_error(hamiltonian, eps=eps, t=time, weight_threshold=0.6, seeds=range(500))
        assert err <= eps

    def test_partial_first_order_accuracy(self):
        """First-order partial evolution stays within tolerance.

        Exercises the order-1 construction path (deterministic terms at full
        angle followed by the qDRIFT block) against the exact unitary, which the
        order-2 accuracy tests do not cover.
        """
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z", "Y"], coefficients=[1.0, 0.5, 0.4])
        eps = 0.05
        time = 0.5
        err = _partial_state_trace_error(
            hamiltonian, eps=eps, t=time, weight_threshold=0.6, seeds=range(400), trotter_order=1
        )
        assert err <= self.TOLERANCE_FACTOR * eps


class TestPartiallyRandomizedModelHamiltonians:
    """Accuracy checks on real model Hamiltonians from the codebase.

    Unlike the hand-crafted Pauli strings used elsewhere, these tests build
    physical lattice models via :mod:`qdk_chemistry.utils.model_hamiltonians`
    (transverse-field Ising and Heisenberg chains) and validate the
    seed-averaged output state against the exact evolution.  Both models have a
    natural weight separation (large two-body couplings vs. smaller fields),
    which the partially randomized split exploits, and both resolve to a
    multi-step (``r > 1``) outer loop at the chosen accuracies.
    """

    TOLERANCE_FACTOR = 3

    def test_transverse_field_ising_chain(self):
        """TFIM 4-site chain: deterministic ZZ couplings, randomized X fields."""
        lattice = LatticeGraph.chain(4)
        # J = 1.0 (ZZ couplings) deterministic; h = 0.5 (X fields) randomized.
        hamiltonian = create_ising_hamiltonian(lattice, j=1.0, h=0.5)
        eps = 0.05
        time = 0.5
        weight_threshold = 0.75  # H_D = {ZZ couplings}, H_R = {X fields}

        probe = PartiallyRandomized(target_accuracy=eps, trotter_order=2, weight_threshold=weight_threshold, time=time)
        assert probe._resolve_num_divisions(hamiltonian, time) >= 2

        err = _partial_state_trace_error(
            hamiltonian, eps=eps, t=time, weight_threshold=weight_threshold, seeds=range(200)
        )
        assert err <= self.TOLERANCE_FACTOR * eps

    def test_heisenberg_chain(self):
        """Heisenberg 4-site chain: deterministic XYZ couplings, randomized Z fields."""
        lattice = LatticeGraph.chain(4)
        # Uniform J = 1.0 (XX+YY+ZZ) deterministic; hz = 0.3 (Z fields) randomized.
        hamiltonian = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0, hz=0.3)
        eps = 0.05
        time = 0.3
        weight_threshold = 0.5  # H_D = {couplings}, H_R = {Z fields}

        probe = PartiallyRandomized(target_accuracy=eps, trotter_order=2, weight_threshold=weight_threshold, time=time)
        assert probe._resolve_num_divisions(hamiltonian, time) >= 2

        err = _partial_state_trace_error(
            hamiltonian, eps=eps, t=time, weight_threshold=weight_threshold, seeds=range(200)
        )
        assert err <= self.TOLERANCE_FACTOR * eps

    def test_ising_deterministic_limit(self):
        """All-deterministic TFIM chain reduces to exact Trotter within eps."""
        lattice = LatticeGraph.chain(4)
        hamiltonian = create_ising_hamiltonian(lattice, j=1.0, h=0.5)
        eps = 0.02
        time = 0.3
        builder = PartiallyRandomized(
            target_accuracy=eps,
            trotter_order=2,
            weight_threshold=0.0,  # all terms deterministic -> lambda_R = 0
            num_random_samples=1,
            seed=0,
            time=time,
            merge_duplicate_terms=False,
        )
        u_built = _container_to_unitary(builder.run(hamiltonian).get_container())
        u_exact = expm(-1j * hamiltonian.to_matrix() * time)
        err = float(np.linalg.norm(u_built - u_exact, 2))
        assert err <= eps
