"""Tests for qDRIFT randomized time evolution builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import ClassVar

import numpy as np
import pytest
from scipy.linalg import expm

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.qdrift import QDrift, QDriftSettings
from qdk_chemistry.data import QubitHamiltonian, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


class TestQDriftSettings:
    """Tests for QDriftSettings configuration."""

    def test_default_settings(self):
        """Verify default settings are properly initialized."""
        settings = QDriftSettings()
        assert settings.get("num_samples") == 100
        assert settings.get("seed") == -1
        assert settings.get("merge_duplicate_terms") is True
        assert settings.get("target_accuracy") == 0.0
        assert settings.get("error_bound") == "campbell"
        assert settings.get("weight_threshold") == pytest.approx(1e-12)

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
        assert builder.type_name() == "hamiltonian_unitary_builder"

    def test_can_create_via_registry(self):
        """Test that QDrift can be created via the algorithm registry."""
        builder = create("hamiltonian_unitary_builder", "qdrift")
        assert isinstance(builder, QDrift)

    def test_can_create_with_settings(self):
        """Test that QDrift can be created with custom settings."""
        builder = create("hamiltonian_unitary_builder", "qdrift", num_samples=200, seed=42)
        assert builder.settings().get("num_samples") == 200
        assert builder.settings().get("seed") == 42


class TestQDriftConstruction:
    """Tests for QDrift time evolution construction."""

    def test_returns_unitary_representation(self):
        """Test that run returns a UnitaryRepresentation."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = QDrift(num_samples=10, seed=42, time=0.1)
        unitary = builder.run(hamiltonian)

        assert isinstance(unitary, UnitaryRepresentation)
        container = unitary.get_container()
        assert isinstance(container, PauliProductFormulaContainer)

    def test_correct_number_of_samples(self):
        """Test that the container has at most num_samples terms (fewer after merging)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        num_samples = 50
        builder = QDrift(num_samples=num_samples, seed=42, time=0.1)
        unitary = builder.run(hamiltonian)

        container = unitary.get_container()
        # Duplicate-term merging may reduce the count below num_samples
        assert len(container.step_terms) <= num_samples
        assert len(container.step_terms) >= 1

    def test_exact_sample_count_without_merging(self):
        """Disabling merge_duplicate_terms gives exactly num_samples terms."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        num_samples = 50
        builder = QDrift(num_samples=num_samples, seed=42, merge_duplicate_terms=False, time=0.1)
        unitary = builder.run(hamiltonian)

        assert len(unitary.get_container().step_terms) == num_samples

    def test_reproducible_with_seed(self):
        """Test that results are reproducible when using a seed."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Y", "Z"], coefficients=[1.0, 0.5, 0.25])

        builder1 = QDrift(num_samples=20, seed=12345, time=0.1)
        builder2 = QDrift(num_samples=20, seed=12345, time=0.1)

        unitary1 = builder1.run(hamiltonian)
        unitary2 = builder2.run(hamiltonian)

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

        builder1 = QDrift(num_samples=30, seed=42, time=0.1)
        builder2 = QDrift(num_samples=30, seed=123, time=0.1)

        unitary1 = builder1.run(hamiltonian)
        unitary2 = builder2.run(hamiltonian)

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
        builder = QDrift(num_samples=num_samples, seed=42, time=time)
        unitary = builder.run(hamiltonian)

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

        builder = QDrift(num_samples=num_samples, seed=42, time=time)
        unitary = builder.run(hamiltonian)

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
        builder = QDrift(num_samples=10, seed=42, time=0.1)
        unitary = builder.run(hamiltonian)

        container = unitary.get_container()
        # Single term: all 10 identical samples merge into one rotation
        assert len(container.step_terms) == 1
        assert container.step_terms[0].pauli_term == {0: "X"}

    def test_single_term_hamiltonian(self):
        """Test QDrift with a single-term Hamiltonian."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0])
        num_samples = 20
        time = 0.1
        builder = QDrift(num_samples=num_samples, seed=42, time=time)
        unitary = builder.run(hamiltonian)

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
        builder = QDrift(num_samples=10, seed=42, time=0.1)
        with pytest.raises(ValueError, match="Non-Hermitian"):
            builder.run(hamiltonian)

    def test_negative_coefficients(self):
        """Test that negative coefficients are handled correctly."""
        hamiltonian = QubitHamiltonian(
            pauli_strings=["X", "Z"],
            coefficients=[-1.0, 0.5],
        )
        time = 0.1
        num_samples = 20
        builder = QDrift(num_samples=num_samples, seed=42, time=time)
        unitary = builder.run(hamiltonian)

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
        builder = QDrift(num_samples=50, seed=42, time=0.1)
        unitary = builder.run(hamiltonian)

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
        builder = QDrift(num_samples=10, seed=42, time=1.0)
        unitary = builder.run(hamiltonian)

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


class TestQDriftTargetAccuracy:
    """Tests for epsilon/T-driven auto-sample-count via the Campbell bound."""

    def test_auto_num_samples_matches_campbell_bound(self):
        """N is auto-computed as ceil(2 lambda^2 t^2 / eps) when target_accuracy is set."""
        coeffs = [1.0, 0.5, 0.25]
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Y", "Z"], coefficients=coeffs)
        lam = sum(abs(c) for c in coeffs)
        t = 0.4
        eps = 1e-2
        expected_n = int(np.ceil(2.0 * (lam * t) ** 2 / eps))

        builder = QDrift(num_samples=1, target_accuracy=eps, seed=42, time=t, merge_duplicate_terms=False)
        unitary = builder.run(hamiltonian)
        assert len(unitary.get_container().step_terms) == expected_n

    def test_manual_num_samples_acts_as_floor(self):
        """num_samples wins when larger than the auto-computed value."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        # Loose target -> auto N is tiny; manual floor dominates.
        builder = QDrift(num_samples=500, target_accuracy=10.0, seed=42, time=0.1, merge_duplicate_terms=False)
        unitary = builder.run(hamiltonian)
        assert len(unitary.get_container().step_terms) == 500

    def test_target_accuracy_zero_preserves_manual(self):
        """target_accuracy=0.0 disables auto-N (regression)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = QDrift(num_samples=37, target_accuracy=0.0, seed=42, time=0.1, merge_duplicate_terms=False)
        unitary = builder.run(hamiltonian)
        assert len(unitary.get_container().step_terms) == 37

    def test_weight_threshold_filters_small_terms(self):
        """Sub-threshold coefficients are dropped before sampling."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Y", "Z"], coefficients=[1.0, 1e-15, 0.5])
        builder = QDrift(num_samples=200, seed=42, time=0.1, weight_threshold=1e-12)
        unitary = builder.run(hamiltonian)
        # The Y term should never appear in any sample.
        for term in unitary.get_container().step_terms:
            assert "Y" not in term.pauli_term.values()

    def test_invalid_error_bound_raises(self):
        """Unknown error_bound strings are rejected by the settings validator."""
        with pytest.raises(ValueError, match="error_bound"):
            QDrift(target_accuracy=1e-2, error_bound="bogus", time=0.1)


def _container_to_unitary(container) -> np.ndarray:
    """Materialise a PauliProductFormulaContainer as a dense unitary matrix."""
    n = container.num_qubits
    U = np.eye(2**n, dtype=complex)  # noqa: N806
    for term in container.step_terms:
        s = ["I"] * n
        for q, op in term.pauli_term.items():
            s[n - 1 - q] = op  # little-endian
        P = pauli_to_dense_matrix(["".join(s)], np.array([1.0]))  # noqa: N806
        U = expm(-1j * term.angle * P) @ U  # noqa: N806
    return np.linalg.matrix_power(U, container.step_reps)


def _unitary_to_choi(U: np.ndarray) -> np.ndarray:  # noqa: N803
    """Choi matrix of the unitary channel: (U ⊗ I)|Ω⟩⟨Ω|(U ⊗ I)†."""
    d = U.shape[0]
    omega = np.zeros(d * d, dtype=complex)
    for i in range(d):
        omega[i * d + i] = 1.0
    v = np.kron(U, np.eye(d, dtype=complex)) @ omega
    return np.outer(v, v.conj())


def _qdrift_channel_error(hamiltonian, *, eps, t, seeds):
    """Estimate ||Choi(Φ_avg) - Choi(U_exact)||_1 over a list of qDRIFT seeds.

    Campbell (2019) bounds the diamond norm of the channel difference. The
    Choi trace-norm upper-bounds the diamond norm, so we use it as a
    computable proxy. The expectation is over the qDRIFT distribution; we
    approximate it by averaging `len(seeds)` independent unitary samples.
    """
    H = hamiltonian.to_matrix()  # noqa: N806
    U_exact = expm(-1j * H * t)  # noqa: N806
    C_exact = _unitary_to_choi(U_exact)  # noqa: N806

    Us = []  # noqa: N806
    for seed in seeds:
        builder = QDrift(
            target_accuracy=eps,
            time=t,
            seed=seed,
            num_samples=1,
            merge_duplicate_terms=False,
        )
        c = builder.run(hamiltonian).get_container()
        Us.append(_container_to_unitary(c))

    C_avg = np.mean([_unitary_to_choi(U) for U in Us], axis=0)  # noqa: N806
    return float(np.linalg.svd(C_avg - C_exact, compute_uv=False).sum())


def _qdrift_state_trace_error(hamiltonian, *, eps, t, seeds, input_state=None):
    """Trace distance between qDRIFT-averaged output state and exact output state.

    For a fixed pure input |ψ⟩, returns

        0.5 * || E_k[U_k |ψ⟩⟨ψ| U_k^†] - U_exact |ψ⟩⟨ψ| U_exact^† ||_1

    This is bounded by the diamond norm of the channel difference, which
    Campbell (2019) guarantees is at most ε when N is chosen via the bound.
    Unlike the Choi trace norm, there is no factor-of-d slack, so the
    assertion `err <= ε` is tight (subject only to finite-sample noise).
    """
    H = hamiltonian.to_matrix()  # noqa: N806
    U_exact = expm(-1j * H * t)  # noqa: N806
    d = U_exact.shape[0]
    psi = np.zeros(d, dtype=complex) if input_state is None else np.asarray(input_state, dtype=complex)
    if input_state is None:
        psi[0] = 1.0  # |0...0>

    rho_exact = np.outer(U_exact @ psi, (U_exact @ psi).conj())

    rho_avg = np.zeros_like(rho_exact)
    for seed in seeds:
        builder = QDrift(
            target_accuracy=eps,
            time=t,
            seed=seed,
            num_samples=1,
            merge_duplicate_terms=False,
        )
        c = builder.run(hamiltonian).get_container()
        U_k = _container_to_unitary(c)  # noqa: N806
        v = U_k @ psi
        rho_avg = rho_avg + np.outer(v, v.conj())
    rho_avg = rho_avg / len(seeds)

    return 0.5 * float(np.linalg.svd(rho_avg - rho_exact, compute_uv=False).sum())


class TestQDriftAccuracyBound:
    """Empirical verification that auto-N qDRIFT achieves the target accuracy.

    Campbell's bound is a statement about the *expected channel* — the
    diamond-norm distance between the qDRIFT random-unitary channel and
    the exact channel exp(-iHt)·exp(+iHt). A single sampled unitary's
    spectral error is O(λt/√N) and would routinely exceed ε; that is
    expected and not a bug. These tests therefore average Choi matrices
    over many seeds and compare to the exact channel.

    We use a tolerance of ~10·ε because (i) the Choi trace-norm upper
    bounds the diamond norm by a factor up to d = 2**nq, and (ii) the
    Monte-Carlo estimate from a finite number of seeds has variance of
    order λt/√(n_seeds·N).
    """

    TOLERANCE_FACTOR = 10  # accept up to 10x the Campbell bound (same order of magnitude)

    def test_accuracy_bound_1q(self):
        """1-qubit X + 0.5 Z: averaged-channel error stays within 10·ε."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        eps = 0.05
        t = 0.5
        seeds = list(range(40))
        err = _qdrift_channel_error(hamiltonian, eps=eps, t=t, seeds=seeds)
        assert err <= self.TOLERANCE_FACTOR * eps, (
            f"Channel error {err:.4e} exceeds {self.TOLERANCE_FACTOR}*ε = {self.TOLERANCE_FACTOR * eps:.4e}"
        )

    def test_accuracy_bound_2q(self):
        """2-qubit TFIM-style ZZ + 0.6 XI + 0.6 IX: averaged-channel error within 10·ε."""
        hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XI", "IX"], coefficients=[1.0, 0.6, 0.6])
        eps = 0.05
        t = 0.3
        seeds = list(range(40))
        err = _qdrift_channel_error(hamiltonian, eps=eps, t=t, seeds=seeds)
        assert err <= self.TOLERANCE_FACTOR * eps, (
            f"Channel error {err:.4e} exceeds {self.TOLERANCE_FACTOR}*ε = {self.TOLERANCE_FACTOR * eps:.4e}"
        )

    def test_error_scales_with_epsilon(self):
        """Tightening ε reduces the averaged-channel error (Campbell scaling)."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        t = 0.5
        seeds = list(range(40))
        err_loose = _qdrift_channel_error(hamiltonian, eps=0.1, t=t, seeds=seeds)
        err_tight = _qdrift_channel_error(hamiltonian, eps=0.02, t=t, seeds=seeds)
        assert err_tight < err_loose, (
            f"Expected channel error to decrease with smaller ε; "
            f"got err(0.02)={err_tight:.4e} >= err(0.1)={err_loose:.4e}"
        )


class TestQDriftStrictAccuracyBound:
    """Strict verification: qDRIFT-averaged state stays within ε of the exact state.

    Uses the state trace distance from a fixed input |0...0⟩, which is
    bounded by the diamond norm of the channel difference. Campbell (2019)
    guarantees diamond norm ≤ ε when N = ceil(2λ²t²/ε). With enough seeds
    to suppress Monte-Carlo noise, the assertion err ≤ ε is tight.
    """

    SEEDS: ClassVar[list[int]] = list(range(500))

    def test_state_trace_within_eps_1q(self):
        """1-qubit X + 0.5 Z: state trace distance ≤ ε."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        eps = 0.05
        t = 0.5
        err = _qdrift_state_trace_error(hamiltonian, eps=eps, t=t, seeds=self.SEEDS)
        assert err <= eps, f"State trace distance {err:.4e} exceeds ε = {eps:.4e}"

    def test_state_trace_within_eps_2q(self):
        """2-qubit ZZ + 0.6 XI + 0.6 IX: state trace distance ≤ ε."""
        hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XI", "IX"], coefficients=[1.0, 0.6, 0.6])
        eps = 0.05
        t = 0.3
        err = _qdrift_state_trace_error(hamiltonian, eps=eps, t=t, seeds=self.SEEDS)
        assert err <= eps, f"State trace distance {err:.4e} exceeds ε = {eps:.4e}"
