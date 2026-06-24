"""Integration tests for iterative QPE with the SOSSA block encoding.

Tests the full pipeline:
    FactorizedHamiltonianContainer → SOSSABuilder → UnitaryRepresentation
    → ControlledUnitary → SOSSAMapper → Circuit → IQPE → energy

Reference: arXiv:2502.15882v1 (Low et al. 2025)
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.algorithms.controlled_circuit_mapper.sossa_mapper import SOSSAMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit, FactorizedHamiltonianContainer
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .test_helpers import create_test_orbitals

_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"
_PROJECT_ROOT = str(_QS_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Hamiltonian construction (small DFTHC-like H2 data)
# ═══════════════════════════════════════════════════════════════════════════════


def _build_h2_dfthc_data():
    """Construct a small H2-like DFTHC factorized Hamiltonian for testing.

    Uses N=2 orbitals, R=1 rank, B=1 basis, C=1 copy with manually chosen
    matrices that produce a known spectrum.

    Returns a dict with all tensor data needed for SOSSA.
    """
    n_orb = 2  # spatial orbitals
    n_ranks = 1  # ranks
    n_bases = 1  # bases
    n_copies = 1  # copies

    # Symmetric one-body matrix (adjusted for Majorana representation)
    h1 = np.array(
        [
            [0.3, 0.1],
            [0.1, -0.2],
        ]
    )

    # Basis vectors: unit vectors in R^N for each (r, b)
    # Tensor shape is (R, B, N)
    basis_vectors = np.array([[[1.0 / sqrt(2), 1.0 / sqrt(2)]]])

    # Two-body weights: [R, B, C]
    two_body_weights = np.array([[[0.15]]])

    # Identity weights (WB): [R, C]
    identity_weight = np.array([[0.08]])

    return {
        "h1": h1,
        "basis_vectors": basis_vectors,
        "two_body_weights": two_body_weights,
        "identity_weight": identity_weight,
        "N": n_orb,
        "R": n_ranks,
        "B": n_bases,
        "C": n_copies,
    }


def _build_dfthc_hamiltonian_matrix(h1, basis_vectors, two_body_weights, identity_weight):
    """Build the SOSSA gap Hamiltonian matrix via Jordan-Wigner mapping.

    Constructs H_gap from the SOS generators (D1² + Q1² + SF²):
        H_gap = Σ_{k: w_k>0} w_k (n_k - 1)² + Σ_{k: w_k<0} |w_k| n_k²
              + ½ Σ_{r,c} (W^{(rc)}·I + Σ_b w_b^{(rc)} L_b^{(r)})²

    where n_k = Σ_{p,q} V_{pk} V_{qk} E_{pq} is the occupation of eigenbasis
    orbital k (eigenvectors V from diagonalizing h1).

    Reference: Eq. 20-21, 29 in arXiv:2502.15882v1.
    """
    num_orbitals = h1.shape[0]
    n_ranks, b_dim, _ = basis_vectors.shape
    _, n_copies = identity_weight.shape
    num_spin_orbitals = 2 * num_orbitals
    dim = 2**num_spin_orbitals

    # Jordan-Wigner operators
    eye2 = np.eye(2, dtype=complex)
    pauli_z = np.diag([1.0, -1.0]).astype(complex)
    sp = np.array([[0, 0], [1, 0]], dtype=complex)  # a† = |1><0|

    def adag(i):
        ops = [pauli_z if j < i else (sp if j == i else eye2) for j in range(num_spin_orbitals)]
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # Excitation operator E_{pq} = Sum_sigma a†_{p,sigma} a_{q,sigma}
    epq_cache = {}

    def excitation_pq(p, q):
        if (p, q) not in epq_cache:
            mat = np.zeros((dim, dim), dtype=complex)
            for sigma in range(2):
                c_dag = adag(2 * p + sigma)
                c = adag(2 * q + sigma).conj().T
                mat += c_dag @ c
            epq_cache[(p, q)] = mat
        return epq_cache[(p, q)]

    # Eigendecompose h1 for D1/Q1 generators
    eigvals, eigvecs = np.linalg.eigh(h1)

    # 1) D1/Q1 terms: linear number operators in h1 eigenbasis
    # D1 generator is proportional to a†_k (creation), so O†O = (1-n_k sigma) summed
    # over spins gives w_k(2 - n_k).
    # Q1 generator is proportional to a_k (annihilation), so O†O = n_k sigma summed
    # over spins gives |w_k| * n_k.
    h_1b = np.zeros((dim, dim), dtype=complex)
    for k in range(num_orbitals):
        # n_k in eigenbasis: n_k = Σ_{p,q} V_{pk} V_{qk} E_{pq}
        n_k_op = np.zeros((dim, dim), dtype=complex)
        for p in range(num_orbitals):
            for q in range(num_orbitals):
                coeff = eigvecs[p, k] * eigvecs[q, k]
                if abs(coeff) > 1e-15:
                    n_k_op += coeff * excitation_pq(p, q)

        w_k = eigvals[k]
        if w_k > 0:
            # D1: w_k * (2I - n_k)
            h_1b += w_k * (2.0 * np.eye(dim) - n_k_op)
        else:
            # Q1: |w_k| * n_k
            h_1b += abs(w_k) * n_k_op

    # 2) SF squares: ½ Σ_{r,c} (W·I + Σ_b w_b L_b)²
    h_2b = np.zeros((dim, dim), dtype=complex)
    for r in range(n_ranks):
        for c_idx in range(n_copies):
            w_rc = identity_weight[r, c_idx] - np.sum(two_body_weights[r, :, c_idx])
            m_op = w_rc * np.eye(dim, dtype=complex)
            for b in range(b_dim):
                l_b = np.zeros((dim, dim), dtype=complex)
                for p in range(num_orbitals):
                    for q in range(num_orbitals):
                        l_b += basis_vectors[r, b, p] * basis_vectors[r, b, q] * excitation_pq(p, q)
                m_op += two_body_weights[r, b, c_idx] * l_b
            h_2b += 0.5 * (m_op @ m_op)

    return (h_1b + h_2b).real


def _get_ground_state_and_energy(h_matrix, num_orbitals, nalpha=1, nbeta=1):
    """Diagonalize H_gap and return ground state within the correct particle sector.

    Returns:
        (ground_energy, ground_state_vector) in the Q# spin-blocked basis ordering.

    """
    dim = h_matrix.shape[0]

    # Build number operator
    n_hat = np.diag([bin(x).count("1") for x in range(dim)]).astype(float)

    eigenvalues, eigenvectors = np.linalg.eigh(h_matrix)

    # Filter to correct particle number sector
    target_n = nalpha + nbeta
    sector_indices = [
        i for i in range(len(eigenvalues)) if round(eigenvectors[:, i] @ n_hat @ eigenvectors[:, i]) == target_n
    ]

    if not sector_indices:
        # Fall back to full spectrum if no particle sector matches
        sector_indices = list(range(len(eigenvalues)))

    # Permute from Python Kron convention to Q# convention
    perm = _python_to_qsharp_permutation(num_orbitals)
    gs_idx = sector_indices[0]
    gs_energy = eigenvalues[gs_idx]
    gs_vec = eigenvectors[:, gs_idx]

    # Apply permutation
    gs_vec_qs = np.zeros(dim)
    for i in range(dim):
        gs_vec_qs[perm[i]] = gs_vec[i]

    return gs_energy, gs_vec_qs


def _python_to_qsharp_permutation(num_orbitals):
    """Compute basis index permutation from Python Kron to Q# convention."""
    n_qubits = 2 * num_orbitals
    dim = 2**n_qubits
    perm = np.zeros(dim, dtype=int)
    for i in range(dim):
        k = 0
        for b in range(n_qubits):
            if (i >> b) & 1:
                j = n_qubits - 1 - b
                p, sigma = j // 2, j % 2
                qs_qubit = sigma * num_orbitals + p
                k |= 1 << qs_qubit
        perm[i] = k
    return perm


# ═══════════════════════════════════════════════════════════════════════════════
# SOSSA QPE helper
# ═══════════════════════════════════════════════════════════════════════════════


# Short name -> registry name for outer_prepare AlgorithmRef
_OUTER_PREP_MAP = {
    "alias_sampling": "alias_sampling",
    "dense_pure": "dense_pure_state",
    "qrom": "qrom",
}


def _sossa_qpe_circuit_builder_ref(
    num_bits: int = 4,
    *,
    outer_prepare_algorithm: str = "dense_pure",
    inner_prepare_algorithm: str = "direct",
    select_algorithm: str = "direct",
    coefficient_bit_precision: int = 10,
    rotation_bit_precision: int = 10,
) -> AlgorithmRef:
    """Return an AlgorithmRef for iterative QPE with SOSSA."""
    ref_name = _OUTER_PREP_MAP.get(outer_prepare_algorithm, outer_prepare_algorithm)
    return AlgorithmRef(
        "qpe_circuit_builder",
        "qdk_iterative",
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper",
            "sossa",
            outer_prepare=AlgorithmRef("state_prep", ref_name),
            inner_prepare_algorithm=inner_prepare_algorithm,
            select_algorithm=select_algorithm,
            coefficient_bit_precision=coefficient_bit_precision,
            rotation_bit_precision=rotation_bit_precision,
        ),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
    )


def _energy_to_qpe_phase(energy_gap, lambda_sos):
    """Convert energy gap to QPE phase for the SOS walk operator.

    For SOS walk: cos(2πφ) = E_gap / Λ - 1
    """
    cos_val = energy_gap / lambda_sos - 1.0
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.acos(cos_val) / (2 * math.pi)


def _energy_to_k_sos(e_gap, num_bits, lambda_sos):
    """Predict the most likely QPE integer for a given e_gap (SOS walk).

    Inverts: E_gap = Λ(1 + cos(2πφ))  →  φ = arccos(E_gap/Λ - 1) / (2π)
    Returns (k, conjugate_k) where k = round(φ · 2^n).
    """
    phi = _energy_to_qpe_phase(e_gap, lambda_sos)
    total_states = 2**num_bits
    k = round(phi * total_states)
    conjugate_k = total_states - k if k != 0 else 0
    return k, conjugate_k


def _run_sossa_qpe(num_bits, mapper_kwargs=None):
    """Helper: run SOSSA QPE on H2 data and assert measured phase matches expected.

    Uses IQPE with the given mapper configuration, computes k_measured from the
    result phase, and asserts it matches k_expect from exact diagonalization.
    """
    data = _build_h2_dfthc_data()
    n_orb, n_ranks, n_bases, n_copies = data["N"], data["R"], data["B"], data["C"]

    # Build reference Hamiltonian matrix and diagonalize
    h_matrix = _build_dfthc_hamiltonian_matrix(
        data["h1"], data["basis_vectors"], data["two_body_weights"], data["identity_weight"]
    )
    gs_energy, gs_vec = _get_ground_state_and_energy(h_matrix, n_orb, nalpha=1, nbeta=1)

    # Create FactorizedHamiltonianContainer
    orbitals = create_test_orbitals(n_orb)
    inactive_fock = np.zeros((n_orb, n_orb))
    fh = FactorizedHamiltonianContainer(
        n_ranks,
        n_bases,
        n_copies,
        0.0,
        data["basis_vectors"].flatten(),
        data["two_body_weights"].flatten(),
        data["h1"],
        data["identity_weight"],
        inactive_fock,
        orbitals,
    )

    # Build SOSSA unitary and get normalization
    builder = SOSSABuilder()
    unitary_rep = builder.run(fh)
    container = unitary_rep.get_container()
    lambda_sos = container.normalization

    # Expected QPE integer
    k_expect, _ = _energy_to_k_sos(gs_energy, num_bits, lambda_sos)

    # Prepare ground state
    num_system_qubits = 2 * n_orb
    state_prep_params = {
        "rowMap": list(range(num_system_qubits - 1, -1, -1)),
        "stateVector": gs_vec.real.tolist(),
        "expansionOps": [],
        "numQubits": num_system_qubits,
    }
    qsharp_factory = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
        parameter=state_prep_params,
    )
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
    state_prep = Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

    # Run IQPE
    mkw = mapper_kwargs or {}
    iqpe = IterativePhaseEstimation(shots_per_bit=5)
    iqpe.settings().set("qpe_circuit_builder", _sossa_qpe_circuit_builder_ref(num_bits=num_bits, **mkw))
    iqpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
    )

    result = iqpe.run(
        state_preparation=state_prep,
        qubit_hamiltonian=fh,
    )

    # Convert measured phase to k
    total_states = 2**num_bits
    k_raw = round(result.phase_fraction * total_states)
    k_measured = min(k_raw % total_states, (total_states - k_raw) % total_states)
    k_expect_sym = min(k_expect, total_states - k_expect) if k_expect != 0 else 0

    # Allow ±1 bin tolerance: IQPE with few shots_per_bit has ~23% error
    # probability on the last bit when the phase is between bin boundaries.
    assert abs(k_measured - k_expect_sym) <= 1, (
        f"Expected k={k_expect_sym}±1, got k={k_measured}, "
        f"phase_fraction={result.phase_fraction:.6f}, "
        f"raw_energy={result.raw_energy:.6f}, "
        f"gs_energy={gs_energy:.6f}, lambda={lambda_sos:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAQPEIntegration:
    """Integration tests for SOSSA QPE using the full builder → mapper → IQPE pipeline."""

    def test_sossa_builder_to_mapper_produces_circuit(self):
        """Verify the SOSSA pipeline produces a valid circuit with Q# ops.

        Tests: FactorizedHamiltonian → SOSSABuilder → UnitaryRep → SOSSAMapper → Circuit.
        """
        data = _build_h2_dfthc_data()
        n_orb, n_ranks, n_bases, n_copies = data["N"], data["R"], data["B"], data["C"]

        # Create FactorizedHamiltonianContainer
        h1 = data["h1"]
        u_matrices = data["basis_vectors"].flatten()
        w_matrices = data["two_body_weights"].flatten()
        wb_matrix = data["identity_weight"]
        orbitals = create_test_orbitals(n_orb)
        inactive_fock = np.zeros((n_orb, n_orb))

        fh = FactorizedHamiltonianContainer(
            n_ranks,
            n_bases,
            n_copies,
            0.0,
            u_matrices,
            w_matrices,
            h1,
            wb_matrix,
            inactive_fock,
            orbitals,
        )

        # Build SOSSA unitary representation
        builder = SOSSABuilder()
        unitary_rep = builder.run(fh)
        assert isinstance(unitary_rep, UnitaryRepresentation)

        container = unitary_rep.get_container()
        assert isinstance(container, SOSSAContainer)
        assert container.normalization > 0

        # Map to controlled circuit
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])
        mapper = SOSSAMapper()
        mapper.settings().set("outer_prepare", AlgorithmRef("state_prep", "dense_pure_state"))
        mapper.settings().set("inner_prepare_algorithm", "direct")
        mapper.settings().set("select_algorithm", "direct")
        circuit = mapper.run(controlled_unitary)
        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None

    def test_sossa_qpe_ground_state_energy(self):
        """End-to-end test: SOSSA QPE recovers ground state energy from DFTHC data.

        Uses a small H2-like DFTHC decomposition, runs IQPE with SOSSA block
        encoding, and verifies the measured energy matches exact diagonalization.
        """
        data = _build_h2_dfthc_data()
        n_orb, n_ranks, n_bases, n_copies = data["N"], data["R"], data["B"], data["C"]

        # Build the Hamiltonian matrix for reference diagonalization
        h_matrix = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        gs_energy, gs_vec = _get_ground_state_and_energy(h_matrix, n_orb, nalpha=1, nbeta=1)

        # Create FactorizedHamiltonianContainer
        h1 = data["h1"]
        u_matrices = data["basis_vectors"].flatten()
        w_matrices = data["two_body_weights"].flatten()
        wb_matrix = data["identity_weight"]
        orbitals = create_test_orbitals(n_orb)
        inactive_fock = np.zeros((n_orb, n_orb))

        fh = FactorizedHamiltonianContainer(
            n_ranks,
            n_bases,
            n_copies,
            0.0,
            u_matrices,
            w_matrices,
            h1,
            wb_matrix,
            inactive_fock,
            orbitals,
        )

        # Build SOSSA
        builder = SOSSABuilder()
        unitary_rep = builder.run(fh)
        container = unitary_rep.get_container()
        lambda_sos = container.normalization

        # Prepare ground state
        num_system_qubits = 2 * n_orb
        state_prep_params = {
            "rowMap": list(range(num_system_qubits - 1, -1, -1)),
            "stateVector": gs_vec.real.tolist(),
            "expansionOps": [],
            "numQubits": num_system_qubits,
        }
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        )
        qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
        state_prep = Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op)

        # Run IQPE
        num_bits = 5
        iqpe = IterativePhaseEstimation(shots_per_bit=5)
        iqpe.settings().set("qpe_circuit_builder", _sossa_qpe_circuit_builder_ref(num_bits=num_bits))
        iqpe.settings().set(
            "circuit_executor",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )

        result = iqpe.run(
            state_preparation=state_prep,
            qubit_hamiltonian=fh,
        )

        # Verify: for SOS walk, raw_energy = Λ(1 + cos(2πφ)) + energy_shift
        measured_e_gap = result.raw_energy - container.energy_shift

        # With 5 bits, discretization error ~ Λ * 2π / 2^5 ≈ Λ * 0.2
        discretization_tol = lambda_sos * 2 * math.pi / (2**num_bits) + 0.05
        assert abs(measured_e_gap - gs_energy) < discretization_tol, (
            f"Energy mismatch: measured E_gap={measured_e_gap:.6f}, "
            f"expected={gs_energy:.6f}, tol={discretization_tol:.6f}"
        )

    def test_sossa_qpe_direct_workflow(self):
        """Test SOSSA QPE by directly constructing the pipeline (no registry).

        This test bypasses AlgorithmRef and directly calls:
            SOSSABuilder → SOSSAMapper → IQPE circuit construction
        to verify the workflow end-to-end.
        """
        data = _build_h2_dfthc_data()
        n_orb, n_ranks, n_bases, n_copies = data["N"], data["R"], data["B"], data["C"]

        # Create FactorizedHamiltonianContainer
        orbitals = create_test_orbitals(n_orb)
        inactive_fock = np.zeros((n_orb, n_orb))
        fh = FactorizedHamiltonianContainer(
            n_ranks,
            n_bases,
            n_copies,
            0.0,
            data["basis_vectors"].flatten(),
            data["two_body_weights"].flatten(),
            data["h1"],
            data["identity_weight"],
            inactive_fock,
            orbitals,
        )

        # Step 1: SOSSABuilder → UnitaryRepresentation
        builder = SOSSABuilder()
        unitary_rep = builder.run(fh)
        container = unitary_rep.get_container()
        assert isinstance(container, SOSSAContainer)

        # Step 2: SOSSAMapper → Circuit
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])
        mapper = SOSSAMapper()
        mapper.settings().set("outer_prepare", AlgorithmRef("state_prep", "dense_pure_state"))
        mapper.settings().set("inner_prepare_algorithm", "direct")
        mapper.settings().set("select_algorithm", "direct")
        circuit = mapper.run(controlled_unitary)

        # Verify circuit has all required components
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

        # Step 3: Verify normalization is accessible
        lambda_sos = container.normalization
        assert lambda_sos > 0

        # Step 4: Compute expected spectrum
        h_matrix = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        eigenvalues = np.linalg.eigvalsh(h_matrix)
        # H_gap should be positive semi-definite
        assert eigenvalues[0] >= -1e-10, f"H_gap has negative eigenvalue: {eigenvalues[0]}"

    def test_sossa_normalization_bounds_spectrum(self):
        """Verify that SOSSA normalization Λ bounds the spectrum: all eigenvalues ≤ 2Λ.

        For a valid SOS walk, E_gap ∈ [0, 2Λ], so all eigenvalues of H_gap
        must satisfy 0 ≤ E ≤ 2Λ.
        """
        data = _build_h2_dfthc_data()
        n_orb, n_ranks, n_bases, n_copies = data["N"], data["R"], data["B"], data["C"]

        orbitals = create_test_orbitals(n_orb)
        inactive_fock = np.zeros((n_orb, n_orb))
        fh = FactorizedHamiltonianContainer(
            n_ranks,
            n_bases,
            n_copies,
            0.0,
            data["basis_vectors"].flatten(),
            data["two_body_weights"].flatten(),
            data["h1"],
            data["identity_weight"],
            inactive_fock,
            orbitals,
        )

        builder = SOSSABuilder()
        unitary_rep = builder.run(fh)
        container = unitary_rep.get_container()
        lambda_sos = container.normalization

        h_matrix = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        eigenvalues = np.linalg.eigvalsh(h_matrix)

        # All eigenvalues should be ≤ 2Λ (with small numerical tolerance)
        assert np.all(eigenvalues <= 2 * lambda_sos + 1e-10), (
            f"Eigenvalue {eigenvalues.max():.6f} exceeds 2Λ={2 * lambda_sos:.6f}"
        )

    @pytest.mark.parametrize("num_bits", [3, 5])
    def test_sossa_qpe(self, num_bits):
        """QPE with direct (non-alias) config should match expected phase index."""
        _run_sossa_qpe(num_bits)

    @pytest.mark.parametrize(
        "mapper_overrides",
        [
            {
                "outer_prepare_algorithm": "alias_sampling",
                "coefficient_bit_precision": 4,
            },
            {
                "inner_prepare_algorithm": "controlled_alias_sampling",
                "coefficient_bit_precision": 4,
            },
        ],
        ids=["alias_outer", "alias_inner"],
    )
    def test_sossa_qpe_features(self, mapper_overrides):
        """QPE with individual features enabled (3 phase bits, H2 data)."""
        _run_sossa_qpe(num_bits=3, mapper_kwargs=mapper_overrides)
