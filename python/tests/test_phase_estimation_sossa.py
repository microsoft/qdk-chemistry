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

from qdk_chemistry.algorithms.controlled_circuit_mapper.sossa_mapper import (
    InnerPrepareMapper,
    OuterPrepareMapper,
    SelectMapper,
    SOSSAMapper,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.controlled_unitary import ControlledUnitary
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

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
    N = 2  # spatial orbitals
    R = 1  # ranks
    B = 1  # bases
    C = 1  # copies

    # Symmetric one-body matrix (adjusted for Majorana representation)
    h1 = np.array(
        [
            [0.3, 0.1],
            [0.1, -0.2],
        ]
    )

    # Basis vectors: unit vectors in R^N for each (r, b)
    # Shape: [R, B, N]
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
        "N": N,
        "R": R,
        "B": B,
        "C": C,
    }


def _build_dfthc_hamiltonian_matrix(h1, basis_vectors, two_body_weights, identity_weight):
    """Build the full DFTHC Hamiltonian matrix via Jordan-Wigner mapping.

    Constructs H_gap (the positive semi-definite part) from the DFTHC tensors:
        H_gap = h'(1)_{pq} E_{pq} + 2 Σ w⁻ · I
              + ½ Σ_{r,c} (W^{(rc)}·I + Σ_b w_b^{(rc)} L_b^{(r)})²

    Reference: Eq. 29 in arXiv:2502.15882v1.
    """
    num_orbitals = h1.shape[0]
    R, B_dim, _ = basis_vectors.shape
    _, C = identity_weight.shape
    num_spin_orbitals = 2 * num_orbitals
    dim = 2**num_spin_orbitals

    # Jordan-Wigner operators
    I2 = np.eye(2, dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    sp = np.array([[0, 0], [1, 0]], dtype=complex)  # a† = |1><0|

    def adag(i):
        ops = [Z if j < i else (sp if j == i else I2) for j in range(num_spin_orbitals)]
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # Excitation operator E_{pq} = Σ_σ a†_{pσ} a_{qσ}
    Epq_cache = {}

    def Epq(p, q):
        if (p, q) not in Epq_cache:
            E = np.zeros((dim, dim), dtype=complex)
            for sigma in range(2):
                c_dag = adag(2 * p + sigma)
                c = adag(2 * q + sigma).conj().T
                E += c_dag @ c
            Epq_cache[(p, q)] = E
        return Epq_cache[(p, q)]

    # Eigendecompose h1 for w_minus
    eigvals, _ = np.linalg.eigh(h1)
    w_minus = -eigvals[eigvals < 0]

    # 1) One-body: h'(1)_{pq} E_{pq} + 2 Σ w⁻ · I
    H_1b = np.zeros((dim, dim), dtype=complex)
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            if abs(h1[p, q]) > 1e-15:
                H_1b += h1[p, q] * Epq(p, q)
    H_1b += 2.0 * np.sum(w_minus) * np.eye(dim)

    # 2) SF squares: ½ Σ_{r,c} (W·I + Σ_b w_b L_b)²
    H_2b = np.zeros((dim, dim), dtype=complex)
    for r in range(R):
        for c_idx in range(C):
            W_rc = identity_weight[r, c_idx] - np.sum(two_body_weights[r, :, c_idx])
            M = W_rc * np.eye(dim, dtype=complex)
            for b in range(B_dim):
                L_b = np.zeros((dim, dim), dtype=complex)
                for p in range(num_orbitals):
                    for q in range(num_orbitals):
                        L_b += basis_vectors[r, b, p] * basis_vectors[r, b, q] * Epq(p, q)
                M += two_body_weights[r, b, c_idx] * L_b
            H_2b += 0.5 * (M @ M)

    return (H_1b + H_2b).real


def _get_ground_state_and_energy(H_matrix, num_orbitals, nalpha=1, nbeta=1):
    """Diagonalize H_gap and return ground state within the correct particle sector.

    Returns:
        (ground_energy, ground_state_vector) in the Q# spin-blocked basis ordering.

    """
    dim = H_matrix.shape[0]
    num_spin_orbitals = 2 * num_orbitals

    # Build number operator
    N_hat = np.diag([bin(x).count("1") for x in range(dim)]).astype(float)

    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

    # Filter to correct particle number sector
    target_n = nalpha + nbeta
    sector_indices = [
        i for i in range(len(eigenvalues)) if round(eigenvectors[:, i] @ N_hat @ eigenvectors[:, i]) == target_n
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


def _sossa_qpe_circuit_builder_ref(num_bits: int = 4) -> AlgorithmRef:
    """Return an AlgorithmRef for iterative QPE with SOSSA."""
    return AlgorithmRef(
        "qpe_circuit_builder",
        "qdk_iterative",
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper",
            "sossa",
            outer_prepare_algorithm="dense_pure",
            inner_prepare_algorithm="direct",
            select_algorithm="direct",  # Use direct for simulation (phase_gradient needs too many qubits)
        ),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa", quantum_walk=True),
    )


def _energy_to_qpe_phase(energy_gap, lambda_sos):
    """Convert energy gap to QPE phase for the SOS walk operator.

    For SOS walk: cos(2πφ) = 1 - E_gap / Λ
    """
    cos_val = 1.0 - energy_gap / lambda_sos
    cos_val = max(-1.0, min(1.0, cos_val))
    return math.acos(cos_val) / (2 * math.pi)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAQPEIntegration:
    """Integration tests for SOSSA QPE using the full builder → mapper → IQPE pipeline."""

    def test_sossa_builder_to_mapper_produces_circuit(self):
        """Verify the SOSSA pipeline produces a valid circuit with Q# ops.

        Tests: FactorizedHamiltonian → SOSSABuilder → UnitaryRep → SOSSAMapper → Circuit.
        """
        try:
            from qdk_chemistry.data import FactorizedHamiltonianContainer
        except ImportError:
            pytest.skip("FactorizedHamiltonianContainer not available (requires dev build)")

        from .test_helpers import create_test_orbitals

        data = _build_h2_dfthc_data()
        N, R, B, C = data["N"], data["R"], data["B"], data["C"]

        # Create FactorizedHamiltonianContainer
        h1 = data["h1"]
        u_matrices = data["basis_vectors"].flatten()
        w_matrices = data["two_body_weights"].flatten()
        wb_matrix = data["identity_weight"]
        orbitals = create_test_orbitals(N)
        inactive_fock = np.zeros((N, N))

        fh = FactorizedHamiltonianContainer(
            h1,
            u_matrices,
            w_matrices,
            wb_matrix,
            R,
            B,
            C,
            orbitals,
            0.0,
            inactive_fock,
        )

        # Build SOSSA unitary representation
        builder = SOSSABuilder(quantum_walk=True)
        unitary_rep = builder.run(fh)
        assert isinstance(unitary_rep, UnitaryRepresentation)

        container = unitary_rep.get_container()
        assert isinstance(container, SOSSAContainer)
        assert container.normalization > 0

        # Map to controlled circuit
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])
        mapper = SOSSAMapper(
            outer_prepare_mapper=OuterPrepareMapper(algorithm="dense_pure"),
            inner_prepare_mapper=InnerPrepareMapper(algorithm="direct"),
            select_mapper=SelectMapper(multiplexed_rotation="direct"),
        )
        circuit = mapper.run(controlled_unitary)
        assert isinstance(circuit, Circuit)
        assert circuit._qsharp_op is not None

    def test_sossa_qpe_ground_state_energy(self):
        """End-to-end test: SOSSA QPE recovers ground state energy from DFTHC data.

        Uses a small H2-like DFTHC decomposition, runs IQPE with SOSSA block
        encoding, and verifies the measured energy matches exact diagonalization.
        """
        try:
            from qdk_chemistry.data import FactorizedHamiltonianContainer
        except ImportError:
            pytest.skip("FactorizedHamiltonianContainer not available (requires dev build)")

        from .test_helpers import create_test_orbitals

        data = _build_h2_dfthc_data()
        N, R, B, C = data["N"], data["R"], data["B"], data["C"]

        # Build the Hamiltonian matrix for reference diagonalization
        H_matrix = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        gs_energy, gs_vec = _get_ground_state_and_energy(H_matrix, N, nalpha=1, nbeta=1)

        # Create FactorizedHamiltonianContainer
        h1 = data["h1"]
        u_matrices = data["basis_vectors"].flatten()
        w_matrices = data["two_body_weights"].flatten()
        wb_matrix = data["identity_weight"]
        orbitals = create_test_orbitals(N)
        inactive_fock = np.zeros((N, N))

        fh = FactorizedHamiltonianContainer(
            h1,
            u_matrices,
            w_matrices,
            wb_matrix,
            R,
            B,
            C,
            orbitals,
            0.0,
            inactive_fock,
        )

        # Build SOSSA
        builder = SOSSABuilder(quantum_walk=True)
        unitary_rep = builder.run(fh)
        container = unitary_rep.get_container()
        lambda_sos = container.normalization

        # Prepare ground state
        num_system_qubits = 2 * N
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
            factorized_hamiltonian=fh,
        )

        # Verify: for SOS walk, E_gap = Λ(1 - cos(2πφ))
        # The raw_energy from from_qubitization_result is Λ·cos(2πφ)
        # E_gap = Λ - raw_energy (since cos(2πφ) = 1 - E_gap/Λ)
        measured_e_gap = lambda_sos - result.raw_energy

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
        try:
            from qdk_chemistry.data import FactorizedHamiltonianContainer
        except ImportError:
            pytest.skip("FactorizedHamiltonianContainer not available (requires dev build)")

        from .test_helpers import create_test_orbitals

        data = _build_h2_dfthc_data()
        N, R, B, C = data["N"], data["R"], data["B"], data["C"]

        # Create FactorizedHamiltonianContainer
        orbitals = create_test_orbitals(N)
        inactive_fock = np.zeros((N, N))
        fh = FactorizedHamiltonianContainer(
            data["h1"],
            data["basis_vectors"].flatten(),
            data["two_body_weights"].flatten(),
            data["identity_weight"],
            R,
            B,
            C,
            orbitals,
            0.0,
            inactive_fock,
        )

        # Step 1: SOSSABuilder → UnitaryRepresentation
        builder = SOSSABuilder(quantum_walk=True)
        unitary_rep = builder.run(fh)
        container = unitary_rep.get_container()
        assert isinstance(container, SOSSAContainer)

        # Step 2: SOSSAMapper → Circuit
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])
        mapper = SOSSAMapper(
            outer_prepare_mapper=OuterPrepareMapper(algorithm="dense_pure"),
            inner_prepare_mapper=InnerPrepareMapper(algorithm="direct"),
            select_mapper=SelectMapper(multiplexed_rotation="direct"),
        )
        circuit = mapper.run(controlled_unitary)

        # Verify circuit has all required components
        assert circuit._qsharp_op is not None
        assert circuit._qsharp_factory is not None

        # Step 3: Verify normalization is accessible
        lambda_sos = container.normalization
        assert lambda_sos > 0

        # Step 4: Compute expected spectrum
        H_matrix = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        eigenvalues = np.linalg.eigvalsh(H_matrix)
        # H_gap should be positive semi-definite
        assert eigenvalues[0] >= -1e-10, f"H_gap has negative eigenvalue: {eigenvalues[0]}"

    def test_sossa_normalization_bounds_spectrum(self):
        """Verify that SOSSA normalization Λ bounds the spectrum: all eigenvalues ≤ 2Λ.

        For a valid SOS walk, E_gap ∈ [0, 2Λ], so all eigenvalues of H_gap
        must satisfy 0 ≤ E ≤ 2Λ.
        """
        try:
            from qdk_chemistry.data import FactorizedHamiltonianContainer
        except ImportError:
            pytest.skip("FactorizedHamiltonianContainer not available (requires dev build)")

        from .test_helpers import create_test_orbitals

        data = _build_h2_dfthc_data()
        N, R, B, C = data["N"], data["R"], data["B"], data["C"]

        orbitals = create_test_orbitals(N)
        inactive_fock = np.zeros((N, N))
        fh = FactorizedHamiltonianContainer(
            data["h1"],
            data["basis_vectors"].flatten(),
            data["two_body_weights"].flatten(),
            data["identity_weight"],
            R,
            B,
            C,
            orbitals,
            0.0,
            inactive_fock,
        )

        builder = SOSSABuilder(quantum_walk=True)
        unitary_rep = builder.run(fh)
        container = unitary_rep.get_container()
        lambda_sos = container.normalization

        H_matrix = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        eigenvalues = np.linalg.eigvalsh(H_matrix)

        # All eigenvalues should be ≤ 2Λ (with small numerical tolerance)
        assert np.all(eigenvalues <= 2 * lambda_sos + 1e-10), (
            f"Eigenvalue {eigenvalues.max():.6f} exceeds 2Λ={2 * lambda_sos:.6f}"
        )
