"""Integration tests for iterative QPE with the SOSSA block encoding.

Tests the full pipeline:
    FactorizedHamiltonianContainer → SOSSABuilder → UnitaryRepresentation
    → SOSSAMapper → Circuit → IQPE → energy

Reference: arXiv:2502.15882v1 (Low et al. 2025)
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from math import sqrt

import numpy as np
import pytest

from qdk_chemistry.algorithms.circuit_executor.qdk import QdkSparseStateSimulator
from qdk_chemistry.algorithms.controlled_circuit_mapper.sossa_mapper import SOSSAMapper
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.standard_builder import QdkStandardQpeCircuitBuilder
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.algorithms.qubit_mapper.sossa import SOSSAQubitMapper
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    FactorizedHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSAWalkContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .test_helpers import create_test_orbitals


def _to_sossa_operator(factorized_hamiltonian):
    num_modes = 2 * factorized_hamiltonian.get_num_orbitals()
    hamiltonian = Hamiltonian(factorized_hamiltonian)
    return SOSSAQubitMapper().run(hamiltonian, MajoranaMapping.jordan_wigner(num_modes))


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


def _jordan_wigner_excitations(num_orbitals):
    r"""Build the spin-summed excitation operators ``E_pq`` in the JW encoding.

    Returns a callable ``E(p, q)`` producing the dense matrix of
    :math:`E_{pq} = \sum_\sigma a^\dagger_{p\\sigma} a_{q\\sigma}` on
    ``2 * num_orbitals`` qubits, where ``|1>`` denotes an occupied spin orbital.
    """
    num_spin_orbitals = 2 * num_orbitals
    dim = 2**num_spin_orbitals
    eye2 = np.eye(2, dtype=complex)
    pauli_z = np.diag([1.0, -1.0]).astype(complex)
    sp = np.array([[0, 0], [1, 0]], dtype=complex)  # a† = |1><0|

    def adag(i):
        ops = [pauli_z if j < i else (sp if j == i else eye2) for j in range(num_spin_orbitals)]
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    cache = {}

    def excitation_pq(p, q):
        if (p, q) not in cache:
            mat = np.zeros((dim, dim), dtype=complex)
            for sigma in range(2):
                mat += adag(2 * p + sigma) @ adag(2 * q + sigma).conj().T
            cache[(p, q)] = mat
        return cache[(p, q)]

    return excitation_pq


def _one_body_operator(matrix, excitation_pq, dim):
    """Return ``sum_pq matrix[p, q] E_pq`` as a dense matrix."""
    out = np.zeros((dim, dim), dtype=complex)
    num_orbitals = matrix.shape[0]
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            if abs(matrix[p, q]) > 1e-15:
                out += matrix[p, q] * excitation_pq(p, q)
    return out


def _dfthc_m_matrices(basis_vectors, two_body_weights):
    """Return ``M^{rc}_{pq} = sum_b w_b^{rc} u^r_{bp} u^r_{bq}`` keyed by ``(r, c)``."""
    n_ranks, b_dim, _ = basis_vectors.shape
    n_copies = two_body_weights.shape[2]
    return {
        (r, c): sum(
            two_body_weights[r, b, c] * np.outer(basis_vectors[r, b], basis_vectors[r, b]) for b in range(b_dim)
        )
        for r in range(n_ranks)
        for c in range(n_copies)
    }


def _h1_majorana(h1, basis_vectors, two_body_weights, identity_weight):
    """Replicate ``FactorizedHamiltonianContainer::get_h1_majorana`` (Eq. 38).

    The SOS generators are built from the *normal-ordering corrected* one-body
    matrix, not from the bare ``h1``::

        h1' = h1 + sum_rc [ -1/2 M_rc^2 + tr(M_rc) M_rc - wB^{rc} M_rc ]
    """
    m_matrices = _dfthc_m_matrices(basis_vectors, two_body_weights)
    h1p = np.array(h1, dtype=float, copy=True)
    for (r, c), m_rc in m_matrices.items():
        h1p -= 0.5 * (m_rc @ m_rc)
        h1p += np.trace(m_rc) * m_rc
        h1p -= identity_weight[r, c] * m_rc
    return h1p


def _sos_energy_shift(h1, basis_vectors, two_body_weights, identity_weight):
    """Replicate ``SOSSAQubitMapper`` ``energy_shift`` (Eq. 32), with zero core/BLISS.

    ``E_SOS = -2 sum_r w_-^{(r)} - 1/2 sum_rc |W^{(rc)}|^2``.
    """
    eigenvalues = np.linalg.eigvalsh(_h1_majorana(h1, basis_vectors, two_body_weights, identity_weight))
    negative_sum = float(-np.sum(eigenvalues[eigenvalues < 0.0]))
    w0 = identity_weight - two_body_weights.sum(axis=1)
    return -2.0 * negative_sum - 0.5 * float(np.sum(w0**2))


def _build_dfthc_hamiltonian_matrix(h1, basis_vectors, two_body_weights, identity_weight):
    r"""Build the SOSSA gap Hamiltonian ``H_gap = sum_G G† G`` via Jordan-Wigner.

    The one-body generators come from diagonalizing the *Majorana-corrected*
    one-body matrix (Eq. 38), matching ``SOSSAQubitMapper``, which reads
    ``container.get_h1_majorana()`` rather than the bare ``h1``.

    For eigenvalue :math:`\lambda_k` of that matrix the mapper emits the LCU
    :math:`\sqrt{|\lambda_k|}\\,(X \pm iY)/2` on the rotated spin orbital.
    Since :math:`(X + iY)/2 = |0\rangle\langle 1| = a`, the ``+iY`` branch
    (positive eigenvalues, ``D1``) contributes :math:`G^\dagger G = \lambda_k n_k`
    and the ``-iY`` branch (negative eigenvalues, ``Q1``) contributes
    :math:`|\lambda_k| (2 - n_k)`::

        H_gap = sum_{k: lam_k>0} lam_k n_k + sum_{k: lam_k<0} |lam_k| (2 - n_k)
              + 1/2 sum_rc (W^{rc} I + sum_b w_b^{rc} L_b^r)^2

    where :math:`n_k = \\sum_{pq} V_{pk} V_{qk} E_{pq}` in the eigenbasis of
    :math:`h_1'` and :math:`W^{rc} = wB^{rc} - \\sum_b w_b^{rc}`.

    With this convention ``H_gap`` is positive semidefinite and
    ``H_gap + E_SOS`` equals the physical Hamiltonian exactly; see
    ``test_sos_decomposition_reproduces_physical_hamiltonian``.

    Reference: Eq. 20-21, 29, 32, 38 in :cite:`Low2025`.
    """
    num_orbitals = h1.shape[0]
    n_ranks, b_dim, _ = basis_vectors.shape
    _, n_copies = identity_weight.shape
    dim = 2 ** (2 * num_orbitals)

    excitation_pq = _jordan_wigner_excitations(num_orbitals)
    h1p = _h1_majorana(h1, basis_vectors, two_body_weights, identity_weight)
    eigvals, eigvecs = np.linalg.eigh(h1p)

    # 1) D1/Q1 generator squares.
    h_1b = np.zeros((dim, dim), dtype=complex)
    for k in range(num_orbitals):
        n_k_op = _one_body_operator(np.outer(eigvecs[:, k], eigvecs[:, k]), excitation_pq, dim)
        w_k = eigvals[k]
        if w_k > 0:
            h_1b += w_k * n_k_op
        else:
            h_1b += abs(w_k) * (2.0 * np.eye(dim) - n_k_op)

    # 2) SF squares: ½ Σ_{r,c} (W·I + Σ_b w_b L_b)²
    h_2b = np.zeros((dim, dim), dtype=complex)
    for r in range(n_ranks):
        for c_idx in range(n_copies):
            w_rc = identity_weight[r, c_idx] - np.sum(two_body_weights[r, :, c_idx])
            m_op = w_rc * np.eye(dim, dtype=complex)
            for b in range(b_dim):
                l_b = _one_body_operator(np.outer(basis_vectors[r, b], basis_vectors[r, b]), excitation_pq, dim)
                m_op += two_body_weights[r, b, c_idx] * l_b
            h_2b += 0.5 * (m_op @ m_op)

    return (h_1b + h_2b).real


def _build_physical_hamiltonian_matrix(h1, basis_vectors, two_body_weights):
    """Build the physical DF-THC electronic Hamiltonian, independently of the SOS form.

    ``H = sum_pq h1_pq E_pq + 1/2 sum_pqrs h2_pqrs (E_pq E_rs - delta_qr E_ps)``
    with the tensor-hypercontracted integrals
    ``h2_pqrs = sum_rc M^{rc}_pq M^{rc}_rs``.

    The ``- delta_qr E_ps`` term is the normal-ordering correction, which
    contracts to ``-1/2 sum_rc (M_rc @ M_rc)`` on the one-body matrix.
    """
    num_orbitals = h1.shape[0]
    dim = 2 ** (2 * num_orbitals)
    excitation_pq = _jordan_wigner_excitations(num_orbitals)

    hamiltonian = _one_body_operator(np.asarray(h1, dtype=float), excitation_pq, dim)
    for m_rc in _dfthc_m_matrices(basis_vectors, two_body_weights).values():
        m_op = _one_body_operator(m_rc, excitation_pq, dim)
        hamiltonian += 0.5 * (m_op @ m_op)
        hamiltonian -= 0.5 * _one_body_operator(m_rc @ m_rc, excitation_pq, dim)
    return hamiltonian.real


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


def _run_sossa_iqpe(num_bits, mapper_kwargs=None):
    """Helper: run SOSSA QPE on H2 data and assert measured phase matches expected.

    Uses IQPE with the given mapper configuration, computes k_measured from the
    result phase, and asserts it matches k_expect from exact diagonalization.
    """
    data = _build_h2_dfthc_data()
    n_orb = data["N"]

    # Build reference Hamiltonian matrix and diagonalize
    h_matrix = _build_dfthc_hamiltonian_matrix(
        data["h1"], data["basis_vectors"], data["two_body_weights"], data["identity_weight"]
    )
    gs_energy, gs_vec = _get_ground_state_and_energy(h_matrix, n_orb, nalpha=1, nbeta=1)

    # Create FactorizedHamiltonianContainer
    orbitals = create_test_orbitals(n_orb)
    inactive_fock = np.zeros((n_orb, n_orb))
    fh = FactorizedHamiltonianContainer(
        0.0,
        data["basis_vectors"].flatten(),
        data["two_body_weights"].flatten(),
        data["identity_weight"],
        data["h1"],
        inactive_fock,
        orbitals,
    )

    # Build SOSSA unitary and get normalization
    sossa_op = _to_sossa_operator(fh)
    builder = SOSSABuilder()
    unitary_rep = builder.run(sossa_op)
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
        qubit_hamiltonian=sossa_op,
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


def _run_sossa_standard_qpe(num_bits, mapper_kwargs=None):
    """Helper: run SOSSA standard QPE on H2 data and assert measured phase matches expected.

    Uses StandardPhaseEstimation (QFT-based, multi-ancilla) with the given mapper
    configuration, computes k_measured from the result phase, and asserts it matches
    k_expect from exact diagonalization.

    Uses the circuit builder + executor directly (not StandardPhaseEstimation) to
    properly handle conjugate eigenphases of the SOS walk operator. The walk has
    eigenvalues e^{±iθ}; coherent interference can make the midpoint bin dominant
    for individual bitstrings, but merging conjugate pairs resolves the correct phase.
    """
    data = _build_h2_dfthc_data()
    n_orb = data["N"]

    # Build reference Hamiltonian matrix and diagonalize
    h_matrix = _build_dfthc_hamiltonian_matrix(
        data["h1"], data["basis_vectors"], data["two_body_weights"], data["identity_weight"]
    )
    gs_energy, gs_vec = _get_ground_state_and_energy(h_matrix, n_orb, nalpha=1, nbeta=1)

    # Create FactorizedHamiltonianContainer
    orbitals = create_test_orbitals(n_orb)
    inactive_fock = np.zeros((n_orb, n_orb))
    fh = FactorizedHamiltonianContainer(
        0.0,
        data["basis_vectors"].flatten(),
        data["two_body_weights"].flatten(),
        data["identity_weight"],
        data["h1"],
        inactive_fock,
        orbitals,
    )

    # Build SOSSA unitary and get normalization
    sossa_op = _to_sossa_operator(fh)
    sossa_builder = SOSSABuilder()
    unitary_rep = sossa_builder.run(sossa_op)
    container = unitary_rep.get_container()
    lambda_sos = container.normalization

    # Expected QPE integer (symmetric: min(k, N-k))
    k_expect, _ = _energy_to_k_sos(gs_energy, num_bits, lambda_sos)
    total_states = 2**num_bits
    k_expect_sym = min(k_expect, total_states - k_expect) if k_expect != 0 else 0

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

    # Build mapper kwargs
    mkw = mapper_kwargs or {}
    ref_name = _OUTER_PREP_MAP.get(mkw.get("outer_prepare_algorithm", "dense_pure"), "dense_pure_state")

    # Build standard QPE circuit using the circuit builder directly
    std_builder = QdkStandardQpeCircuitBuilder(
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef(
            "controlled_circuit_mapper",
            "sossa",
            outer_prepare=AlgorithmRef("state_prep", ref_name),
            inner_prepare_algorithm=mkw.get("inner_prepare_algorithm", "direct"),
            select_algorithm=mkw.get("select_algorithm", "direct"),
            coefficient_bit_precision=mkw.get("coefficient_bit_precision", 10),
            rotation_bit_precision=mkw.get("rotation_bit_precision", 10),
        ),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
    )
    circuits = std_builder.run(state_preparation=state_prep, qubit_hamiltonian=sossa_op)

    # Execute with enough shots to resolve conjugate phases
    executor = QdkSparseStateSimulator()
    result = executor.run(circuits[0], shots=200)

    # Merge conjugate phases: SOS walk has eigenvalues e^{±iθ}, so bins k and N-k
    # correspond to the same energy. Sum their counts before picking dominant.
    merged_counts: dict[int, int] = {}
    for bitstring, count in result.bitstring_counts.items():
        k = int(bitstring, 2)
        k_sym = min(k % total_states, (total_states - k) % total_states)
        merged_counts[k_sym] = merged_counts.get(k_sym, 0) + count

    k_measured = max(merged_counts, key=merged_counts.__getitem__)

    assert k_measured == k_expect_sym, (
        f"Expected k={k_expect_sym}, got k={k_measured}, "
        f"merged_counts={dict(sorted(merged_counts.items(), key=lambda x: -x[1])[:5])}, "
        f"gs_energy={gs_energy:.6f}, lambda={lambda_sos:.6f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAQPEIntegration:
    """Integration tests for SOSSA QPE using the full builder → mapper → IQPE pipeline."""

    def test_sossa_qpe_ground_state_energy(self):
        """End-to-end test: SOSSA QPE recovers ground state energy from DFTHC data.

        Uses a small H2-like DFTHC decomposition, runs IQPE with SOSSA block
        encoding, and verifies the measured energy matches exact diagonalization.
        """
        data = _build_h2_dfthc_data()
        n_orb = data["N"]

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
            0.0,
            u_matrices,
            w_matrices,
            wb_matrix,
            h1,
            inactive_fock,
            orbitals,
        )

        # Build SOSSA
        sossa_op = _to_sossa_operator(fh)
        builder = SOSSABuilder()
        unitary_rep = builder.run(sossa_op)
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
            qubit_hamiltonian=sossa_op,
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
        n_orb = data["N"]

        # Create FactorizedHamiltonianContainer
        orbitals = create_test_orbitals(n_orb)
        inactive_fock = np.zeros((n_orb, n_orb))
        fh = FactorizedHamiltonianContainer(
            0.0,
            data["basis_vectors"].flatten(),
            data["two_body_weights"].flatten(),
            data["identity_weight"],
            data["h1"],
            inactive_fock,
            orbitals,
        )

        # Step 1: SOSSABuilder → UnitaryRepresentation
        builder = SOSSABuilder()
        unitary_rep = builder.run(_to_sossa_operator(fh))
        container = unitary_rep.get_container()
        assert isinstance(container, SOSSAWalkContainer)

        # Step 2: SOSSAMapper → Circuit
        mapper = SOSSAMapper()
        mapper.settings().set("outer_prepare", AlgorithmRef("state_prep", "dense_pure_state"))
        mapper.settings().set("inner_prepare_algorithm", "direct")
        mapper.settings().set("select_algorithm", "direct")
        circuit = mapper.run(unitary_rep)

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
        n_orb = data["N"]

        orbitals = create_test_orbitals(n_orb)
        inactive_fock = np.zeros((n_orb, n_orb))
        fh = FactorizedHamiltonianContainer(
            0.0,
            data["basis_vectors"].flatten(),
            data["two_body_weights"].flatten(),
            data["identity_weight"],
            data["h1"],
            inactive_fock,
            orbitals,
        )

        builder = SOSSABuilder()
        unitary_rep = builder.run(_to_sossa_operator(fh))
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

    def test_sos_decomposition_is_positive_semidefinite(self):
        """H_gap = Σ_G G†G must be PSD, since it is a sum of operator squares.

        This is the property the spectrum amplification relies on: the walk
        eigenphase encodes E_gap ∈ [0, 2Λ], so a negative eigenvalue would make
        the ``arccos`` decoding ill-defined (:cite:`Low2025`, Eq. 11, 20-21).
        """
        data = _build_h2_dfthc_data()
        h_gap = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        assert np.linalg.eigvalsh(h_gap).min() >= -1e-12

    def test_sos_decomposition_reproduces_physical_hamiltonian(self):
        """H_gap + E_SOS must equal the physical Hamiltonian, eigenvalue by eigenvalue.

        This is the correctness statement that makes SOSSA a chemistry
        algorithm rather than a spectral trick: the sum-of-squares generators
        are constructed so that

            H_physical = Σ_G G†G + E_SOS · I

        with the energy shift of :cite:`Low2025` Eq. 32,
        ``E_SOS = -2 Σ_r w_-^{(r)} - ½ Σ_rc |W^{(rc)}|²``, evaluated on the
        Majorana-corrected one-body matrix of Eq. 38.  Recovering the physical
        energy from a phase estimate therefore requires adding ``E_SOS`` back.

        The whole spectrum is compared (not just the ground state) so that a
        wrong particle/hole convention or a missing normal-ordering correction
        cannot hide behind an accidental agreement at the band edge.
        """
        data = _build_h2_dfthc_data()
        h_gap = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        h_physical = _build_physical_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
        )
        energy_shift = _sos_energy_shift(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )

        np.testing.assert_allclose(
            np.linalg.eigvalsh(h_gap) + energy_shift,
            np.linalg.eigvalsh(h_physical),
            atol=1e-10,
        )

    def test_energy_shift_matches_qubit_mapper(self):
        """The mapper's ``energy_shift`` must equal Eq. 32 evaluated on h1_majorana."""
        data = _build_h2_dfthc_data()
        n_orb = data["N"]
        orbitals = create_test_orbitals(n_orb)
        fh = FactorizedHamiltonianContainer(
            0.0,
            data["basis_vectors"].flatten(),
            data["two_body_weights"].flatten(),
            data["identity_weight"],
            data["h1"],
            np.zeros((n_orb, n_orb)),
            orbitals,
        )
        container = _to_sossa_operator(fh).get_container()

        expected = (
            fh.get_core_energy()
            + fh.get_bliss_shift()
            + _sos_energy_shift(
                data["h1"],
                data["basis_vectors"],
                data["two_body_weights"],
                data["identity_weight"],
            )
        )
        assert container.energy_shift == pytest.approx(expected, abs=1e-12)

    def test_ground_state_energy_recovered_from_walk_eigenphase(self):
        """Decoding a walk eigenphase must return the exact physical ground energy.

        Exercises the full decoding chain the QPE driver uses: the walk
        eigenvalue associated with gap energy ``E_gap`` is ``e^{±2πiφ}`` with
        ``E_gap = Λ(1 + cos 2πφ)`` (:cite:`Low2025`, Eq. 11), and
        ``SOSSAWalkContainer.eigenvalue_from_phase`` adds ``E_SOS`` back.
        Feeding it the phase fraction of the exact gap ground state must
        reproduce the exact ground-state energy of the *physical* Hamiltonian,
        which is what makes the energy shift meaningful.
        """
        data = _build_h2_dfthc_data()
        n_orb = data["N"]
        orbitals = create_test_orbitals(n_orb)
        fh = FactorizedHamiltonianContainer(
            0.0,
            data["basis_vectors"].flatten(),
            data["two_body_weights"].flatten(),
            data["identity_weight"],
            data["h1"],
            np.zeros((n_orb, n_orb)),
            orbitals,
        )
        container = SOSSABuilder().run(_to_sossa_operator(fh)).get_container()

        h_gap = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        gap_energy, _ = _get_ground_state_and_energy(h_gap, n_orb, nalpha=1, nbeta=1)

        # Gap energy -> walk phase fraction, then back through the container decoder.
        phase_fraction = math.acos(np.clip(gap_energy / container.normalization - 1.0, -1.0, 1.0)) / (2 * math.pi)
        recovered = container.eigenvalue_from_phase(phase_fraction)

        h_physical = _build_physical_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
        )
        exact_energy, _ = _get_ground_state_and_energy(h_physical, n_orb, nalpha=1, nbeta=1)
        assert recovered == pytest.approx(exact_energy, abs=1e-10)

    @pytest.mark.parametrize("num_bits", [3, 5])
    def test_sossa_qpe(self, num_bits):
        """QPE with direct (non-alias) config should match expected phase index."""
        _run_sossa_iqpe(num_bits)

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
        _run_sossa_iqpe(num_bits=3, mapper_kwargs=mapper_overrides)

    @pytest.mark.parametrize("num_bits", [3, 5])
    def test_sossa_standard_qpe(self, num_bits):
        """Standard QPE with direct (non-alias) config should match expected phase index."""
        _run_sossa_standard_qpe(num_bits)
