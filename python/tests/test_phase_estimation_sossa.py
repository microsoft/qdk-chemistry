"""Integration tests for phase estimation with the SOSSA block encoding.

Tests the full pipeline:
    FactorizedHamiltonianContainer → SOSSABuilder → UnitaryRepresentation
    → SOSSAMapper → Circuit → unary-iteration QPE → energy

and standard and iterative QPE through ControlledSOSSAMapper.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import math
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.algorithms.phase_estimation.standard_phase_estimation import StandardPhaseEstimation
from qdk_chemistry.algorithms.phase_estimation.unary_phase_estimation import UnaryPhaseEstimation
from qdk_chemistry.algorithms.state_preparation.qrom_state_prep import QROMStatePreparation
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    Configuration,
    FactorizedHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .helpers import (
    create_random_factorized_hamiltonian,
    create_sparse_wavefunction,
    create_test_orbitals,
    factorized_hamiltonian_to_sossa_operator,
)
from .reference_tolerances import ci_energy_tolerance

# ═══════════════════════════════════════════════════════════════════════════════
# Test Hamiltonian construction (small DFTHC-like H2 data)
# ═══════════════════════════════════════════════════════════════════════════════


#: The factorized H2 shipped with the examples, which the SOSSA notebook also loads.
H2_DFTHC_EXAMPLE = Path(__file__).resolve().parent / "test_data" / "h2_dfthc_r2_b2_c1.hamiltonian.json"


def _build_h2_dfthc_data():
    """Load the shipped factorized H2 and return its tensors.

    Returns a dict with all tensor data needed for SOSSA.
    """
    container = json.loads(H2_DFTHC_EXAMPLE.read_text())["container"]
    n_orb = int(container["orbitals"]["num_orbitals"])
    n_ranks = int(container["num_ranks"])
    n_bases = int(container["num_bases"])
    n_copies = int(container["num_copies"])

    return {
        "h1": np.asarray(container["one_body_integrals"], dtype=float).reshape(n_orb, n_orb),
        # u_matrices is [R, B, N] and w_matrices [R, B, C], both stored flat.
        "basis_vectors": np.asarray(container["u_matrices"], dtype=float).reshape(n_ranks, n_bases, n_orb),
        "two_body_weights": np.asarray(container["w_matrices"], dtype=float).reshape(n_ranks, n_bases, n_copies),
        "identity_weight": np.asarray(container["wb_matrix"], dtype=float).reshape(n_ranks, n_copies),
        "core_energy": float(container["core_energy"]),
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
    """Replicate ``FactorizedHamiltonianContainer::get_h1_prime`` (Eq. 36).

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
    """Replicate ``SumOfSquaresQubitMapper`` ``energy_shift`` (Eq. 30), with zero core/BLISS.

    ``E_SOS = -2 sum_r w_-^{(r)} - 1/2 sum_rc |W^{(rc)}|^2``.
    """
    eigenvalues = np.linalg.eigvalsh(_h1_majorana(h1, basis_vectors, two_body_weights, identity_weight))
    negative_sum = float(-np.sum(eigenvalues[eigenvalues < 0.0]))
    w0 = identity_weight - two_body_weights.sum(axis=1)
    return -2.0 * negative_sum - 0.5 * float(np.sum(w0**2))


def _build_dfthc_hamiltonian_matrix(h1, basis_vectors, two_body_weights, identity_weight):
    """Build the SOSSA gap Hamiltonian ``H_gap = sum_G G† G`` via Jordan-Wigner."""
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
    """Build the physical DFTHC electronic Hamiltonian, independently of the SOS form.

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
    gs_idx = sector_indices[0]
    gs_energy = eigenvalues[gs_idx]
    return gs_energy, _to_qsharp_ordering(eigenvectors[:, gs_idx], num_orbitals)


def _to_qsharp_ordering(state_vector, num_orbitals):
    """Reorder a Python-Kron state vector into Q# order, with fermionic signs."""
    perm = _python_to_qsharp_permutation(num_orbitals)
    signs = _python_to_qsharp_sign(num_orbitals)
    reordered = np.zeros(len(state_vector))
    for i in range(len(state_vector)):
        reordered[perm[i]] = signs[i] * state_vector[i]
    return reordered


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


def _python_to_qsharp_sign(num_orbitals):
    """Fermionic sign of the mode reordering ``_python_to_qsharp_permutation`` performs.

    That permutation moves the interleaved ``(p, sigma)`` modes into spin-blocked order.
    Reordering fermionic modes reorders the creation operators of a determinant, so each
    basis state also picks up the parity of the permutation restricted to its occupied
    modes -- which relabelling the index on its own does not carry. Omitting it leaves a
    diagonal similarity transform between the two conventions: spectra agree, individual
    eigenvectors and matrix elements do not.
    """
    n_qubits = 2 * num_orbitals
    signs = np.ones(2**n_qubits)
    for i in range(2**n_qubits):
        modes = sorted(n_qubits - 1 - b for b in range(n_qubits) if (i >> b) & 1)
        images = [(j % 2) * num_orbitals + j // 2 for j in modes]
        signs[i] = (-1.0) ** sum(
            1 for a in range(len(images)) for b in range(a + 1, len(images)) if images[a] > images[b]
        )
    return signs


# ═══════════════════════════════════════════════════════════════════════════════
# SOSSA QPE helper
# ═══════════════════════════════════════════════════════════════════════════════


def _direct_state_prep(state_vector):
    """Prepare a dense real state vector given in Q# order."""
    num_system_qubits = len(state_vector).bit_length() - 1
    state_prep_params = {
        "rowMap": list(range(num_system_qubits - 1, -1, -1)),
        "stateVector": np.asarray(state_vector, dtype=float).tolist(),
        "expansionOps": [],
        "numQubits": num_system_qubits,
    }
    return Circuit(
        qsharp_factory=QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        ),
        qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params),
    )


def _sossa_circuit_mapper_ref(
    *,
    outer_prepare_algorithm: str = "dense_pure_state",
    inner_prepare_algorithm: str = "direct",
    select_algorithm: str = "direct",
    coefficient_bit_precision: int = 10,
    rotation_bit_precision: int = 10,
) -> AlgorithmRef:
    """Return an AlgorithmRef for the SOSSA circuit mapper."""
    return AlgorithmRef(
        "circuit_mapper",
        "sossa",
        outer_prepare_algorithm=AlgorithmRef("state_prep", outer_prepare_algorithm),
        inner_prepare_algorithm=inner_prepare_algorithm,
        select_algorithm=select_algorithm,
        coefficient_bit_precision=coefficient_bit_precision,
        rotation_bit_precision=rotation_bit_precision,
    )


def _energy_to_qpe_phase(energy_gap, lambda_sos):
    r"""Convert energy gap to QPE phase for the SOS walk operator.

    For SOS walk: cos(2πφ) = E_gap / Λ - 1

    Args:
        energy_gap: Energy measured from the SOS shift, expected in :math:`[0, 2\Lambda]`.
        lambda_sos: The block encoding normalization :math:`\Lambda`.

    Returns:
        The corresponding phase fraction in :math:`[0, 1/2]`.

    Raises:
        ValueError: If the energy falls outside the band the walk can represent. The
            previous unconditional clamp silently mapped a NaN to phase 0 and pulled an
            out-of-band spectrum back to an edge, so a block encoding with the wrong
            normalization still produced a plausible-looking phase.

    """
    cos_val = energy_gap / lambda_sos - 1.0
    if not -1.0 - 1e-9 <= cos_val <= 1.0 + 1e-9:
        raise ValueError(
            f"Energy gap {energy_gap!r} is outside the SOSSA walk band [0, {2 * lambda_sos!r}]: "
            f"cos(2*pi*phi) would have to be {cos_val!r}."
        )
    return math.acos(max(-1.0, min(1.0, cos_val))) / (2 * math.pi)


def _run_sossa_unary_qpe(num_queries, mapper_kwargs=None, shots=100, seed=20250815):
    """Run unary-iteration QPE on the H2 data and assert the measured bin is the predicted one.

    ``num_queries`` must be one less than a power of two, so the ``num_queries + 1``
    reflection slots exactly fill the phase register and a bin is worth
    ``1 / (num_queries + 1)``.

    The schedule applies ``W^(num_queries - 2t)``, so the register encodes *twice* the walk
    phase; ``QpeResult.phase_fraction`` is that doubled phase, folded into ``[0, 1/2]`` by
    the branch resolution in ``_post_process_phase_estimation``. The exact eigenvalue is
    folded the same way before comparing, which puts both on one scale and pins the
    schedule, the inverse QFT and the bitstring decoding together.

    Args:
        num_queries: Number of walk blocks the schedule applies.
        mapper_kwargs: Overrides for :func:`_sossa_circuit_mapper_ref`.
        shots: Number of shots to sample.
        seed: Simulator seed, fixed so the measured bin is reproducible.

    Returns:
        The QPE result, the exact ground-state energy of the gap Hamiltonian, the exact
        physical ground-state energy, and the SOSSA walk container, so a caller can also
        assert on the decoded energy.

    """
    num_bins = num_queries + 1
    assert num_bins & (num_bins - 1) == 0, "num_queries must be one less than a power of two"

    data = _build_h2_dfthc_data()
    n_orb = data["N"]

    h_matrix = _build_dfthc_hamiltonian_matrix(
        data["h1"],
        data["basis_vectors"],
        data["two_body_weights"],
        data["identity_weight"],
    )
    gs_energy, gs_vec = _get_ground_state_and_energy(h_matrix, n_orb, nalpha=1, nbeta=1)

    # The same state measured against the physical Hamiltonian, built straight from h1 and
    # the DFTHC factors. It never references the SOS identity weight, so it is an oracle
    # for the decoded energy that is independent of the container's ``energy_shift`` --
    # unlike ``gs_energy``, which is the quantity ``energy_shift`` is defined to offset.
    physical_energy, _ = _get_ground_state_and_energy(
        _build_physical_hamiltonian_matrix(data["h1"], data["basis_vectors"], data["two_body_weights"]),
        n_orb,
        nalpha=1,
        nbeta=1,
    )

    fh = FactorizedHamiltonianContainer(
        one_body_integrals=data["h1"],
        u_matrices=data["basis_vectors"].flatten(),
        w_matrices=data["two_body_weights"].flatten(),
        wb_matrix=data["identity_weight"],
        orbitals=create_test_orbitals(n_orb),
        core_energy=0.0,
        inactive_fock_matrix=np.zeros((n_orb, n_orb)),
    )
    sossa_op = factorized_hamiltonian_to_sossa_operator(fh)
    container = SOSSABuilder().run(sossa_op).get_container()
    state_prep = _direct_state_prep(gs_vec.real)

    qpe = UnaryPhaseEstimation(shots=shots)
    qpe.settings().set(
        "qpe_circuit_builder",
        AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_unary",
            num_queries=num_queries,
            circuit_mapper=_sossa_circuit_mapper_ref(**(mapper_kwargs or {})),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
        ),
    )
    qpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator", seed=seed),
    )
    result = qpe.run(qubit_hamiltonian=sossa_op, state_preparation=state_prep)

    exact_phase = _energy_to_qpe_phase(gs_energy, container.normalization)
    exact_doubled_phase = 2.0 * min(exact_phase, 0.5 - exact_phase)
    bin_error = abs(result.phase_fraction - exact_doubled_phase) * num_bins
    assert bin_error <= 1.0, (
        f"Measured 2*phi={result.phase_fraction:.6f} is {bin_error:.2f} bins away from the "
        f"exact {exact_doubled_phase:.6f} (num_bins={num_bins})."
    )
    return result, gs_energy, physical_energy, container


#: Block-encoding normalization of :func:`_exact_walk_phase_problem`, which its ``b`` leaves unchanged.
_EXACT_PHASE_LAMBDA = 1.69

#: Phase-gradient width that holds every rotation of :func:`_exact_walk_phase_problem` exactly.
_EXACT_PHASE_ROTATION_BITS = 4


def _exact_walk_phase_problem(walk_phase):
    """Build a two-orbital SOSSA problem with an eigenstate at ``walk_phase``, exact at 4 rotation bits."""
    a, w1, w2 = -1.0, 0.5, 0.3
    # Lambda does not depend on b, so b alone moves the bonding pair's energy 2(a + b) + w1^2 to walk_phase.
    b = 0.5 * (_EXACT_PHASE_LAMBDA * (math.cos(2.0 * math.pi * walk_phase) - 1.0) - w1**2) - a
    s = 1.0 / math.sqrt(2.0)
    h1 = np.array([[a, b], [b, a]])
    basis_vectors = np.array([[[s, s], [s, -s]]])
    two_body_weights = np.array([[[w1], [w2]]])
    identity_weight = np.zeros((1, 1))
    n_orb = 2

    fh = FactorizedHamiltonianContainer(
        one_body_integrals=h1,
        u_matrices=basis_vectors.flatten(),
        w_matrices=two_body_weights.flatten(),
        wb_matrix=identity_weight,
        orbitals=create_test_orbitals(n_orb),
        core_energy=0.0,
        inactive_fock_matrix=np.zeros((n_orb, n_orb)),
    )
    operator = factorized_hamiltonian_to_sossa_operator(fh)
    container = SOSSABuilder().run(operator).get_container()
    assert container.normalization == pytest.approx(_EXACT_PHASE_LAMBDA)
    for angles in (container.select.one_body_rotation_angles, container.select.two_body_rotation_angles):
        steps = np.asarray(angles) / (math.pi / 4.0)
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-12)

    gap_energies, gap_states = np.linalg.eigh(
        _build_dfthc_hamiltonian_matrix(h1, basis_vectors, two_body_weights, identity_weight)
    )
    distances = np.abs(gap_energies - container.normalization * (1.0 + math.cos(2.0 * math.pi * walk_phase)))
    index = int(np.argmin(distances))
    assert distances[index] < 1e-9, "no eigenstate at the target walk phase"
    assert np.partition(distances, 1)[1] > 0.1, "the target eigenstate must be non-degenerate"
    assert _energy_to_qpe_phase(gap_energies[index], container.normalization) == pytest.approx(walk_phase)

    state = gap_states[:, index]
    physical = _build_physical_hamiltonian_matrix(h1, basis_vectors, two_body_weights)
    return {
        "operator": operator,
        "container": container,
        "state": _to_qsharp_ordering(state, n_orb),
        "energy": float(state @ physical @ state),
    }


def _exact_phase_state_prep(state_vector, consumes_gradient):
    """Prepare ``state_vector`` directly, or with a QROM that consumes the shared gradient."""
    if not consumes_gradient:
        return _direct_state_prep(state_vector)
    support = np.flatnonzero(np.abs(state_vector) > 1e-12)
    wavefunction = create_sparse_wavefunction(
        len(state_vector).bit_length() - 1, support.tolist(), np.asarray(state_vector)[support].tolist()
    )
    state_prep = QROMStatePreparation(
        rotation_bit_precision=_EXACT_PHASE_ROTATION_BITS, allocate_phase_gradient=False
    ).run(wavefunction)
    assert state_prep.metadata.num_phase_gradient_ancillas == _EXACT_PHASE_ROTATION_BITS
    return state_prep


def _controlled_walk_builder_settings(num_bits, select_algorithm):
    """QPE builder settings for the controlled SOSSA walk."""
    return {
        "num_bits": num_bits,
        "unitary_builder": AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
        "controlled_circuit_mapper": AlgorithmRef(
            "controlled_circuit_mapper",
            "sossa",
            outer_prepare_algorithm=AlgorithmRef("state_prep", "dense_pure_state"),
            inner_prepare_algorithm="direct",
            select_algorithm=select_algorithm,
            rotation_bit_precision=_EXACT_PHASE_ROTATION_BITS,
        ),
    }


def _run_controlled_walk_qpe(qpe, builder_name, problem, *, num_bits, select_algorithm, prepare_consumes_gradient):
    """Run ``qpe`` on ``problem`` through the controlled SOSSA walk."""
    qpe.settings().set(
        "qpe_circuit_builder",
        AlgorithmRef(
            "qpe_circuit_builder", builder_name, **_controlled_walk_builder_settings(num_bits, select_algorithm)
        ),
    )
    qpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator", seed=7))
    return qpe.run(
        qubit_hamiltonian=problem["operator"],
        state_preparation=_exact_phase_state_prep(problem["state"], prepare_consumes_gradient),
    )


def _walk_qpe_circuit_counts(
    builder_name, problem, *, num_bits, select_algorithm, prepare_consumes_gradient, shots, **builder_settings
):
    """Run the QPE circuit directly and return all its MSB-first bitstring counts."""
    builder = create(
        "qpe_circuit_builder",
        builder_name,
        **_controlled_walk_builder_settings(num_bits, select_algorithm),
        **builder_settings,
    )
    (circuit,) = builder.run(
        state_preparation=_exact_phase_state_prep(problem["state"], prepare_consumes_gradient),
        qubit_hamiltonian=problem["operator"],
    )
    executor = create("circuit_executor", "qdk_sparse_state_simulator", seed=7)
    return executor.run(circuit, shots=shots).bitstring_counts


#: ``(select_algorithm, prepare_consumes_gradient)``; a QROM prep needs the gradient SELECT.
_CONTROLLED_WALK_CASES = [
    pytest.param("qrom_phase_gradient", False, id="gradient_walk"),
    pytest.param("qrom_phase_gradient", True, id="gradient_walk_and_prep"),
    pytest.param("direct", False, id="gradient_free_walk"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAQPEIntegration:
    """Integration tests for SOSSA QPE using the full builder → mapper → unary QPE pipeline."""

    def test_sos_decomposition_reproduces_physical_hamiltonian(self):
        """H_gap + E_SOS must equal the physical Hamiltonian, eigenvalue by eigenvalue.

        This is the correctness statement that makes SOSSA a chemistry
        algorithm rather than a spectral trick: the sum-of-squares generators
        are constructed so that

            H_physical = Σ_G G†G + E_SOS · I

        with the energy shift of :cite:`Low2025` Eq. 30,
        ``E_SOS = -2 Σ_r w_-^{(r)} - ½ Σ_rc |W^{(rc)}|²``, evaluated on the
        Majorana-corrected one-body matrix of Eq. 36.  Recovering the physical
        energy from a phase estimate therefore requires adding ``E_SOS`` back.

        The whole spectrum is compared (not just the ground state) so that a
        wrong particle/hole convention or a missing normal-ordering correction
        cannot hide behind an accidental agreement at the band edge.

        The ground state is then cross-checked against a CASCI solve of the
        *container's own* integrals.  ``macis_cas`` reaches the factorized
        container through ``get_two_body_integrals``, which reconstructs
        ``h2_pqrs = sum_rc s_r M^rc_pq M^rc_rs`` in C++.  That closes the loop on
        the NumPy oracle: a convention error shared by
        ``_build_dfthc_hamiltonian_matrix`` and
        ``_build_physical_hamiltonian_matrix`` cancels in the spectrum
        comparison above, but not against the shipped reconstruction.
        """
        data = _build_h2_dfthc_data()
        n_orb = data["N"]
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
            rtol=0.0,
            atol=1e-10,
        )

        fh = FactorizedHamiltonianContainer(
            one_body_integrals=data["h1"],
            u_matrices=data["basis_vectors"].flatten(),
            w_matrices=data["two_body_weights"].flatten(),
            wb_matrix=data["identity_weight"],
            orbitals=create_test_orbitals(n_orb),
            core_energy=0.0,
            inactive_fock_matrix=np.zeros((n_orb, n_orb)),
        )
        # ``Hamiltonian`` takes ownership of the C++ container, so read the scalar
        # offset off ``fh`` first: ``macis_cas`` returns an energy that includes it.
        core_energy = fh.get_core_energy()
        casci_energy, _ = create("multi_configuration_calculator", "macis_cas").run(Hamiltonian(fh), 1, 1)

        gap_energy, _ = _get_ground_state_and_energy(h_gap, n_orb, nalpha=1, nbeta=1)
        assert gap_energy + energy_shift == pytest.approx(casci_energy - core_energy, abs=ci_energy_tolerance)

    def test_ground_state_energy_recovered_from_walk_eigenphase(self):
        """Decoding a walk eigenphase must return the exact physical ground energy.

        Exercises the full decoding chain the QPE driver uses: the walk
        eigenvalue associated with gap energy ``E_gap`` is ``e^{±2πiφ}`` with
        ``E_gap = Λ(1 + cos 2πφ)`` (:cite:`Low2025`, Eq. 11), and
        ``SOSSABlockEncodingContainer.eigenvalue_from_phase`` adds ``E_SOS`` back.
        Feeding it the phase fraction of the exact gap ground state must
        reproduce the exact ground-state energy of the *physical* Hamiltonian,
        which is what makes the energy shift meaningful.
        """
        data = _build_h2_dfthc_data()
        n_orb = data["N"]
        orbitals = create_test_orbitals(n_orb)
        fh = FactorizedHamiltonianContainer(
            one_body_integrals=data["h1"],
            u_matrices=data["basis_vectors"].flatten(),
            w_matrices=data["two_body_weights"].flatten(),
            wb_matrix=data["identity_weight"],
            orbitals=orbitals,
            core_energy=0.0,
            inactive_fock_matrix=np.zeros((n_orb, n_orb)),
        )
        container = SOSSABuilder().run(factorized_hamiltonian_to_sossa_operator(fh)).get_container()

        h_gap = _build_dfthc_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
            data["identity_weight"],
        )
        gap_energy, _ = _get_ground_state_and_energy(h_gap, n_orb, nalpha=1, nbeta=1)

        # Gap energy -> walk phase fraction, then back through the container decoder.
        phase_fraction = _energy_to_qpe_phase(gap_energy, container.normalization)
        recovered = container.eigenvalue_from_phase(phase_fraction)

        h_physical = _build_physical_hamiltonian_matrix(
            data["h1"],
            data["basis_vectors"],
            data["two_body_weights"],
        )
        exact_energy, _ = _get_ground_state_and_energy(h_physical, n_orb, nalpha=1, nbeta=1)
        assert recovered == pytest.approx(exact_energy, abs=1e-10)

    def test_lambda_eff_is_the_slope_of_the_walk_energy_decoder(self):
        r"""``lambda_eff`` must equal ``sqrt(E_gap (2L - E_gap))`` *and* the decoder slope.

        Phase estimation on this walk resolves an energy to :math:`\sigma` using
        :math:`\pi\lambda_{\text{eff}}/(2\sigma)` queries (:cite:`Low2025`, Eq. (11)), so
        :math:`\lambda_{\text{eff}}` -- not :math:`\Lambda` -- is the quantity that sizes
        a query schedule. It is checked here two independent ways:

        1. Against the closed form evaluated on an ``E_gap`` obtained by diagonalizing
           ``H_gap`` in NumPy, while the builder itself only ever sees the *physical*
           ground-state energy and subtracts ``energy_shift``. The two routes to
           ``E_gap`` share no code, so this simultaneously pins the shift convention:
           ``reference_ground_state_energy`` must be a total energy, core included.
        2. Against the derivative of :meth:`SOSSABlockEncodingContainer.eigenvalue_from_phase`,
           which is what actually converts a phase estimate into an energy. Because
           :math:`E(\varphi) = \Lambda(1 + \cos 2\pi\varphi) + E_{\text{SOS}}`,
           :math:`|\mathrm{d}E/\mathrm{d}\varphi| = 2\pi\lambda_{\text{eff}}` at the
           ground-state phase; the paper differentiates this same inverse in its
           Appendix G. This control never evaluates the closed form, so it would catch a
           formula that agreed with check 1 only because both were wrong the same way.

        The asymptotic :math:`\sqrt{2\Lambda E_{\text{gap}}}` quoted in the paper's
        abstract, and :math:`\Lambda` itself, are asserted *not* to match, so the pin is
        not passing for want of discriminating power.
        """
        data = _build_h2_dfthc_data()
        n_orb = data["N"]
        # The shipped file's core energy is used rather than 0 so that the "total energy"
        # half of the contract is actually under test: with a zero core, passing an
        # active-space-only energy would be indistinguishable from passing a total one.
        fh = FactorizedHamiltonianContainer(
            one_body_integrals=data["h1"],
            u_matrices=data["basis_vectors"].flatten(),
            w_matrices=data["two_body_weights"].flatten(),
            wb_matrix=data["identity_weight"],
            orbitals=create_test_orbitals(n_orb),
            core_energy=data["core_energy"],
            inactive_fock_matrix=np.zeros((n_orb, n_orb)),
        )
        # ``factorized_hamiltonian_to_sossa_operator`` consumes the container, so read the offset off ``fh`` first.
        core_energy = fh.get_core_energy()
        assert core_energy != 0.0, "a zero core energy would make the convention check below vacuous"

        # The builder's only extra input: the physical ground-state energy, core included.
        h_physical = _build_physical_hamiltonian_matrix(data["h1"], data["basis_vectors"], data["two_body_weights"])
        physical_energy, _ = _get_ground_state_and_energy(h_physical, n_orb, nalpha=1, nbeta=1)
        reference_ground_state_energy = physical_energy + core_energy

        operator = factorized_hamiltonian_to_sossa_operator(fh)
        container = (
            SOSSABuilder(reference_ground_state_energy=reference_ground_state_energy).run(operator).get_container()
        )
        lambda_eff = container.lambda_eff

        # (1) Reference from an independently diagonalized H_gap.
        h_gap = _build_dfthc_hamiltonian_matrix(
            data["h1"], data["basis_vectors"], data["two_body_weights"], data["identity_weight"]
        )
        energy_gap, _ = _get_ground_state_and_energy(h_gap, n_orb, nalpha=1, nbeta=1)
        two_lambda = 2.0 * container.normalization
        expected = math.sqrt(energy_gap * (two_lambda - energy_gap))
        assert lambda_eff == pytest.approx(expected, abs=1e-12)

        # (2) Slope of the decoder, evaluated without the closed form.
        phase = _energy_to_qpe_phase(energy_gap, container.normalization)
        step = 1e-6
        slope = (container.eigenvalue_from_phase(phase + step) - container.eigenvalue_from_phase(phase - step)) / (
            2 * step
        )
        assert abs(slope) == pytest.approx(2 * math.pi * lambda_eff, rel=1e-8)

        # Discriminating power: the wrong-but-plausible forms are far outside the tolerance.
        assert lambda_eff < container.normalization
        assert lambda_eff != pytest.approx(math.sqrt(two_lambda * energy_gap), abs=1e-3)
        assert lambda_eff != pytest.approx(container.normalization, abs=1e-3)

        # The core-energy convention is load-bearing, not decorative: the active-space
        # energy sits a full core below the shift, so it must be rejected outright.
        with pytest.raises(ValueError, match="outside the representable window"):
            SOSSABuilder(reference_ground_state_energy=physical_energy).run(operator)

    @pytest.mark.parametrize(
        "mapper_overrides",
        [
            {},
            pytest.param(
                {
                    "outer_prepare_algorithm": "alias_sampling",
                    "coefficient_bit_precision": 4,
                },
                marks=pytest.mark.slow,
            ),
            pytest.param(
                {
                    "inner_prepare_algorithm": "controlled_alias_sampling",
                    "coefficient_bit_precision": 4,
                },
                marks=pytest.mark.slow,
            ),
        ],
        ids=["direct", "alias_outer", "alias_inner"],
    )
    def test_sossa_qpe_features(self, mapper_overrides):
        """QPE over 8 phase bins of the H2 data, with the direct backends and each alias variant."""
        _run_sossa_unary_qpe(num_queries=7, mapper_kwargs=mapper_overrides)

    @pytest.mark.slow
    def test_unary_qpe_recovers_the_h2_ground_state_energy(self):
        """Recover the H2 ground-state energy end-to-end with unary-iteration QPE.

        Exercises the full production path: SOSSABuilder -> SOSSAMapper ->
        ``MakeUnaryQPECircuit`` -> sparse simulator -> bitstring decoding.
        """
        num_queries = 31
        result, gs_energy, physical_energy, container = _run_sossa_unary_qpe(num_queries)

        # _run_sossa_unary_qpe bounds the *doubled* phase to one bin, and raw_energy is
        # decoded from the canonical phase, which is half of it -- so the canonical phase
        # carries half a bin of uncertainty.
        half_bin = 1.0 / (2.0 * (num_queries + 1))
        exact_phase = _energy_to_qpe_phase(gs_energy, container.normalization)
        # E_gap(phi) = 2*Lambda*cos^2(pi*phi) is monotone across the canonical range
        # [0, 1/2], so the widest energy excursion over the window sits at an edge.
        edges = [min(0.5, max(0.0, exact_phase + sign * half_bin)) for sign in (-1.0, 1.0)]
        tol = max(abs(container.eigenvalue_from_phase(p) - container.metadata.energy_shift - gs_energy) for p in edges)

        assert abs(result.raw_energy - physical_energy) <= tol, (
            f"Energy mismatch: measured={result.raw_energy:.6f}, expected={physical_energy:.6f}, tol={tol:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Standard and iterative QPE through the controlled walk
# ═══════════════════════════════════════════════════════════════════════════════


class TestControlledSOSSAMapper:
    """Tests for the ``sossa`` controlled circuit mapper."""

    @staticmethod
    def _operator():
        factorized = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=1, num_bases=2, num_copies=1)
        return factorized_hamiltonian_to_sossa_operator(factorized)

    def test_only_the_sossa_controlled_mapper_accepts_the_block_encoding(self):
        """Other controlled mappers reject the SOSSA container."""
        unitary = SOSSABuilder().run(self._operator())

        mappers = set(available("controlled_circuit_mapper"))
        assert "sossa" in mappers
        assert mappers - {"sossa"}, "expected other registered controlled circuit mappers"
        for name in mappers - {"sossa"}:
            with pytest.raises(ValueError, match="not supported"):
                create("controlled_circuit_mapper", name).run(unitary)

    @pytest.mark.parametrize("select_algorithm", ["direct", "qrom_phase_gradient"])
    def test_walk_acts_on_the_block_encoding_register(self, select_algorithm):
        """Controlled-W^p declares the block encoding's width and gradient, independent of p."""
        operator = self._operator()
        block = create("circuit_mapper", "sossa", select_algorithm=select_algorithm).run(SOSSABuilder().run(operator))
        walk = create("controlled_circuit_mapper", "sossa", select_algorithm=select_algorithm).run(
            SOSSABuilder(power=4).run(operator)
        )

        assert walk.num_qubits == block.num_qubits
        gradient = walk.metadata.num_phase_gradient_ancillas
        assert gradient == block.metadata.num_phase_gradient_ancillas
        expected = (
            create("controlled_circuit_mapper", "sossa").settings().get("rotation_bit_precision")
            if select_algorithm == "qrom_phase_gradient"
            else 0
        )
        assert gradient == expected

    @pytest.mark.parametrize("control_indices", [[1], [0, 1]])
    def test_rejects_controls_other_than_one_qubit_at_index_zero(self, control_indices):
        """The walk takes one control qubit, at index 0."""
        unitary = SOSSABuilder().run(self._operator())
        mapper = create("controlled_circuit_mapper", "sossa", control_indices=control_indices)
        with pytest.raises(ValueError, match="single control qubit at index 0"):
            mapper.run(unitary)


class TestSOSSAControlledWalkQPE:
    """Standard and iterative QPE through the controlled SOSSA walk; shots land on phi or 1 - phi."""

    @pytest.mark.parametrize(("select_algorithm", "prepare_consumes_gradient"), _CONTROLLED_WALK_CASES)
    def test_standard_qpe_measures_the_walk_phase(self, select_algorithm, prepare_consumes_gradient):
        """Every standard QPE shot lands on phi = 3/8 or 5/8, and the energy decodes."""
        problem = _exact_walk_phase_problem(3.0 / 8.0)
        result = _run_controlled_walk_qpe(
            StandardPhaseEstimation(shots=8),
            "qdk_standard",
            problem,
            num_bits=3,
            select_algorithm=select_algorithm,
            prepare_consumes_gradient=prepare_consumes_gradient,
        )

        assert any(result.phase_fraction == pytest.approx(phase, abs=1e-12) for phase in (3.0 / 8.0, 5.0 / 8.0)), (
            f"measured phase {result.phase_fraction} is neither 3/8 nor 5/8"
        )
        assert result.raw_energy == pytest.approx(problem["energy"], abs=1e-9)

        counts = _walk_qpe_circuit_counts(
            "qdk_standard",
            problem,
            num_bits=3,
            select_algorithm=select_algorithm,
            prepare_consumes_gradient=prepare_consumes_gradient,
            shots=16,
        )
        assert set(counts) <= {"011", "101"}, f"shots leaked off 3/8 and 5/8: {counts}"

    @pytest.mark.parametrize("combine_iterations", [False, True], ids=["per_bit", "combined"])
    @pytest.mark.parametrize(("select_algorithm", "prepare_consumes_gradient"), _CONTROLLED_WALK_CASES)
    def test_iterative_qpe_measures_the_walk_phase(
        self, select_algorithm, prepare_consumes_gradient, combine_iterations
    ):
        """Per-bit and combined IQPE measure phi = 1/4 or 3/4 and decode the energy."""
        problem = _exact_walk_phase_problem(1.0 / 4.0)
        result = _run_controlled_walk_qpe(
            IterativePhaseEstimation(shots_per_bit=3, combine_iterations=combine_iterations),
            "qdk_iterative",
            problem,
            num_bits=3,
            select_algorithm=select_algorithm,
            prepare_consumes_gradient=prepare_consumes_gradient,
        )

        assert any(result.phase_fraction == pytest.approx(phase, abs=1e-12) for phase in (1.0 / 4.0, 3.0 / 4.0)), (
            f"measured phase {result.phase_fraction} is neither 1/4 nor 3/4"
        )
        assert result.raw_energy == pytest.approx(problem["energy"], abs=1e-9)

    @pytest.mark.parametrize(("select_algorithm", "prepare_consumes_gradient"), _CONTROLLED_WALK_CASES)
    def test_iterative_qpe_iteration_measures_the_walk_overlap(self, select_algorithm, prepare_consumes_gradient):
        """A power-1 iteration at phi = 3/8 measures 1 with probability (1 - cos 2 pi phi) / 2, within 4 sigma."""
        walk_phase = 3.0 / 8.0
        shots = 64
        counts = _walk_qpe_circuit_counts(
            "qdk_iterative",
            _exact_walk_phase_problem(walk_phase),
            num_bits=1,
            select_algorithm=select_algorithm,
            prepare_consumes_gradient=prepare_consumes_gradient,
            shots=shots,
            num_iteration=0,
        )

        probability_one = 0.5 * (1.0 - math.cos(2.0 * math.pi * walk_phase))
        tolerance = 4.0 * math.sqrt(shots * probability_one * (1.0 - probability_one))
        assert abs(counts.get("1", 0) - shots * probability_one) <= tolerance, (
            f"expected about {shots * probability_one:.1f} of {shots} shots on 1, got {counts}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Resource estimation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAResourceEstimation:
    """Logical-resource estimation of the SOSSA unary-iteration QPE circuit."""

    def test_fe2s2_logical_resource_estimate(self):
        """Pin the Fe2S2-20 logical cost of the circuit that actually runs.

        Uses a random factorized Hamiltonian rather than the H2 data because the point is
        to exercise the register widths at ``(N, R, B, C) = (20, 14, 15, 5)``, not to
        recover an energy.
        """
        num_orbitals = 20
        factorized = create_random_factorized_hamiltonian(
            num_orbitals=num_orbitals, num_ranks=14, num_bases=15, num_copies=5
        )
        operator = create("qubit_mapper", "sum_of_squares").run(
            Hamiltonian(factorized), MajoranaMapping.jordan_wigner(2 * num_orbitals)
        )

        hf_config = Configuration.canonical_hf_configuration(15, 15, num_orbitals)
        reference = Wavefunction(StateVectorContainer(hf_config, create_test_orbitals(num_orbitals)))
        state_prep = create("state_prep", "sparse_isometry").run(reference)

        builder = create(
            "qpe_circuit_builder",
            "qdk_unary",
            num_queries=10_162,
            circuit_mapper=_sossa_circuit_mapper_ref(
                outer_prepare_algorithm="alias_sampling",
                inner_prepare_algorithm="controlled_alias_sampling",
                select_algorithm="qrom_phase_gradient",
                coefficient_bit_precision=11,
                rotation_bit_precision=15,
            ),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
        )
        circuit = builder.run(state_preparation=state_prep, qubit_hamiltonian=operator)[0]

        logical_counts = circuit.estimate().logical_counts

        toffoli_count = logical_counts["cczCount"] + logical_counts["ccixCount"]

        assert toffoli_count == pytest.approx(42_558_509, rel=0.01)
        assert logical_counts["numQubits"] == 486
