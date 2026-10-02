"""Integration tests for unary-iteration QPE with the SOSSA block encoding.

Tests the full pipeline:
    DFTHCHamiltonianContainer → SOSSABuilder → UnitaryRepresentation
    → SOSSAMapper → Circuit → unary-iteration QPE → energy
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
from qdk_chemistry.algorithms.circuit_mapper.sossa_mapper import rotation_batch_size_for
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.phase_estimation.unary_phase_estimation import UnaryPhaseEstimation
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    Configuration,
    DFTHCHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .reference_tolerances import ci_energy_tolerance
from .test_helpers import (
    create_random_factorized_hamiltonian,
    create_test_orbitals,
    factorized_hamiltonian_to_sossa_operator,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Test Hamiltonian construction (small DFTHC-like H2 data)
# ═══════════════════════════════════════════════════════════════════════════════


#: The factorized H2 shipped with the examples, which the SOSSA notebook also loads.
H2_DFTHC_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "data" / "h2_dfthc_r2_b2_c1.hamiltonian.json"


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
    """Replicate ``DFTHCHamiltonianContainer::get_h1_prime`` (Eq. 36).

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
    perm = _python_to_qsharp_permutation(num_orbitals)
    signs = _python_to_qsharp_sign(num_orbitals)
    gs_idx = sector_indices[0]
    gs_energy = eigenvalues[gs_idx]
    gs_vec = eigenvectors[:, gs_idx]

    # Apply permutation
    gs_vec_qs = np.zeros(dim)
    for i in range(dim):
        gs_vec_qs[perm[i]] = signs[i] * gs_vec[i]

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


def _sossa_circuit_mapper_ref(
    *,
    outer_prepare_algorithm: str = "dense_pure_state",
    inner_prepare_algorithm: str = "direct",
    select_algorithm: str = "direct",
    coefficient_bit_precision: int = 10,
    rotation_bit_precision: int = 10,
    **settings,
) -> AlgorithmRef:
    """Return an AlgorithmRef for the SOSSA circuit mapper.

    Extra keyword arguments are passed through to the mapper's settings, so a caller can
    vary one knob -- ``rotation_batch_size``, say -- without this helper having to know
    about it.
    """
    return AlgorithmRef(
        "circuit_mapper",
        "sossa",
        outer_prepare_algorithm=AlgorithmRef("state_prep", outer_prepare_algorithm),
        inner_prepare_algorithm=inner_prepare_algorithm,
        select_algorithm=select_algorithm,
        coefficient_bit_precision=coefficient_bit_precision,
        rotation_bit_precision=rotation_bit_precision,
        **settings,
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

    fh = DFTHCHamiltonianContainer(
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

    num_system_qubits = 2 * n_orb
    state_prep_params = {
        "rowMap": list(range(num_system_qubits - 1, -1, -1)),
        "stateVector": gs_vec.real.tolist(),
        "expansionOps": [],
        "numQubits": num_system_qubits,
    }
    state_prep = Circuit(
        qsharp_factory=QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=state_prep_params,
        ),
        qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params),
    )

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

        fh = DFTHCHamiltonianContainer(
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
        fh = DFTHCHamiltonianContainer(
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
        fh = DFTHCHamiltonianContainer(
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
# Resource estimation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSOSSAQPEScope:
    """SOSSA is supported under unary QPE only."""

    def test_no_controlled_mapper_accepts_the_sossa_block_encoding(self):
        """Iterative and standard QPE cannot drive the SOSSA walk.

        Both build their circuits through a ``controlled_circuit_mapper``, so SOSSA would
        need one that accepts ``SOSSABlockEncodingContainer``. None is registered, which is what
        confines SOSSA to ``qdk_unary``. If a controlled SOSSA mapper is ever added, this
        test fails and the unary-only claim in the mapper and container docstrings should
        be revisited alongside it.
        """
        factorized = create_random_factorized_hamiltonian(num_orbitals=2, num_ranks=1, num_bases=2, num_copies=1)
        unitary = SOSSABuilder().run(factorized_hamiltonian_to_sossa_operator(factorized))

        mappers = list(available("controlled_circuit_mapper"))
        assert mappers, "expected at least one registered controlled circuit mapper"
        for name in mappers:
            with pytest.raises(ValueError, match="not supported"):
                create("controlled_circuit_mapper", name).run(unitary)


class TestSOSSAResourceEstimation:
    """Logical-resource estimation of the SOSSA unary-iteration QPE circuit."""

    # Largest rotation batch that still reaches the floor the peak width can be pushed to.
    # Derived rather than swept: width falls linearly as angles leave the register, so
    # W(lambda) = max(W_floor, W_resident - (A - lambda) * b_rot) over A = N - 1 = 19 angles,
    # and the two branches meet at A - ceil((W_resident - W_floor) / b_rot), here
    # 19 - ceil((486 - 379) / 15) = 11. Two estimates fix it, not a sweep.
    #
    # The floor is 379 rather than 427 because ``ComputeOptimalLambda2D`` declines the widest
    # swap network; narrowing PREPARE lowers the floor, which moves this knee *down* by
    # putting SELECT back on the critical path sooner. That coupling is the reason the
    # ``inner_prepare_swap_bits`` docs say to re-derive the batch size afterwards.
    #
    # Note this is where the floor *ends*, not the batch to recommend. Being the largest
    # lambda that still reaches the floor makes it the one with no headroom left, and
    # lambda = 10 buys the identical circuit; see
    # ``test_only_the_smallest_batch_of_each_step_is_worth_choosing``.
    _FE2S2_BATCH_KNEE = 11

    @staticmethod
    def _fe2s2_logical_counts(**select_settings):
        """Estimate the Fe2S2-20 circuit, varying only the SELECT rotation settings.

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
                **select_settings,
            ),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
        )
        circuit = builder.run(state_preparation=state_prep, qubit_hamiltonian=operator)[0]

        counts = circuit.estimate().logical_counts
        return counts["numQubits"], counts["cczCount"] + counts["ccixCount"]

    def test_fe2s2_logical_resource_estimate(self):
        """Pin the Fe2S2-20 logical cost of the circuit that actually runs."""
        num_qubits, toffoli_count = self._fe2s2_logical_counts()

        assert toffoli_count == pytest.approx(42_558_509, rel=0.01)
        assert num_qubits == 486

    def test_streaming_the_rotation_angles_trades_toffolis_for_qubits(self):
        """Streaming is a Pareto move, not a free win, and the default must stay put.

        Inside SELECT the resident angle register is the dominant cost -- all ``N - 1``
        words at once, 285 of its qubits at ``b_rot = 15``. Across the whole walk it is not
        the only wide stage, so streaming lowers the peak until a different stage becomes
        the bottleneck and then stops helping. What is pinned here is the direction of both
        axes, not the magnitude of either, so the lookup cost model stays free to change.
        """
        resident_qubits, resident_toffolis = self._fe2s2_logical_counts()
        streamed_qubits, streamed_toffolis = self._fe2s2_logical_counts(rotation_batch_size=self._FE2S2_BATCH_KNEE)

        assert streamed_qubits < resident_qubits, (
            f"streaming should lower the peak width: {resident_qubits} -> {streamed_qubits}"
        )
        assert streamed_toffolis > resident_toffolis, (
            "streaming reloads each batch to uncompute, so it cannot be free in Toffolis"
        )

    def test_the_tightest_batch_is_dominated_by_larger_ones_at_the_same_width(self):
        """Over-shrinking the batch buys nothing, which is half of the usage guidance.

        The Toffoli penalty tracks the *number* of batches, ``ceil((N - 1) / lambda)``, not
        ``lambda`` itself, while the qubit saving stops once some other stage sets the peak
        width. At Fe2S2 that makes every ``lambda`` from 1 up to the knee land on the same
        qubit count, so the smallest one is strictly worse than larger ones that reach the
        same floor -- same width, several times the Toffolis.

        The other half is that growing ``lambda`` past the point where the batch count drops
        is equally pointless, so the rule is neither "smallest" nor "widest": pick the batch
        count and derive ``lambda``. See
        ``test_only_the_smallest_batch_of_each_step_is_worth_choosing``.
        """
        tight_qubits, tight_toffolis = self._fe2s2_logical_counts(rotation_batch_size=1)
        knee_qubits, knee_toffolis = self._fe2s2_logical_counts(rotation_batch_size=self._FE2S2_BATCH_KNEE)

        assert knee_qubits == tight_qubits, (
            f"lambda=1 bought extra width over lambda={self._FE2S2_BATCH_KNEE}: {tight_qubits} vs {knee_qubits}"
        )
        assert knee_toffolis < tight_toffolis, (
            f"lambda=1 should cost strictly more than lambda={self._FE2S2_BATCH_KNEE} for the "
            f"same width: {tight_toffolis} vs {knee_toffolis}"
        )

    def test_only_the_smallest_batch_of_each_step_is_worth_choosing(self):
        """The usage guidance: choose the batch *count*, then derive lambda from it.

        Streaming reloads once per batch, so the Toffoli cost is a function of
        ``ceil((N - 1) / lambda)`` and not of ``lambda``, which makes it a step function --
        every ``lambda`` from 10 to 15 splits the 19 angles into two passes and costs
        exactly the same. The rotation register, though, keeps growing at ``b_rot`` qubits
        per angle right across that step. So every ``lambda`` above the smallest in its step
        pays width for nothing.

        That is why "take the widest batch that fits your budget" is the wrong rule and
        ``rotation_batch_size_for`` exists: given room for 440 qubits that rule hands back
        ``lambda = 16``, which is strictly worse than the ``lambda = 10`` derived here --
        61 qubits more for an identical circuit.
        """
        smallest = rotation_batch_size_for(20, num_batches=2)
        assert smallest == 10, f"two passes over 19 angles needs batches of 10, got {smallest}"

        smallest_qubits, smallest_toffolis = self._fe2s2_logical_counts(rotation_batch_size=smallest)
        wider_qubits, wider_toffolis = self._fe2s2_logical_counts(rotation_batch_size=16)

        assert wider_toffolis == smallest_toffolis, (
            f"lambda=16 still makes two passes, so it cannot undercut lambda={smallest}: "
            f"{wider_toffolis} vs {smallest_toffolis}"
        )
        assert wider_qubits > smallest_qubits, (
            f"the six extra angles lambda=16 keeps resident have to cost width: {wider_qubits} vs {smallest_qubits}"
        )

    def test_the_lookup_method_trades_width_against_toffolis_as_advertised(self):
        """Each loader has to be worth choosing somewhere, and none may be a silent regression.

        At the Fe2S2 table shape the SF lookup is only 224 rows against a 15-bit angle word,
        well under the ``numData > 32 * numBits`` point where a *borrowed* swap network starts
        to pay, so ``dirty_select_swap`` is expected to decline it and leave the plain lookup
        in place -- selecting it must therefore cost nothing at all here.
        See ``TestDirtyQROAMCostModel`` for the regime where it does pay.

        A *clean* swap network has no such threshold: it buys the same Toffoli reduction with
        allocated scratch instead of borrowed qubits, so ``select_swap`` should cut Toffolis
        here where ``dirty_select_swap`` cannot. That scratch is the price, and the point of
        offering all three is that the right trade depends on which budget is binding.
        """
        plain_qubits, plain_toffolis = self._fe2s2_logical_counts(
            rotation_batch_size=1, rotation_lookup_method="select"
        )
        clean_qubits, clean_toffolis = self._fe2s2_logical_counts(
            rotation_batch_size=1, rotation_lookup_method="select_swap"
        )
        dirty_qubits, dirty_toffolis = self._fe2s2_logical_counts(
            rotation_batch_size=1, rotation_lookup_method="dirty_select_swap"
        )

        assert (dirty_qubits, dirty_toffolis) == (plain_qubits, plain_toffolis), (
            "the dirty cost model should have declined at this shape and left the plain lookup "
            f"in place, but it changed the cost: {(plain_qubits, plain_toffolis)} -> "
            f"{(dirty_qubits, dirty_toffolis)}"
        )
        assert clean_toffolis < plain_toffolis, (
            f"a clean swap network should undercut the plain lookup: {plain_toffolis} -> {clean_toffolis}"
        )
        assert clean_qubits >= plain_qubits, (
            f"a clean swap network allocates scratch, so it cannot also be narrower: {plain_qubits} -> {clean_qubits}"
        )

    def test_narrowing_the_alias_lookup_only_pays_once_the_angles_are_streamed(self):
        """The two optimisations are sequential, not additive, because the peak stage moves.

        Peak width is a maximum over stages, so narrowing anything but the current widest
        stage is invisible. While the angle word is resident it is the widest thing in the
        walk, and the inner PREPARE lookup -- however it is routed -- sits underneath it.
        Only after streaming removes the angle register does the alias QROAM's swap scratch
        become the stage that sets the peak, and only then does ``inner_prepare_swap_bits``
        move the number at all.

        This is the reason to reach for it second rather than instead: on its own it is pure
        Toffoli overhead.
        """
        batch = rotation_batch_size_for(20, num_batches=2)

        resident_auto_qubits, _ = self._fe2s2_logical_counts()
        resident_narrow_qubits, _ = self._fe2s2_logical_counts(inner_prepare_swap_bits=2)

        assert resident_narrow_qubits == resident_auto_qubits, (
            "with the angle word resident the alias lookup is not the peak stage, so narrowing "
            f"it cannot help: {resident_auto_qubits} -> {resident_narrow_qubits}"
        )

        streamed_auto_qubits, _ = self._fe2s2_logical_counts(rotation_batch_size=batch)
        streamed_narrow_qubits, _ = self._fe2s2_logical_counts(rotation_batch_size=batch, inner_prepare_swap_bits=2)

        assert streamed_narrow_qubits < streamed_auto_qubits, (
            "once the angles are streamed the alias lookup sets the peak, so narrowing it must "
            f"show up: {streamed_auto_qubits} -> {streamed_narrow_qubits}"
        )

    def test_narrowing_the_alias_lookup_buys_width_far_more_cheaply_than_streaming(self):
        """Having streamed the angles, the next qubit is much cheaper than the last one was.

        Streaming pays for width by reloading each batch to uncompute it, so its Toffoli
        bill is steep. Re-routing a QROAM is different in kind: the swap width ``k`` only
        decides how the *same* table is split between address iteration and a swap network,
        so halving the scratch costs one extra select layer rather than a second pass over
        the data.

        Both widths are pinned explicitly so this measures the mechanism rather than the
        default. ``k = 3`` is the Toffoli optimum -- what a width-blind rule picks -- and is
        the honest baseline for "what does re-routing buy", since the shipped default has
        already taken part of that saving for itself.

        What is pinned is the ordering -- the second step must recover at least as much
        width as the first, at a far smaller fraction of the Toffoli count. Magnitudes stay
        free to move with the cost model.
        """
        batch = rotation_batch_size_for(20, num_batches=2)

        resident_qubits, resident_toffolis = self._fe2s2_logical_counts(inner_prepare_swap_bits=3)
        streamed_qubits, streamed_toffolis = self._fe2s2_logical_counts(
            rotation_batch_size=batch, inner_prepare_swap_bits=3
        )
        narrowed_qubits, narrowed_toffolis = self._fe2s2_logical_counts(
            rotation_batch_size=batch, inner_prepare_swap_bits=2
        )

        streaming_saved = resident_qubits - streamed_qubits
        narrowing_saved = streamed_qubits - narrowed_qubits
        assert narrowing_saved >= streaming_saved, (
            f"narrowing the lookup should recover at least what streaming did: "
            f"{narrowing_saved} vs {streaming_saved} qubits"
        )

        streaming_overhead = (streamed_toffolis - resident_toffolis) / resident_toffolis
        narrowing_overhead = (narrowed_toffolis - streamed_toffolis) / streamed_toffolis
        assert narrowing_overhead < streaming_overhead, (
            f"re-routing a lookup should be cheaper than a second pass over the angles: "
            f"{narrowing_overhead:.1%} vs {streaming_overhead:.1%}"
        )

    def test_the_default_alias_swap_width_declines_the_toffoli_optimum_to_save_scratch(self):
        """The default takes the narrowest width that is nearly Toffoli-optimal, not the best.

        ``ComputeOptimalLambda2D`` scores widths by Toffoli count but then takes the
        *narrowest* one within a fifth of the minimum, because each extra swap bit doubles a
        scratch block that sets the peak width of the whole walk. At this shape that declines
        ``k = 3`` -- genuinely the Toffoli minimum -- in favour of ``k = 2``.

        Pinning the comparison rather than the chosen width is what keeps this honest: the
        declined width has to be really cheaper in Toffolis, or the default is not trading
        anything. The premium it pays must also stay small next to the width it buys,
        otherwise the tolerance is set wrong.
        """
        batch = rotation_batch_size_for(20, num_batches=2)

        auto_qubits, auto_toffolis = self._fe2s2_logical_counts(rotation_batch_size=batch)
        optimum_qubits, optimum_toffolis = self._fe2s2_logical_counts(
            rotation_batch_size=batch, inner_prepare_swap_bits=3
        )

        assert auto_qubits < optimum_qubits, (
            f"the default should be declining the widest network: {optimum_qubits} -> {auto_qubits}"
        )
        assert auto_toffolis > optimum_toffolis, (
            f"and it cannot also be cheaper, or nothing is being traded: {optimum_toffolis} -> {auto_toffolis}"
        )

        width_saved = (optimum_qubits - auto_qubits) / optimum_qubits
        toffoli_premium = (auto_toffolis - optimum_toffolis) / optimum_toffolis
        assert toffoli_premium < width_saved, (
            f"the default's Toffoli premium has to be small next to the width it buys, or the "
            f"tolerance is mis-set: {toffoli_premium:.1%} Toffolis for {width_saved:.1%} width"
        )

        narrower_qubits, narrower_toffolis = self._fe2s2_logical_counts(
            rotation_batch_size=batch, inner_prepare_swap_bits=1
        )
        assert narrower_toffolis > auto_toffolis, (
            f"one bit narrower than the default must cost Toffolis: {auto_toffolis} -> {narrower_toffolis}"
        )
        assert narrower_qubits < auto_qubits, (
            f"and it has to be freeing scratch to be worth considering: {auto_qubits} -> {narrower_qubits}"
        )

    def test_only_the_smallest_swap_width_of_each_step_is_worth_choosing(self):
        """The same step-function trap as the rotation batch, in the other knob.

        Scratch grows as ``2^k`` times the loaded word, but the peak is a maximum over
        stages, so once the swap network drops below whatever stage is next-widest, further
        shrinking is invisible while the Toffoli cost keeps climbing. ``k = 1`` lands on the
        same peak as ``k = 2`` here and pays for the privilege, exactly as ``lambda = 1``
        does against the derived batch size.

        So the rule is the same shape in both knobs: take the *largest* setting that still
        reaches the floor, never the smallest that fits.
        """
        batch = rotation_batch_size_for(20, num_batches=2)

        tight_qubits, tight_toffolis = self._fe2s2_logical_counts(rotation_batch_size=batch, inner_prepare_swap_bits=1)
        floor_qubits, floor_toffolis = self._fe2s2_logical_counts(rotation_batch_size=batch, inner_prepare_swap_bits=2)

        assert tight_qubits == floor_qubits, (
            f"k=1 bought no width over k=2, so it is dominated: {tight_qubits} vs {floor_qubits}"
        )
        assert tight_toffolis > floor_toffolis, (
            f"k=1 splits the table into more select layers, so it must cost more: {tight_toffolis} vs {floor_toffolis}"
        )

    def test_an_over_wide_swap_request_is_clamped_rather_than_faulted(self):
        """A setting inside its declared range must never fault inside Q#.

        ``SwappedLoadShape`` asserts ``numSwapBits <= nRequired``, and the table's address
        width is far below the setting's upper bound, so the request is clamped to the
        widest network the table can support. It is then free to be a bad choice -- a wider
        network than the default picked can only add scratch -- but it has to produce an
        estimate rather than an assertion failure.
        """
        batch = rotation_batch_size_for(20, num_batches=2)

        clamped_qubits, clamped_toffolis = self._fe2s2_logical_counts(
            rotation_batch_size=batch, inner_prepare_swap_bits=30
        )
        floor_qubits, _ = self._fe2s2_logical_counts(rotation_batch_size=batch, inner_prepare_swap_bits=2)

        assert clamped_qubits > 0, "an over-wide request should still produce an estimate"
        assert clamped_toffolis > 0, "an over-wide request should still produce an estimate"
        assert clamped_qubits > floor_qubits, (
            f"the widest network the table allows should be wider than the floor: {clamped_qubits} vs {floor_qubits}"
        )
