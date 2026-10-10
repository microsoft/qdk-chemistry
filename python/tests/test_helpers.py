"""Shared helper functions for QDK/Chemistry tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np

from qdk_chemistry.algorithms.qubit_mapper.sum_of_squares import SumOfSquaresQubitMapper
from qdk_chemistry.data import (
    Ansatz,
    BasisSet,
    CanonicalFourCenterHamiltonianContainer,
    Configuration,
    FactorizedHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    ModelOrbitals,
    Orbitals,
    OrbitalType,
    Shell,
    StateVectorContainer,
    Structure,
    Wavefunction,
)
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.qsharp import get_qsharp_context


def create_sparse_wavefunction(num_qubits: int, indices: list[int], amplitudes: list[float]) -> Wavefunction:
    """Create a ``Wavefunction`` occupying only *indices* of a ``num_qubits`` register."""
    dets = [Configuration.from_bitstring(format(idx, f"0{num_qubits}b")[::-1]) for idx in indices]
    container = StateVectorContainer(np.array([float(a) for a in amplitudes]), dets, ModelOrbitals(num_qubits))
    return Wavefunction(container)


def create_dense_wavefunction(amplitudes: list[float]) -> Wavefunction:
    """Create a ``Wavefunction`` holding *amplitudes* on consecutive basis states."""
    num_qubits = math.ceil(math.log2(len(amplitudes))) if len(amplitudes) > 1 else 1
    return create_sparse_wavefunction(num_qubits, list(range(len(amplitudes))), amplitudes)


def create_test_basis_set(num_atomic_orbitals, name="test-basis", structure=None):
    """Create a test basis set with the specified number of atomic orbitals.

    Args:
        num_atomic_orbitals: Number of atomic orbitals to generate
        name: Name for the basis set
        structure: a structure to attach

    Returns:
        qdk_chemistry.data.BasisSet: A valid basis set for testing

    """
    shells = []
    atom_index = 0
    functions_created = 0

    # Create shells to reach the desired number of atomic orbitals
    while functions_created < num_atomic_orbitals:
        remaining = num_atomic_orbitals - functions_created

        if remaining >= 3:
            # Add a P shell (3 functions: Px, Py, Pz)
            exps = np.array([1.0, 0.5])
            coefs = np.array([0.6, 0.4])
            shell = Shell(atom_index, OrbitalType.P, exps, coefs)
            shells.append(shell)
            functions_created += 3
        elif remaining >= 1:
            # Add S shells for remaining functions (1 function each)
            for _ in range(remaining):
                exps = np.array([1.0])
                coefs = np.array([1.0])
                shell = Shell(atom_index, OrbitalType.S, exps, coefs)
                shells.append(shell)
                functions_created += 1
    if structure is not None:
        return BasisSet(name, shells, structure)
    return BasisSet(name, shells)


def create_test_orbitals(num_orbitals: int):
    """Helper: construct Orbitals immutably with identity coeffs and occupations.

    Occupations are set in restricted form as total occupancy per MO (0/1/2).
    """
    coeffs = np.eye(num_orbitals)
    basis_set = create_test_basis_set(num_orbitals)
    return Orbitals(coeffs, None, None, basis_set)


def create_test_hamiltonian(num_orbitals: int):
    """Helper function to create test Hamiltonian objects."""
    one_body = np.eye(num_orbitals)
    two_body = np.zeros(num_orbitals**4)
    fock = np.eye(0)
    orbitals = create_test_orbitals(num_orbitals)
    return Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.0, fock))


def create_nontrivial_test_hamiltonian(num_orbitals: int = 2):
    """Create a Hamiltonian with nonzero one- and two-body integrals.

    Generates integrals deterministically (fixed seed) so that every orbital
    participates in both one-body and two-body terms, producing non-trivial
    qubit operators for any ``num_orbitals`` value.  The two-body tensor has
    full 8-fold permutation symmetry appropriate for real orbitals in chemist
    notation ``(pq|rs)``.

    Args:
        num_orbitals: Number of spatial orbitals (default 2).

    Returns:
        qdk_chemistry.data.Hamiltonian: A Hamiltonian with realistic integrals.

    """
    n = num_orbitals
    rng = np.random.default_rng(42)

    # Symmetric one-body matrix with diagonal dominance
    raw = rng.standard_normal((n, n)) * 0.3
    one_body = (raw + raw.T) / 2
    one_body += np.diag(np.linspace(1.0, -0.5, n))

    # Two-body integrals with 8-fold symmetry for real orbitals:
    #   (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr)
    #           = (rs|pq) = (sr|pq) = (rs|qp) = (sr|qp)
    h2 = np.zeros((n, n, n, n))
    seen: set[tuple[int, ...]] = set()
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    perms = {
                        (p, q, r, s),
                        (q, p, r, s),
                        (p, q, s, r),
                        (q, p, s, r),
                        (r, s, p, q),
                        (s, r, p, q),
                        (r, s, q, p),
                        (s, r, q, p),
                    }
                    canon = min(perms)
                    if canon in seen:
                        continue
                    seen.add(canon)
                    val = rng.standard_normal() * 0.2
                    for a, b, c, d in perms:
                        h2[a, b, c, d] = val

    two_body = h2.ravel()
    fock = np.eye(0)
    orbitals = create_test_orbitals(n)
    return Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.5, fock))


def create_test_shells(num_atoms: int = 1, atoms_types: list | None = None):
    """Helper function to create test shells for BasisSet."""
    if atoms_types is None:
        atoms_types = [1] * num_atoms  # Default to hydrogen atoms

    shells = []
    for atom_idx in range(num_atoms):
        # Create s shell
        s_shell = Shell(atom_idx, OrbitalType.S)
        s_shell.add_primitive(1.0, 1.0)
        shells.append(s_shell)

        # For heavier atoms, add p shell
        if atoms_types[atom_idx] > 2:  # Beyond helium
            p_shell = Shell(atom_idx, OrbitalType.P)
            p_shell.add_primitive(0.5, 1.0)
            shells.append(p_shell)

    return shells


def create_test_structure(atoms: list | None = None):
    """Helper: create Structure immutably from a list of (pos, Z)."""
    if atoms is None:
        atoms = [([0.0, 0.0, 0.0], 1), ([1.4, 0.0, 0.0], 1)]
    coords = np.array([pos for pos, _ in atoms], dtype=float)
    charges = [z for _, z in atoms]
    return Structure(coords, charges)


def create_h2_molecule():
    """Create a standard H2 molecule for testing."""
    return create_test_structure(
        [
            ([0.0, 0.0, 0.0], 1),  # H1
            ([1.4, 0.0, 0.0], 1),  # H2
        ]
    )


def create_h2o_molecule():
    """Create a standard H2O molecule for testing."""
    return create_test_structure(
        [
            ([0.0, 0.0, 0.0], 8),  # O
            ([0.96, 0.0, 0.0], 1),  # H1
            ([-0.24, 0.93, 0.0], 1),  # H2
        ]
    )


def create_he_atom():
    """Create a helium atom for testing."""
    return create_test_structure([([0.0, 0.0, 0.0], 2)])


def create_test_wavefunction(num_orbitals: int = 2):
    """Helper function to create a simple CAS wavefunction for testing.

    Args:
        num_orbitals: Number of orbitals (default 2)

    Returns:
        qdk_chemistry.data.Wavefunction: A simple wavefunction with single determinant

    """
    orbitals = create_test_orbitals(num_orbitals)

    # Create single determinant configuration (e.g., "20" for 2 electrons in first orbital)
    config_string = "2" + "0" * (num_orbitals - 1)
    det = Configuration.from_spin_half_string(config_string)

    # Single determinant with coefficient 1.0
    coeffs = np.array([1.0])
    container = StateVectorContainer(coeffs, [det], orbitals)

    return Wavefunction(container)


def create_test_ansatz(num_orbitals: int = 2):
    """Helper function to create a test Ansatz for testing.

    Args:
        num_orbitals: Number of orbitals (default 2)

    Returns:
        qdk_chemistry.data.Ansatz: A simple ansatz with hamiltonian and wavefunction

    """
    # Create shared orbitals for both hamiltonian and wavefunction
    orbitals = create_test_orbitals(num_orbitals)

    # Create hamiltonian using the shared orbitals
    one_body = np.eye(num_orbitals)
    two_body = np.zeros(num_orbitals**4)
    fock = np.eye(0)
    hamiltonian = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.0, fock))

    # Create wavefunction using the same shared orbitals
    # Create single determinant configuration (e.g., "20" for 2 electrons in first orbital)
    config_string = "2" + "0" * (num_orbitals - 1)
    det = Configuration.from_spin_half_string(config_string)

    # Single determinant with coefficient 1.0
    coeffs = np.array([1.0])
    container = StateVectorContainer(coeffs, [det], orbitals)
    wavefunction = Wavefunction(container)

    return Ansatz(hamiltonian, wavefunction)


def create_random_factorized_hamiltonian(
    num_orbitals: int = 2,
    num_ranks: int = 2,
    num_bases: int = 1,
    num_copies: int = 1,
    *,
    seed: int = 42,
):
    """Create a random FactorizedHamiltonianContainer for testing.

    Args:
        num_orbitals: Number of spatial orbitals (N).
        num_ranks: Number of ranks (R).
        num_bases: Number of bases (B).
        num_copies: Number of copies (C).
        seed: Random seed for reproducibility.

    Returns:
        FactorizedHamiltonianContainer from C++ pybind11.

    """
    rng = np.random.default_rng(seed)
    n, r, b, c = num_orbitals, num_ranks, num_bases, num_copies

    # Symmetric one-body integrals
    h1 = rng.standard_normal((n, n))
    h1 = (h1 + h1.T) / 2

    # Random normalized basis vectors (U), flattened [R*B*N]
    u_matrices = np.zeros(r * b * n)
    for ri in range(r):
        for bi in range(b):
            v = rng.standard_normal(n)
            v /= np.linalg.norm(v)
            u_matrices[ri * b * n + bi * n : ri * b * n + (bi + 1) * n] = v

    # Two-body weights W [R*B*C]
    w_matrices = rng.standard_normal(r * b * c)

    # Identity weights WB [R, C]
    wb_matrix = rng.standard_normal((r, c))

    orbitals = create_test_orbitals(n)
    inactive_fock = np.zeros((n, n))

    return FactorizedHamiltonianContainer(
        one_body_integrals=h1,
        u_matrices=u_matrices,
        w_matrices=w_matrices,
        wb_matrix=wb_matrix,
        orbitals=orbitals,
        core_energy=0.0,
        inactive_fock_matrix=inactive_fock,
    )


def factorized_hamiltonian_to_sossa_operator(factorized_hamiltonian):
    """Map a factorized Hamiltonian to the SOSSA QubitOperator the SOSSA builder expects.

    Args:
        factorized_hamiltonian: The FactorizedHamiltonianContainer to map.

    Returns:
        The SOSSA QubitOperator.

    """
    num_modes = 2 * factorized_hamiltonian.get_num_orbitals()
    hamiltonian = Hamiltonian(factorized_hamiltonian)
    return SumOfSquaresQubitMapper().run(hamiltonian, MajoranaMapping.jordan_wigner(num_modes))


def create_random_bitstring_matrix(
    n_electrons: int,
    n_orbitals: int,
    n_dets: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate a random bitstring matrix for sparse isometry testing.

    Args:
        n_electrons: Total number of electrons.
        n_orbitals: Number of spatial orbitals.
        n_dets: Target number of determinants (columns).
        seed: Random seed for reproducibility.

    Returns:
        Binary matrix of shape ``(2 * n_orbitals, n_dets)`` where rows are
        qubits and columns are determinants.

    """
    n_alpha = n_electrons // 2
    n_beta = n_electrons - n_alpha
    rng = np.random.default_rng(seed)

    hf_config = Configuration.canonical_hf_configuration(n_alpha, n_beta, n_orbitals)
    alpha_bits, beta_bits = hf_config.to_binary_strings(n_orbitals)
    hf = np.array([int(bit) for bit in alpha_bits + beta_bits], dtype=np.int8)

    seen: set[bytes] = {hf.tobytes()}
    dets = [hf]
    for _ in range(n_dets * 200):
        if len(dets) >= n_dets:
            break
        new_det = hf.copy()
        for channel_start in (0, n_orbitals):
            channel = hf[channel_start : channel_start + n_orbitals]
            occupied = np.where(channel == 1)[0]
            virtual = np.where(channel == 0)[0]
            if len(occupied) == 0 or len(virtual) == 0:
                continue
            order = rng.integers(0, min(len(occupied), len(virtual)) + 1)
            if order == 0:
                continue
            occ = rng.choice(occupied, size=order, replace=False)
            vir = rng.choice(virtual, size=order, replace=False)
            new_det[channel_start + occ] = 0
            new_det[channel_start + vir] = 1
        if not np.array_equal(new_det, hf) and new_det.tobytes() not in seen:
            seen.add(new_det.tobytes())
            dets.append(new_det)

    return np.array(dets, dtype=np.int8).T


def create_random_wavefunction(
    n_electrons: int,
    n_orbitals: int,
    n_dets: int,
    seed: int = 0,
) -> Wavefunction:
    """Generate a random normalised Wavefunction for testing.

    Builds physically meaningful determinants from the Hartree-Fock reference
    plus random excitations, assigns random normalised coefficients.

    Args:
        n_electrons: Total number of electrons.
        n_orbitals: Number of spatial orbitals.
        n_dets: Target number of determinants.
        seed: Random seed for reproducibility.

    Returns:
        A normalised :class:`Wavefunction` with ``n_dets`` determinants.

    """
    det_matrix = create_random_bitstring_matrix(n_electrons, n_orbitals, n_dets, seed).T
    actual_n_dets = det_matrix.shape[0]

    mapping = {(1, 1): "2", (1, 0): "u", (0, 1): "d", (0, 0): "0"}
    configs = [
        Configuration.from_spin_half_string(
            "".join(mapping[int(row[i]), int(row[n_orbitals + i])] for i in range(n_orbitals))
        )
        for row in det_matrix
    ]

    coeff_rng = np.random.default_rng(seed)
    raw = coeff_rng.standard_normal(actual_n_dets)
    coeffs = raw / np.linalg.norm(raw)

    orbitals = create_test_orbitals(n_orbitals)
    return Wavefunction(StateVectorContainer(coeffs, configs, orbitals))


def apply_controlled_operation(operation, state: np.ndarray) -> np.ndarray:
    """Return the state a ``(control, systems)`` Q# operation produces from a real-amplitude ``state``.

    Qubit 0 is the control and the most significant bit of the basis index; the rest are the systems.
    The whole machine is dumped instead of using ``DumpRegister``, which rejects states with nonzero
    amplitudes below its zero cutoff as not separable. Every helper qubit is released by then.
    """
    context = get_qsharp_context()
    if not hasattr(context.code, "_ControlledTestDumpMachine"):
        context.eval(
            "operation _ControlledTestDumpMachine("
            "op : ((Qubit, Qubit[]) => Unit), numQubits : Int, initial : Double[]) : Unit {"
            " use qubits = Qubit[numQubits];"
            " Std.StatePreparation.PreparePureStateD(initial, qubits);"
            " op(qubits[0], qubits[1...]);"
            " Std.Diagnostics.DumpMachine();"
            " ResetAll(qubits); }"
        )
    num_qubits = round(math.log2(len(state)))
    run = context.run(
        context.code._ControlledTestDumpMachine, 1, operation, num_qubits, state.tolist(), save_events=True
    )
    result = np.zeros(len(state), dtype=complex)
    for index, amplitude in run[0]["events"][-1].state_dump().get_dict().items():
        result[index] = amplitude
    return result


def controlled_product_formula_state(container: PauliProductFormulaContainer, state: np.ndarray) -> np.ndarray:
    r"""Apply the controlled product formula of ``container`` to ``state`` exactly, term by term.

    The layout is that of :func:`apply_controlled_operation`. Each factor uses
    :math:`e^{-i\theta P} = \cos\theta - i\sin\theta P` on the control-one half only.
    """
    num_qubits = round(math.log2(len(state)))
    indices = np.arange(len(state))
    controlled = (indices >> (num_qubits - 1)) & 1 == 1
    terms = [*container.prefix_terms, *container.step_terms * container.step_reps, *container.suffix_terms]
    result = state.astype(complex)
    for term in terms:
        flip = parity_mask = num_y = 0
        for qubit, axis in term.pauli_term.items():
            bit = 1 << (num_qubits - 2 - qubit)
            flip |= bit if axis in "XY" else 0
            parity_mask |= bit if axis in "YZ" else 0
            num_y += axis == "Y"
        parity = np.zeros(len(state), dtype=int)
        for position in range(num_qubits):
            if parity_mask >> position & 1:
                parity ^= (indices >> position) & 1
        pauli = np.zeros_like(result)
        pauli[indices ^ flip] = 1j**num_y * (1 - 2 * parity) * result
        rotated = np.cos(term.angle) * result - 1j * np.sin(term.angle) * pauli
        result = np.where(controlled, rotated, result)
    return result


def random_sparse_state(num_qubits: int, support: int, seed: int) -> np.ndarray:
    """Return a normalized real state on ``support`` random basis states."""
    rng = np.random.default_rng(seed)
    state = np.zeros(2**num_qubits)
    state[rng.choice(len(state), size=support, replace=False)] = rng.normal(size=support)
    return state / np.linalg.norm(state)


def assert_states_match_up_to_global_phase(actual: np.ndarray, expected: np.ndarray, atol: float) -> None:
    """Assert two normalized states agree once the global phase of ``actual`` is aligned to ``expected``."""
    overlap = np.vdot(actual, expected)
    np.testing.assert_allclose(actual * overlap / abs(overlap), expected, atol=atol, rtol=0)
