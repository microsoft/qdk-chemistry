"""Shared helper functions for QDK/Chemistry tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import (
    Ansatz,
    BasisSet,
    CanonicalFourCenterHamiltonianContainer,
    CasWavefunctionContainer,
    Configuration,
    Hamiltonian,
    Orbitals,
    OrbitalType,
    Shell,
    Structure,
    Wavefunction,
)


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
    det = Configuration(config_string)

    # Single determinant with coefficient 1.0
    coeffs = np.array([1.0])
    container = CasWavefunctionContainer(coeffs, [det], orbitals)

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
    det = Configuration(config_string)

    # Single determinant with coefficient 1.0
    coeffs = np.array([1.0])
    container = CasWavefunctionContainer(coeffs, [det], orbitals)
    wavefunction = Wavefunction(container)

    return Ansatz(hamiltonian, wavefunction)


def _hf_determinant(n_alpha: int, n_beta: int, n_orbitals: int) -> np.ndarray:
    """Build the Hartree-Fock reference determinant [alpha|beta].

    Args:
        n_alpha: Number of alpha electrons.
        n_beta: Number of beta electrons.
        n_orbitals: Number of spatial orbitals.

    Returns:
        Occupation array of shape ``(2 * n_orbitals,)`` with 1s for occupied
        alpha/beta orbitals and 0s elsewhere.

    """
    det = np.zeros(2 * n_orbitals, dtype=np.int8)
    det[:n_alpha] = 1
    det[n_orbitals : n_orbitals + n_beta] = 1
    return det


def _random_excitation(det: np.ndarray, n_orbitals: int, rng: np.random.Generator) -> np.ndarray | None:
    """Apply a random excitation independently in alpha and beta channels.

    Args:
        det: Base determinant occupation array.
        n_orbitals: Number of spatial orbitals.
        rng: NumPy random generator.

    Returns:
        New determinant array, or ``None`` if no excitation was applied.

    """
    new_det = det.copy()
    for channel_start in (0, n_orbitals):
        channel = det[channel_start : channel_start + n_orbitals]
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
    return None if np.array_equal(new_det, det) else new_det


def _generate_determinant_matrix(
    n_electrons: int,
    n_orbitals: int,
    n_dets: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate a determinant occupation matrix from HF + random excitations.

    Builds the Hartree-Fock reference determinant and applies random
    single/double excitations to produce ``n_dets`` distinct determinants.

    Args:
        n_electrons: Total number of electrons (split equally between alpha/beta).
        n_orbitals: Number of spatial orbitals.
        n_dets: Target number of determinants.
        seed: Random seed for reproducibility.

    Returns:
        Occupation matrix of shape ``(n_dets, 2 * n_orbitals)`` where each row
        is a determinant with alpha and beta occupation blocks.

    """
    n_alpha = n_electrons // 2
    n_beta = n_electrons - n_alpha
    rng = np.random.default_rng(seed)
    hf = _hf_determinant(n_alpha, n_beta, n_orbitals)

    seen: set[bytes] = {hf.tobytes()}
    dets = [hf]
    for _ in range(n_dets * 200):
        if len(dets) >= n_dets:
            break
        exc = _random_excitation(hf, n_orbitals, rng)
        if exc is not None and exc.tobytes() not in seen:
            seen.add(exc.tobytes())
            dets.append(exc)

    return np.array(dets, dtype=np.int8)


def create_random_bitstring_matrix(
    n_electrons: int,
    n_orbitals: int,
    n_dets: int,
    seed: int = 0,
) -> np.ndarray:
    """Generate a random bitstring matrix suitable for GF2+X elimination.

    Builds physically meaningful determinants from the Hartree-Fock reference
    plus random excitations, then transposes to the binary matrix form
    expected by ``gf2x_with_tracking`` and ``BinaryEncodingSynthesizer``.

    Args:
        n_electrons: Total number of electrons.
        n_orbitals: Number of spatial orbitals.
        n_dets: Target number of determinants (columns).
        seed: Random seed for reproducibility.

    Returns:
        Binary matrix of shape ``(2 * n_orbitals, n_dets)`` where rows are
        qubits and columns are determinants.

    """
    det_matrix = _generate_determinant_matrix(n_electrons, n_orbitals, n_dets, seed)
    return det_matrix.T


def create_random_wavefunction(
    n_electrons: int,
    n_orbitals: int,
    n_dets: int,
    seed: int = 0,
) -> Wavefunction:
    """Generate a random normalised Wavefunction for testing.

    Builds physically meaningful determinants from the Hartree-Fock reference
    plus random excitations, assigns random normalised coefficients, and wraps
    them in a :class:`Wavefunction`.

    Args:
        n_electrons: Total number of electrons.
        n_orbitals: Number of spatial orbitals.
        n_dets: Target number of determinants.
        seed: Random seed for reproducibility.

    Returns:
        A normalised :class:`Wavefunction` with ``n_dets`` determinants.

    """
    det_matrix = _generate_determinant_matrix(n_electrons, n_orbitals, n_dets, seed)
    actual_n_dets = det_matrix.shape[0]

    mapping = {(1, 1): "2", (1, 0): "u", (0, 1): "d", (0, 0): "0"}
    configs = [
        Configuration("".join(mapping[int(row[i]), int(row[n_orbitals + i])] for i in range(n_orbitals)))
        for row in det_matrix
    ]

    coeff_rng = np.random.default_rng(seed)
    raw = coeff_rng.standard_normal(actual_n_dets)
    coeffs = raw / np.linalg.norm(raw)

    orbitals = create_test_orbitals(n_orbitals)
    return Wavefunction(CasWavefunctionContainer(coeffs, configs, orbitals))
