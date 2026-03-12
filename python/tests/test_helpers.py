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
    DensityFittedHamiltonianContainer,
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


def create_test_hamiltonian(num_orbitals: int, container_type: str = "canonical_four_center"):
    """Helper function to create test Hamiltonian objects.

    Args:
        num_orbitals: Number of molecular orbitals
        container_type: Type of container to use. Options are:
            - "canonical_four_center" (default): Uses CanonicalFourCenterHamiltonianContainer
            - "density_fitted": Uses DensityFittedHamiltonianContainer

    Returns:
        Hamiltonian: A test Hamiltonian with the specified container type

    """
    one_body = np.eye(num_orbitals)
    fock = np.eye(0)
    orbitals = create_test_orbitals(num_orbitals)

    if container_type == "canonical_four_center":
        two_body = np.zeros(num_orbitals**4)
        return Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.0, fock))
    if container_type == "density_fitted":
        # Create three-center integrals [n_aux x n_geminals]
        # Using n_aux = num_orbitals auxiliary basis functions
        n_geminals = num_orbitals**2
        n_aux = num_orbitals
        three_center = np.zeros((n_aux, n_geminals))
        return Hamiltonian(DensityFittedHamiltonianContainer(one_body, three_center, orbitals, 0.0, fock))
    raise ValueError(f"Unknown container_type: {container_type}. Use 'canonical_four_center' or 'density_fitted'.")


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
