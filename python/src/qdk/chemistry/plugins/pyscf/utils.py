"""Utilities for converting between QDK/Chemistry and PySCF data structures.

This module provides conversion functions to bridge between QDK/Chemistry and PySCF data structures.
It enables seamless integration between the two quantum chemistry libraries by handling
the conversion of molecular structures, basis sets, and Hamiltonians.

The main functionality includes:

* Converting QDK/Chemistry Structure objects to PySCF atom format
* Converting QDK/Chemistry BasisSet objects to PySCF Mole objects
* Converting PySCF Mole objects back to QDK/Chemistry BasisSet objects
* Converting QDK/Chemistry Hamiltonian objects to PySCF SCF objects

These utilities are essential for workflows that need to leverage both QDK/Chemistry's
data management capabilities and PySCF's quantum chemistry calculations.

Note:
    * Currently supports spherical basis functions only
    * Cartesian basis set support is planned for future versions
    * Assumes atomic numbers do not exceed 200

Examples:
    >>> from qdk.chemistry.plugins.pyscf.utils import structure_to_pyscf_atom_labels, basis_to_pyscf_mol
    >>> # Convert structure to PySCF format
    >>> atoms, pyscf_symbols, elements = structure_to_pyscf_atom_labels(structure)
    >>> # Convert basis set to PySCF Mole object
    >>> pyscf_mol = basis_to_pyscf_mol(basis_set)

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections import Counter

import numpy as np
from pyscf import gto, scf

from qdk.chemistry.data import BasisSet, BasisType, Hamiltonian, Orbitals, Shell, Structure


def structure_to_pyscf_atom_labels(structure: Structure) -> tuple:
    """Convert QDK/Chemistry Structure to PySCF atom labels format.

    This function transforms a QDK/Chemistry Structure object into the format (a tuple) required by PySCF
    for molecular calculations. It extracts atomic information and formats it with unique labels
    for each atom.

    Args:
        structure: QDK/Chemistry Structure object containing molecular geometry and atomic information.

    Returns:
        A tuple containing three elements:

        - atoms : list[str]
            List of atom strings in PySCF format, where each string contains
            the atom label and its Cartesian coordinates (x, y, z) in Angstroms.
            Format: ``Symbol<count> x y z`` (e.g., ``H1 0.000000000000 0.000000000000 0.000000000000``).

        - pyscf_symbols : list[str]
            List of unique atom labels used in PySCF, where each atom of the same
            element is numbered sequentially (e.g., ``["H1", "H2", "O1"]``).

        - elements : list[str]
            List of atomic symbols without numbering, preserving the original
            element symbols from the structure (e.g., ``["H", "H", "O"]``).

    Note:
        - Coordinates are formatted with 12 decimal places for precision.
        - Each atom of the same element receives a unique numerical suffix starting from 1.
        - The function assumes atomic numbers do not exceed 200.

    Examples:
        >>> structure = Structure(...)  # Create or load a structure (coords in Bohr)
        >>> atoms, pyscf_symbols, elements = structure_to_pyscf_atom_labels(structure)
        >>> print(atoms[0])
        'H1 0.000000000000 0.757000000000 0.587000000000'

    """
    # Extract Structure
    elements = structure.get_atomic_symbols()
    coordinates = structure.get_coordinates()
    pyscf_symbols = []
    element_counts: Counter = Counter()
    natoms = len(elements)
    atoms: list[str] = []
    for i in range(natoms):
        symbol = elements[i]
        coords = coordinates[i]
        element_counts[symbol] += 1
        pyscf_symbols.append(f"{symbol}{element_counts[symbol]}")
        atoms.append(f"{symbol}{element_counts[symbol]} {coords[0]:.12f} {coords[1]:.12f} {coords[2]:.12f}")
    return atoms, pyscf_symbols, elements


def basis_to_pyscf_mol(basis: BasisSet) -> gto.Mole:
    """Convert QDK/Chemistry BasisSet instance to PySCF Mole object.

    This function extracts the structure and basis information from the QDK/Chemistry
    BasisSet instance and uses it to initialize a PySCF Mole object.

    Args:
        basis: QDK/Chemistry BasisSet instance with populated basis set.

    Returns:
        PySCF Mole object initialized with the QDK/Chemistry basis set data.

    Examples:
        >>> pyscf_mol = basis_to_pyscf_mol(basis)
        >>> print(pyscf_mol.atom)

    """
    atoms, pyscf_symbols, elements = structure_to_pyscf_atom_labels(basis.get_structure())
    natoms = len(atoms)
    # Copy the basis set from QDK/Chemistry to PySCF
    basis_dict = {}
    for i in range(natoms):
        atom_basis = []
        shells = basis.get_shells_for_atom(i)
        for shell in shells:
            shell_rec = f"{elements[i]:10}{str(shell.orbital_type)[-1]}\n"
            exponents = shell.exponents
            coefficients = shell.coefficients
            for j in range(len(exponents)):
                shell_rec += f"{exponents[j]:16.8f} {coefficients[j]:16.8f}\n"
            atom_basis.append(gto.parse(shell_rec))
        basis_dict[pyscf_symbols[i]] = atom_basis

    # TODO Handle Cartesian basis sets
    # https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41406
    mol = gto.Mole(atom=atoms, basis=basis_dict, unit="Bohr")

    # Store the original QDK/Chemistry basis name as an attribute for round-trip conversion
    mol.qdk_basis_name = basis.get_name()

    mol.build()

    return mol


def pyscf_mol_to_qdk_basis(pyscf_mol: gto.Mole, structure: Structure, basis_name: str | None = None) -> BasisSet:
    """Convert PySCF Mole object to QDK/Chemistry BasisSet instance.

    This function extracts the basis set information from a PySCF Mole object
    and returns a corresponding QDK/Chemistry BasisSet instance.

    Args:
        pyscf_mol: PySCF Mole object with basis set data.
        structure: QDK/Chemistry Structure instance that defines the atomic positions and types.
        basis_name: Name for the basis set. If None, attempts to derive from the PySCF
            molecule's basis set or defaults to "pyscf_basis".

    Returns:
        QDK/Chemistry BasisSet instance initialized with the PySCF basis set data.

    """
    # Determine the basis set name if not provided
    if basis_name is None:
        # Try to extract basis set name from stored QDK/Chemistry basis name (for round-trip conversion)
        if hasattr(pyscf_mol, "qdk_basis_name"):
            basis_name = pyscf_mol.qdk_basis_name
        # Try to extract basis set name from PySCF molecule
        elif hasattr(pyscf_mol, "basis") and isinstance(pyscf_mol.basis, str):
            basis_name = pyscf_mol.basis
        else:
            basis_name = "pyscf_basis"
    # Create shells from PySCF molecule data first
    shells = []
    # TODO: This should deduce the structure from the PySCF Mole object
    # For now, we'll collect the shells and pass structure to constructor
    # https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41407

    # TODO Handle Cartesian
    # https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41406
    atom_symbols = [pyscf_mol.atom_symbol(i) for i in range(pyscf_mol.natm)]
    for iatm in range(pyscf_mol.natm):
        atom_symbol = atom_symbols[iatm]
        for shell in pyscf_mol._basis[atom_symbol]:  # noqa: SLF001
            angular_momentum = shell[0]
            exponents = []
            coefficients = []
            for iprim in range(1, len(shell)):
                exponents.append(shell[iprim][0])
                coefficients.append(shell[iprim][1:])
            for j in range(len(coefficients[0])):
                j_coeffs = [coefficients[i][j] for i in range(len(coefficients))]
                # Create a shell and add it to the shells list
                qdk_shell = Shell(iatm, BasisSet.l_to_orbital_type(angular_momentum), exponents, j_coeffs)
                shells.append(qdk_shell)

    # Create BasisSet with name, shells, structure and basis type
    return BasisSet(basis_name, shells, structure, BasisType.Spherical)


def orbitals_to_scf(orbitals: Orbitals, occ_alpha: np.ndarray, occ_beta: np.ndarray, force_restricted: bool = False):
    """Convert an Orbitals object to a PySCF SCF object.

    This function takes a QDK/Chemistry Orbitals object and converts it into the appropriate
    PySCF self-consistent field (SCF) object based on the orbital characteristics.

    Args:
        orbitals: The QDK/Chemistry Orbitals object containing molecular orbital information including basis set,
            coefficients, occupations, and energies.
        occ_alpha: Occupation numbers for alpha (spin-up) electrons.
        occ_beta: Occupation numbers for beta (spin-down) electrons.
        force_restricted: If True, forces the creation of a restricted SCF object (RHF or ROHF) even if the orbitals
            are unrestricted. Default is False.

    Returns:
        A PySCF SCF object (RHF, ROHF, or UHF) populated with the molecular orbital data from the input ``Orbitals``
        object. The type of SCF object returned depends on:
            * RHF: for restricted closed-shell calculations
            * ROHF: for restricted open-shell calculations
            * UHF: for unrestricted calculations

    Note:
        The function automatically determines the appropriate SCF method based on whether
        the orbitals are restricted/unrestricted and closed-shell/open-shell.

    """
    mol = basis_to_pyscf_mol(orbitals.get_basis_set())
    coeff_a, coeff_b = orbitals.get_coefficients()
    # Get energies if available, otherwise use zero arrays as placeholders
    if orbitals.has_energies():
        energy_a, energy_b = orbitals.get_energies()
    else:
        # Energies not set (e.g., from rotated orbitals) - use zero arrays as placeholders
        nmos = orbitals.get_num_mos()
        energy_a = np.zeros(nmos)
        energy_b = np.zeros(nmos)

    if force_restricted or orbitals.is_restricted():
        # For restricted Orbitals, internal occupations are per-spin (each 0 or 1 for closed shell),
        # so total occupancy per MO is occ_a + occ_b
        total_occ = occ_alpha + occ_beta
        if np.any(occ_alpha != occ_beta):
            mf = scf.ROHF(mol)
            mf.mo_coeff = coeff_a
            mf.mo_energy = energy_a
            mf.mo_occ = total_occ
        else:
            mf = scf.RHF(mol)
            mf.mo_coeff = coeff_a
            mf.mo_energy = energy_a
            mf.mo_occ = total_occ
    else:
        mf = scf.UHF(mol)
        mf.mo_coeff = (coeff_a, coeff_b)
        mf.mo_energy = (energy_a, energy_b)
        mf.mo_occ = (occ_alpha, occ_beta)

    return mf


def orbitals_to_scf_from_n_electrons_and_multiplicity(
    orbitals: Orbitals,
    n_electrons: int,
    multiplicity: int = 1,
    force_restricted: bool = False,
):
    """Convert an Orbitals object to a PySCF SCF object.

    This is a convenience wrapper around :func:`orbtials_to_scf` that automatically constructs
    occupation arrays from the total number of electrons and spin multiplicity.

    Args:
        orbitals: The QDK/Chemistry Orbitals object containing molecular orbital information including basis set,
            coefficients, occupations, and energies.
        n_electrons: Total number of electrons in the system.
        multiplicity: Spin multiplicity (2S + 1), where S is the total spin. Default is 1 (singlet).
        force_restricted: If True, forces the creation of a restricted SCF object (RHF or ROHF) even if the orbitals
            are unrestricted. Default is False.

    Returns:
        A PySCF SCF object (RHF, ROHF, or UHF) populated with the molecular orbital data from the input ``Orbitals``
        object. The type of SCF object returned depends on:
            * RHF: for restricted closed-shell calculations
            * ROHF: for restricted open-shell calculations
            * UHF: for unrestricted calculations

    Raises:
        ValueError: If the electron count or multiplicity is invalid.

    Note:
        The function automatically determines the appropriate SCF method based on whether
        the orbitals are restricted/unrestricted and closed-shell/open-shell.

    """
    n_orbitals = orbitals.get_num_mos()
    alpha_occ, beta_occ = occupations_from_n_electrons_and_multiplicity(n_orbitals, n_electrons, multiplicity)

    return orbitals_to_scf(orbitals, alpha_occ, beta_occ, force_restricted)


def hamiltonian_to_scf(hamiltonian: Hamiltonian, alpha_occ: np.ndarray, beta_occ: np.ndarray) -> scf.RHF:
    """Convert QDK/Chemistry Hamiltonian to PySCF SCF object.

    This function creates a PySCF SCF object from a QDK/Chemistry Hamiltonian object, making it possible to use
    QDK/Chemistry Hamiltonian data with PySCF's post-HF methods such as Coupled Cluster. It extracts one- and two-body
    integrals, core energy, and electron counts from the Hamiltonian and configures them in a PySCF SCF object without
    performing an actual SCF calculation.

    Args:
        hamiltonian: QDK/Chemistry Hamiltonian object containing the electronic structure information including one- and
            two-body integrals, core energy, and orbital data.
        alpha_occ: Occupation numbers for alpha (spin-up) electrons.
        beta_occ: Occupation numbers for beta (spin-down) electrons.

    Returns:
        PySCF RHF object initialized with the Hamiltonian data, ready for post-HF calculations. This is a "fake" SCF
        object that provides the necessary interfaces for post-HF methods without having performed an SCF calculation.

    Raises:
        ValueError: If the Hamiltonian uses unsupported features like unrestricted orbitals, open-shell systems, or
        active spaces.

    Note:
        * Currently only supports restricted, closed-shell calculations without active spaces.
        * Future versions may add support for unrestricted and open-shell calculations.
        * The function creates a "fake" SCF object with the necessary interfaces for post-HF methods without actually
          performing an SCF calculation.
        * The returned SCF object contains dummy molecular orbitals and occupations suitable for post-HF method
          initialization.
        * For an interface using n_electrons and multiplicity, see
          :func:`hamiltonian_to_scf_from_n_electrons_and_multiplicity`.

    Examples:
        >>> import numpy as np
        >>> from qdk.chemistry.plugins.pyscf.utils import hamiltonian_to_scf
        >>> # Convert a QDK/Chemistry Hamiltonian to a PySCF SCF object
        >>> # Example for 10-electron system with 5 doubly occupied orbitals
        >>> norb = hamiltonian.get_orbitals().get_num_mos()
        >>> alpha_occ = np.zeros(norb)
        >>> beta_occ = np.zeros(norb)
        >>> alpha_occ[:5] = 1.0  # 5 alpha electrons
        >>> beta_occ[:5] = 1.0   # 5 beta electrons
        >>> pyscf_scf = hamiltonian_to_scf(hamiltonian, alpha_occ, beta_occ)
        >>> # Use with PySCF post-HF methods
        >>> from pyscf import cc
        >>> cc_calc = cc.CCSD(pyscf_scf)
        >>> cc_calc.kernel()

    """
    # Convenience aliases
    orbitals = hamiltonian.get_orbitals()
    norb = orbitals.get_num_mos()

    # Consistency checks
    if not orbitals.is_restricted():
        raise ValueError("Unrestricted is not supported.")
    if np.any(alpha_occ != beta_occ):
        raise ValueError("Open-shell is not supported.")
    if orbitals.has_active_space() and len(orbitals.get_active_space_indices()[0]) != orbitals.get_num_mos():
        raise ValueError("Active space is not supported.")

    # Dummy molecule
    mol = gto.M()

    # Calculate electron numbers from occupation arrays
    nalpha = int(np.sum(alpha_occ))
    nbeta = int(np.sum(beta_occ))

    # Create a fake SCF object
    # TODO: Handle unrestricted / open-shell
    fake_scf = scf.RHF(mol)
    fake_scf.mol.nelectron = nalpha + nbeta

    # Store integrals in the SCF object
    eri = hamiltonian.get_two_body_integrals()
    eri = np.reshape(eri, (norb, norb, norb, norb))
    h1e = hamiltonian.get_one_body_integrals()
    # Use _eri directly as it's the established way to access this in PySCF
    # even though it's technically a private member
    fake_scf._eri = eri  # noqa: SLF001
    fake_scf.get_hcore = lambda *_: h1e
    fake_scf.get_ovlp = lambda *_: np.eye(norb)
    fake_scf.energy_nuc = lambda *_: hamiltonian.get_core_energy()

    # Setup dummy MOs
    fake_scf.mo_coeff = np.eye(norb)
    fake_scf.mo_energy = np.diag(h1e)

    # Setup occupations from the provided arrays
    # For restricted calculations, PySCF expects total occupation (alpha + beta)
    fake_scf.mo_occ = alpha_occ + beta_occ

    return fake_scf


def hamiltonian_to_scf_from_n_electrons_and_multiplicity(
    hamiltonian: Hamiltonian,
    n_electrons: int,
    multiplicity: int = 1,
) -> scf.RHF:
    """Convert QDK/Chemistry Hamiltonian to PySCF SCF object using electron count and spin multiplicity.

    This is a convenience wrapper around :func:`hamiltonian_to_scf` that automatically constructs
    occupation arrays from the total number of electrons and spin multiplicity.

    Args:
        hamiltonian: QDK/Chemistry Hamiltonian object containing the electronic structure information.
        n_electrons: Total number of electrons in the system.
        multiplicity: Spin multiplicity (2S + 1), where S is the total spin. Default is 1 (singlet).

    Returns:
        PySCF RHF object initialized with the Hamiltonian data, ready for post-HF calculations.

    Raises:
        ValueError: If the electron count or multiplicity is invalid.

    Examples:
        >>> from qdk.chemistry.plugins.pyscf.utils import hamiltonian_to_scf_from_n_electrons_and_multiplicity
        >>> # Convert a QDK/Chemistry Hamiltonian to a PySCF SCF object
        >>> # Example for a 10-electron singlet system
        >>> pyscf_scf = hamiltonian_to_scf_from_n_electrons_and_multiplicity(
               hamiltonian, n_electrons=10, multiplicity=1
            )
        >>> # Use with PySCF post-HF methods
        >>> from pyscf import cc
        >>> cc_calc = cc.CCSD(pyscf_scf)
        >>> cc_calc.kernel()

    """
    n_orbitals = hamiltonian.get_orbitals().get_num_mos()
    alpha_occ, beta_occ = occupations_from_n_electrons_and_multiplicity(n_orbitals, n_electrons, multiplicity)

    return hamiltonian_to_scf(hamiltonian, alpha_occ, beta_occ)


def occupations_from_n_electrons_and_multiplicity(
    n_orbitals: int, n_electrons: int, multiplicity: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Convert total number of electrons and spin multiplicity to alpha and beta occupation arrays.

    Args:
        n_orbitals: Total number of molecular orbitals.
        n_electrons: Total number of electrons in the system.
        multiplicity: Spin multiplicity (2S + 1), where S is the total spin. Default is 1 (singlet).

    Returns:
        alpha_occ: Occupation numbers for alpha (spin-up) electrons.
        beta_occ: Occupation numbers for beta (spin-down) electrons.

    Raises:
        ValueError: If the total number of electrons or multiplicity is invalid.

    """
    # Validate inputs
    if n_electrons < 0:
        raise ValueError(f"The number of electrons must be non-negative, got {n_electrons}.")
    if multiplicity < 1:
        raise ValueError(f"The multiplicity must be at least 1, got {multiplicity}.")
    if n_electrons % 2 == 0 and multiplicity % 2 == 0:
        raise ValueError("An even number of electrons requires an odd multiplicity.")
    if n_electrons % 2 == 1 and multiplicity % 2 == 1:
        raise ValueError("An odd number of electrons requires an even multiplicity.")
    if n_electrons < multiplicity - 1:
        raise ValueError(f"A multiplicity of {multiplicity} requires more than {n_electrons} electrons.")

    # Calculate the number of singly and doubly occupied orbitals
    n_singly_occupied = multiplicity - 1
    n_doubly_occupied = (n_electrons - n_singly_occupied) // 2
    if n_singly_occupied + n_doubly_occupied > n_orbitals:
        raise ValueError(
            f"Not enough orbitals ({n_orbitals}) to accommodate {n_electrons} electrons with a multiplicity of "
            f"{multiplicity} ({n_singly_occupied + n_doubly_occupied} orbitals needed)."
        )

    # Construct occupation arrays
    alpha_occ = np.zeros(n_orbitals)
    beta_occ = np.zeros(n_orbitals)
    alpha_occ[: n_singly_occupied + n_doubly_occupied] = 1.0
    beta_occ[:n_doubly_occupied] = 1.0

    return alpha_occ, beta_occ
