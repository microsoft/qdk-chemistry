"""Utilities for converting between QDK/Chemistry and PySCF data structures.

This module provides conversion functions to bridge between QDK/Chemistry and PySCF data structures.
It enables integration between the two quantum chemistry libraries by handling the conversion of molecular structures,
basis sets, and Hamiltonians.

The main functionality includes:

* Converting QDK/Chemistry Structure objects to PySCF atom format.
* Converting QDK/Chemistry BasisSet objects to PySCF Mole objects.
* Converting PySCF Mole objects back to QDK/Chemistry BasisSet objects.
* Converting QDK/Chemistry Hamiltonian objects to PySCF SCF objects.

These utilities are essential for workflows that need to leverage both QDK/Chemistry's data management capabilities and
PySCF's quantum chemistry calculations.

Note:
    Currently supports spherical atomic orbitals only. Cartesian basis set support is planned for future versions,
    and the helper routines assume atomic numbers do not exceed 200.

Examples:
    >>> from qdk_chemistry.plugins.pyscf.conversion import structure_to_pyscf_atom_labels, basis_to_pyscf_mol
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
import pyscf

from qdk_chemistry.data import AOType, BasisSet, Hamiltonian, Orbitals, Shell, Structure
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.enum import CaseInsensitiveStrEnum

__all__ = [
    "basis_to_pyscf_mol",
    "hamiltonian_to_scf",
    "pyscf_mol_to_qdk_basis",
    "structure_to_pyscf_atom_labels",
]


class SCFType(CaseInsensitiveStrEnum):
    """Enum to specify the type of SCF calculation out of auto/restricted/unrestricted.

    Attributes:
        AUTO: Auto-detect based on restricted character of orbitals/hamiltonian
        RESTRICTED: Force restricted calculation
        UNRESTRICTED: Force unrestricted calculation

    """

    AUTO = "auto"
    RESTRICTED = "restricted"
    UNRESTRICTED = "unrestricted"


def structure_to_pyscf_atom_labels(structure: Structure) -> tuple:
    """Convert QDK/Chemistry Structure to PySCF atom labels format.

    This function transforms a QDK/Chemistry Structure object into the format (a tuple) required by PySCF
    for molecular calculations. It extracts atomic information and formats it with unique labels
    for each atom.

    Args:
        structure: QDK/Chemistry Structure object containing molecular geometry and atomic information.

    Returns:
        tuple[list[str], list[str], list[str]]: A tuple containing three lists:

        - atoms: List of atom strings in PySCF format, where each string contains
          the atom label and its Cartesian coordinates (x, y, z) in Angstroms.
          Format: ``Symbol<count> x y z`` (e.g., ``H1 0.000000000000 0.000000000000 0.000000000000``).
        - pyscf_symbols: List of unique atom labels used in PySCF, where each atom of the same
          element is numbered sequentially (e.g., ``["H1", "H2", "O1"]``).
        - elements: List of atomic symbols without numbering, preserving the original
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
    Logger.trace_entering()
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


def basis_to_pyscf_mol(basis: BasisSet, charge: int = 0, multiplicity: int = 1) -> pyscf.gto.mole.Mole:
    """Convert QDK/Chemistry BasisSet instance to PySCF Mole object.

    This function extracts the structure and basis information from the QDK/Chemistry
    BasisSet instance and uses it to initialize a PySCF Mole object. If the BasisSet
    contains ECP shells, they are converted to PySCF's ECP format.

    Args:
        basis: QDK/Chemistry BasisSet instance with populated basis set.
        charge: Total charge of the molecule (default: 0).
        multiplicity: Spin multiplicity (2S + 1) of the molecule (default: 1).

    Returns:
        PySCF Mole object initialized with the QDK/Chemistry basis set data,
        including ECP shells if present.

    Note:
        When ECP shells are present, the function reconstructs the full PySCF ECP
        structure from QDK's ECP shells with radial powers, preserving all
        exponents, coefficients, and r^n terms for each angular momentum channel.

    Examples:
        >>> pyscf_mol = basis_to_pyscf_mol(basis, charge=0, multiplicity=1)
        >>> print(pyscf_mol.atom)

    """
    Logger.trace_entering()
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
            atom_basis.append(pyscf.gto.parse(shell_rec))
        basis_dict[pyscf_symbols[i]] = atom_basis

    # TODO Handle Cartesian basis sets
    mol = pyscf.gto.mole.Mole(atom=atoms, basis=basis_dict, unit="Bohr", charge=charge, spin=multiplicity - 1)

    # Store the original QDK/Chemistry basis name as an attribute for round-trip conversion
    mol.qdk_basis_name = basis.get_name()

    # Handle ECP (Effective Core Potential) if present
    if basis.has_ecp_shells() and basis.has_ecp_electrons():
        # Build PySCF ECP structure from QDK ECP shells
        ecp_dict = {}
        ecp_electrons = basis.get_ecp_electrons()

        for iatm in range(natoms):
            ncore = ecp_electrons[iatm]

            if ncore > 0:
                # Get ECP shells for this atom
                ecp_shells_atom = basis.get_ecp_shells_for_atom(iatm)

                if ecp_shells_atom:
                    # Group shells by angular momentum: {l_value: {r_power: [(exp, coeff), ...]}}
                    shells_by_l: dict[int, dict[int, list[tuple[float, float]]]] = {}

                    for shell in ecp_shells_atom:
                        # Get l value from orbital type (OrbitalType enum values == l values)
                        l_value = int(shell.orbital_type)

                        if l_value not in shells_by_l:
                            shells_by_l[l_value] = {}

                        # Each primitive may have different r-power
                        for k in range(len(shell.exponents)):
                            r_power = int(shell.rpowers[k])
                            exp = float(shell.exponents[k])
                            coeff = float(shell.coefficients[k])

                            if r_power not in shells_by_l[l_value]:
                                shells_by_l[l_value][r_power] = []

                            shells_by_l[l_value][r_power].append((exp, coeff))

                    # Build PySCF format: [ncore, [[l, [term0, term1, ...]], ...]]
                    l_components = []
                    for l_value in sorted(shells_by_l.keys()):
                        # Find max r-power for this l to know array size
                        max_r = max(shells_by_l[l_value].keys())

                        # Build terms list with empty lists for unused r-powers
                        terms: list[list[tuple[float, float]]] = [[] for _ in range(max_r + 1)]
                        for r_power, primitives in shells_by_l[l_value].items():
                            terms[r_power] = primitives

                        l_components.append([l_value, terms])

                    # Store in ecp_dict using elements
                    ecp_dict[elements[iatm]] = [ncore, l_components]

        if ecp_dict:
            mol.ecp = ecp_dict
            # Store ECP name as attribute for roundtrip conversion
            mol.qdk_ecp_name = basis.get_ecp_name()
    elif basis.has_ecp_electrons():
        # Fallback: only ECP name available, no shells
        mol.ecp = basis.get_ecp_name()

    mol.build()

    return mol


def pyscf_mol_to_qdk_basis(
    pyscf_mol: pyscf.gto.mole.Mole, structure: Structure, basis_name: str | None = None
) -> BasisSet:
    """Convert PySCF Mole object to QDK/Chemistry BasisSet instance.

    This function extracts the basis set information from a PySCF Mole object
    and returns a corresponding QDK/Chemistry BasisSet instance. Both regular
    basis shells and ECP (Effective Core Potential) shells are extracted and
    converted.

    Args:
        pyscf_mol: PySCF Mole object with basis set data.
        structure: QDK/Chemistry Structure instance that defines the atomic positions and types.
        basis_name: Name for the basis set.

            If None, attempts to derive from the PySCF molecule's basis set or defaults to "pyscf_basis".

    Returns:
        QDK/Chemistry BasisSet instance initialized with the PySCF basis set data,
        including both regular shells and ECP shells.

    Note:
        ECP shells are extracted with their radial powers (r^n terms) preserved.
        Each combination of (atom, angular momentum, radial power) with non-empty
        primitives creates a separate ECP shell.

    """
    Logger.trace_entering()
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

    # TODO Handle Cartesian
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

    # Extract ECP shells if present
    ecp_shells = []
    if hasattr(pyscf_mol, "_ecp") and pyscf_mol._ecp:  # noqa: SLF001
        for iatm in range(pyscf_mol.natm):
            atom_symbol = atom_symbols[iatm]
            element = atom_symbol.rstrip("0123456789")
            if element in pyscf_mol._ecp:  # noqa: SLF001
                ecp_data = pyscf_mol._ecp[element]  # noqa: SLF001
                # Structure: [ncore, [[l, [[[exp, coeff]], ...]], ...]], where the inner structure has r-power terms
                ecp_components = ecp_data[1]

                for component in ecp_components:
                    l_value = component[0]
                    terms = component[1]

                    # Process each r-power term
                    for r_power, term in enumerate(terms):
                        if term:  # Skip empty terms
                            # Extract exponents and coefficients from [exp, coeff] pairs
                            exponents = [pair[0] for pair in term]
                            coefficients = [pair[1] for pair in term]
                            rpowers = [r_power] * len(exponents)

                            # Create ECP shell with radial powers
                            ecp_shell = Shell(
                                iatm, BasisSet.l_to_orbital_type(l_value), exponents, coefficients, rpowers
                            )
                            ecp_shells.append(ecp_shell)

    # Extract ECP name and electron counts if present
    if hasattr(pyscf_mol, "ecp") and pyscf_mol.ecp:
        ecp_name = "none"
        ecp_electrons = [0] * pyscf_mol.natm

        if isinstance(pyscf_mol.ecp, str):
            # Simple case: ECP specified as a uniform string name
            ecp_name = pyscf_mol.ecp

            # Extract electron counts from PySCF molecule
            if hasattr(pyscf_mol, "atom_nelec_core"):
                ecp_electrons = [pyscf_mol.atom_nelec_core(iatm) for iatm in range(pyscf_mol.natm)]
            else:
                raise RuntimeError("ECP electron counts could not be determined from PySCF Mole object.")

        elif isinstance(pyscf_mol.ecp, dict) and len(pyscf_mol.ecp) > 0:
            # Dictionary case: check if all values are strings (uniform ECP name)
            # or full structure [ncore, [[l, terms], ...]] from basis_to_pyscf_mol
            ecp_dict_values = list(pyscf_mol.ecp.values())
            first_value = ecp_dict_values[0]

            # Case 1: Dictionary with uniform string values (ECP names)
            if isinstance(first_value, str):
                # Check that all values are strings and identical
                if not all(isinstance(v, str) for v in ecp_dict_values):
                    raise ValueError("ECP dictionary contains mixed value types (strings and non-strings).")
                ecp_names_set = set(ecp_dict_values)
                if len(ecp_names_set) != 1:
                    raise NotImplementedError(f"Non-uniform ECP names are not supported: {ecp_names_set}.")
                ecp_name = next(iter(ecp_names_set))

                # Extract electron counts from PySCF molecule
                if hasattr(pyscf_mol, "atom_nelec_core"):
                    ecp_electrons = [pyscf_mol.atom_nelec_core(iatm) for iatm in range(pyscf_mol.natm)]
                else:
                    raise RuntimeError("ECP electron counts could not be determined from PySCF Mole object.")

            # Case 2: Dictionary with full ECP structure [ncore, [[l, terms], ...]]
            elif isinstance(first_value, list) and len(first_value) >= 2:
                # Verify all values have the expected structure
                for atom_sym, ecp_data in pyscf_mol.ecp.items():
                    if not isinstance(ecp_data, list) or len(ecp_data) < 2:
                        raise ValueError(
                            f"Invalid ECP structure for atom '{atom_sym}': "
                            f"expected [ncore, [[l, terms], ...]], got {type(ecp_data)}"
                        )

                # Extract ECP name from stored attribute if available (roundtrip case) otherwise use generic name
                ecp_name = pyscf_mol.qdk_ecp_name if hasattr(pyscf_mol, "qdk_ecp_name") else "custom"

                # Extract ncore values directly from the ECP dictionary structure
                for iatm in range(pyscf_mol.natm):
                    element = atom_symbols[iatm].rstrip("0123456789")
                    if element in pyscf_mol.ecp:
                        ecp_electrons[iatm] = pyscf_mol.ecp[element][0]

                # Validate consistency with mol.atom_nelec_core if available
                if hasattr(pyscf_mol, "atom_nelec_core"):
                    for iatm in range(pyscf_mol.natm):
                        mol_ncore = pyscf_mol.atom_nelec_core(iatm)
                        if ecp_electrons[iatm] != mol_ncore:
                            raise ValueError(
                                f"Inconsistent ECP electron count for atom {iatm}: "
                                f"ECP dict has {ecp_electrons[iatm]}, mol.atom_nelec_core has {mol_ncore}"
                            )
            else:
                raise ValueError(
                    f"Unsupported ECP dictionary value type: {type(first_value)}. "
                    "Expected uniform strings or [ncore, [[l, terms], ...]] structure."
                )
        else:
            raise ValueError(f"PySCF ECP data must be a string or dict, got {type(pyscf_mol.ecp)}.")

        # Create BasisSet with name, shells, ecp_name, ecp_shells, ecp_electrons, structure, and basis type
        if any(n > 0 for n in ecp_electrons):
            return BasisSet(basis_name, shells, ecp_name, ecp_shells, ecp_electrons, structure, AOType.Spherical)

    # Create BasisSet with name, shells, ecp_shells, structure, and basis type
    return BasisSet(basis_name, shells, ecp_shells, structure, AOType.Spherical)


def orbitals_to_scf(
    orbitals: Orbitals,
    occ_alpha: np.ndarray,
    occ_beta: np.ndarray,
    scf_type: SCFType | str = SCFType.AUTO,
    method: str = "hf",
):
    """Convert an Orbitals object to a PySCF SCF object.

    This function takes a QDK/Chemistry Orbitals object and converts it into the appropriate
    PySCF self-consistent field (SCF) object based on the orbital characteristics.

    Args:
        orbitals: The QDK/Chemistry Orbitals object containing molecular orbital information.

            Includes basis set, coefficients, occupations, and energies.

        occ_alpha: Occupation numbers for alpha (spin-up) electrons.
        occ_beta: Occupation numbers for beta (spin-down) electrons.
        scf_type: Type of SCF calculation to create. Can be:
            * ``"auto"`` or ``SCFType.AUTO`` (default): Automatically detect based on ``orbitals.is_restricted()``
            * ``"restricted"`` or ``SCFType.RESTRICTED``: Force restricted calculation (RHF or ROHF)
            * ``"unrestricted"`` or ``SCFType.UNRESTRICTED``: Force unrestricted calculation (UHF)

        method: The electronic structure method to use. Default is "hf" (Hartree-Fock).
            Any other value is treated as a DFT exchange-correlation functional (e.g., "b3lyp", "pbe").

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
    Logger.trace_entering()
    if isinstance(scf_type, str):
        scf_type = SCFType(scf_type.lower())

    # n_electrons and multiplicity from occupations
    n_alpha = int(np.sum(occ_alpha))
    n_beta = int(np.sum(occ_beta))
    n_electrons = n_alpha + n_beta
    multiplicity = abs(n_alpha - n_beta) + 1

    # Calculate charge from structure's atomic symbols (accounting for ECPs)
    basis_set = orbitals.get_basis_set()
    atomic_symbols = basis_set.get_structure().get_atomic_symbols()
    neutral_electrons = sum(pyscf.gto.charge(symbol) for symbol in atomic_symbols)
    if basis_set.has_ecp_electrons():
        neutral_electrons -= sum(basis_set.get_ecp_electrons())
    charge = neutral_electrons - n_electrons

    mol = basis_to_pyscf_mol(basis_set, charge=charge, multiplicity=multiplicity)

    coeff_a, coeff_b = orbitals.get_coefficients()
    # Get energies if available, otherwise use zero arrays as placeholders
    if orbitals.has_energies():
        energy_a, energy_b = orbitals.get_energies()
    else:
        # Energies not set (e.g., from rotated orbitals) - use zero arrays as placeholders
        num_molecular_orbitals = orbitals.get_num_molecular_orbitals()
        energy_a = np.zeros(num_molecular_orbitals)
        energy_b = np.zeros(num_molecular_orbitals)

    if scf_type == SCFType.RESTRICTED or (scf_type == SCFType.AUTO and orbitals.is_restricted()):
        # For restricted Orbitals, internal occupations are per-spin (each 0 or 1 for closed shell),
        # so total occupancy per MO is occ_a + occ_b
        total_occ = occ_alpha + occ_beta
        if np.any(occ_alpha != occ_beta):
            # Restricted open-shell
            if method.lower() == "hf":
                mf = pyscf.scf.ROHF(mol)
            else:
                mf = pyscf.scf.ROKS(mol)
                mf.xc = method
            mf.mo_coeff = coeff_a
            mf.mo_energy = energy_a
            mf.mo_occ = total_occ
        else:
            # Restricted closed-shell
            if method.lower() == "hf":
                mf = pyscf.scf.RHF(mol)
            else:
                mf = pyscf.scf.RKS(mol)
                mf.xc = method
            mf.mo_coeff = coeff_a
            mf.mo_energy = energy_a
            mf.mo_occ = total_occ
    else:
        # Unrestricted
        if method.lower() == "hf":
            mf = pyscf.scf.UHF(mol)
        else:
            mf = pyscf.scf.UKS(mol)
            mf.xc = method
        mf.mo_coeff = (coeff_a, coeff_b)
        mf.mo_energy = (energy_a, energy_b)
        mf.mo_occ = (occ_alpha, occ_beta)

    return mf


def orbitals_to_scf_from_n_electrons_and_multiplicity(
    orbitals: Orbitals,
    n_electrons: int,
    multiplicity: int = 1,
    scf_type: SCFType | str = SCFType.AUTO,
    method: str = "hf",
):
    """Convert an Orbitals object to a PySCF SCF object.

    This is a convenience wrapper around :func:`orbitals_to_scf` that automatically constructs
    occupation arrays from the total number of electrons and spin multiplicity.

    Args:
        orbitals: The QDK/Chemistry Orbitals object containing molecular orbital information.

            Includes basis set, coefficients, occupations, and energies.

        n_electrons: Total number of electrons in the system.
        multiplicity: Spin multiplicity (2S + 1), where S is the total spin. Default is 1 (singlet).
        scf_type: Type of SCF calculation to create. Can be:
            * ``"auto"`` or ``SCFType.AUTO`` (default): Automatically detect based on ``orbitals.is_restricted()``
            * ``"restricted"`` or ``SCFType.RESTRICTED``: Force restricted calculation (RHF or ROHF)
            * ``"unrestricted"`` or ``SCFType.UNRESTRICTED``: Force unrestricted calculation (UHF)

        method: The electronic structure method to use. Default is "hf" (Hartree-Fock).
            Any other value is treated as a DFT exchange-correlation functional (e.g., "b3lyp", "pbe").

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
    Logger.trace_entering()
    n_orbitals = orbitals.get_num_molecular_orbitals()
    alpha_occ, beta_occ = occupations_from_n_electrons_and_multiplicity(n_orbitals, n_electrons, multiplicity)

    return orbitals_to_scf(orbitals, alpha_occ, beta_occ, scf_type, method)


def hamiltonian_to_scf(hamiltonian: Hamiltonian, alpha_occ: np.ndarray, beta_occ: np.ndarray) -> pyscf.scf.RHF:
    """Convert QDK/Chemistry Hamiltonian to PySCF SCF object.

    This function creates a PySCF SCF object from a QDK/Chemistry Hamiltonian object, making it possible to use
    QDK/Chemistry Hamiltonian data with PySCF's post-HF methods such as Coupled Cluster. It extracts one- and two-body
    integrals, core energy, and electron counts from the Hamiltonian and configures them in a PySCF SCF object without
    performing an actual SCF calculation.

    Args:
        hamiltonian: QDK/Chemistry Hamiltonian object.

            Contains the electronic structure information including one- and two-body integrals,
            core energy, and orbital data.

        alpha_occ: Occupation numbers for alpha (spin-up) electrons.
        beta_occ: Occupation numbers for beta (spin-down) electrons.

    Returns:
        PySCF RHF object initialized with the Hamiltonian data, ready for post-HF calculations. This is a "fake" SCF
        object that provides the necessary interfaces for post-HF methods without having performed an SCF calculation.

    Raises:
        ValueError: If the Hamiltonian uses unsupported features like model Hamiltonian with unrestricted orbitals,
            open-shell systems, or active spaces.

    Note:
        * This function is intended for (restricted) model hamiltonian usage, since the orbitals are not used directly.
        * If a non-model Hamiltonian is passed, this function automatically re-routes to orbitals_to_scf.
        * Active spaces are not supported.
        * The function creates a "fake" SCF object with the necessary interfaces for post-HF methods without actually
          performing an SCF calculation.
        * The returned SCF object contains dummy molecular orbitals and occupations suitable for post-HF method
          initialization.
        * For an interface using n_electrons and multiplicity, see
          ``hamiltonian_to_scf_from_n_electrons_and_multiplicity``.

    Examples:
        >>> import numpy as np
        >>> from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf
        >>> # Convert a QDK/Chemistry Hamiltonian to a PySCF SCF object
        >>> # Example for 10-electron system with 5 doubly occupied orbitals
        >>> norb = hamiltonian.get_orbitals().get_num_molecular_orbitals()
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
    Logger.trace_entering()
    orbitals = hamiltonian.get_orbitals()
    try:
        orbitals.get_coefficients()
        # is not a model hamiltonian - reroute to orbitals_to_scf
        return orbitals_to_scf(orbitals, occ_alpha=alpha_occ, occ_beta=beta_occ)
    except RuntimeError:
        if hamiltonian.is_unrestricted():
            raise ValueError("You cannot pass an unrestricted model Hamiltonian here.") from None

    norb = orbitals.get_num_molecular_orbitals()

    # Consistency checks
    if np.any(alpha_occ != beta_occ):
        raise ValueError("Open-shell is not supported.")
    if (
        orbitals.has_active_space()
        and len(orbitals.get_active_space_indices()[0]) != orbitals.get_num_molecular_orbitals()
    ):
        raise ValueError("Active space is not supported.")

    # Dummy molecule
    mol = pyscf.gto.M()

    # Calculate electron numbers from occupation arrays
    num_alpha = int(np.sum(alpha_occ))
    num_beta = int(np.sum(beta_occ))

    # Create a fake SCF object
    fake_scf = pyscf.scf.RHF(mol)
    fake_scf.mol.nelectron = num_alpha + num_beta

    # Store integrals in the SCF object
    (eri, _, _) = hamiltonian.get_two_body_integrals()
    eri = np.reshape(eri, (norb, norb, norb, norb))
    (h1e, _) = hamiltonian.get_one_body_integrals()
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
) -> pyscf.scf.RHF:
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
        >>> from qdk_chemistry.plugins.pyscf.conversion import hamiltonian_to_scf_from_n_electrons_and_multiplicity
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
    Logger.trace_entering()
    n_orbitals = hamiltonian.get_orbitals().get_num_molecular_orbitals()
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

    Returns: tuple including
        - alpha_occ: Occupation numbers for alpha (spin-up) electrons.
        - beta_occ: Occupation numbers for beta (spin-down) electrons.

    Raises:
        ValueError: If the total number of electrons or multiplicity is invalid.

    """
    Logger.trace_entering()
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
