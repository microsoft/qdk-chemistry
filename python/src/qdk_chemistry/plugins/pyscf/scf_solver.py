"""PySCF-based Self-Consistent Field (SCF) solver implementation for qdk_chemistry.

This module provides integration between QDK/Chemistry and PySCF for performing
Self-Consistent Field calculations supporting both Hartree-Fock (HF) and
Density Functional Theory (DFT) methods. It implements solvers that can be
used within the QDK/Chemistry framework for electronic structure calculations.

The module contains:

* :class:`PyscfScfSettings`: Configuration class for SCF calculation parameters
* :class:`PyscfScfSolver`: Main solver class that performs HF and DFT calculations
* Registration utilities to integrate the solver with QDK/Chemistry's plugin system

The solver handles automatic conversion between QDK/Chemistry molecular structures and PySCF
format, performs the SCF calculation using the specified method (HF or DFT functional),
and returns results (energy and orbitals) in QDK/Chemistry-compatible format. It supports
various basis sets and exchange-correlation functionals through the settings interface.

Upon import, this module automatically registers the PySCF solver with QDK/Chemistry's
SCF solver registry under the name "pyscf".

>>> from qdk_chemistry.plugins.pyscf.scf_solver import PyscfScfSolver
>>> solver = PyscfScfSolver()
>>> solver.settings()["method"] = "b3lyp"  # Use B3LYP DFT
>>> energy, orbitals = solver.run(molecule)

This module requires both QDK/Chemistry and PySCF to be installed. The solver supports both
restricted and unrestricted variants for both HF and DFT calculations.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings

import numpy as np
from pyscf import gto, scf
from pyscf.gto.mole import bse_predefined_ecp

from qdk_chemistry.algorithms import ScfSolver, register
from qdk_chemistry.data import (
    Configuration,
    ElectronicStructureSettings,
    Orbitals,
    SlaterDeterminantContainer,
    Structure,
    Wavefunction,
)
from qdk_chemistry.plugins.pyscf.utils import orbitals_to_scf, pyscf_mol_to_qdk_basis, structure_to_pyscf_atom_labels


class PyscfScfSettings(ElectronicStructureSettings):
    """Settings configuration for the PySCF SCF solver.

    This class manages the configuration parameters for the PySCF Self-Consistent
    Field (SCF) solver, inheriting common electronic structure defaults from
    ElectronicStructureSettings and adding PySCF-specific customizations.

    Inherits from ElectronicStructureSettings:
        method (str, default="hf"): The electronic structure method (Hartree-Fock).
        basis_set (str, default="def2-svp"): The basis set used for quantum chemistry calculations.
            Common options include "def2-svp", "def2-tzvp", "cc-pvdz", etc.
        tolerance (float, default=1e-6): Convergence tolerance.
        max_iterations (int, default=50): Maximum number of iterations.
        reference_type (str, default="auto"): Reference type selection - "auto", "restricted", or "unrestricted".
            "auto" uses restricted for closed-shell and unrestricted for open-shell systems.
            "restricted" uses ROHF/ROKS (restricted open-shell Hartree-Fock/Kohn-Sham) for open-shell systems.
            "unrestricted" always uses UHF/UKS.

    Examples:
        >>> settings = PyscfScfSettings()
        >>> settings.get("basis_set")
        'def2-svp'
        >>> settings.set("basis_set", "cc-pvdz")
        >>> settings.get("basis_set")
        'cc-pvdz'

    Notes:
        The PySCF SCF solver is used for performing self-consistent field calculations in quantum chemistry, which
        are fundamental for determining electronic structure and molecular properties.

    """

    def __init__(self):
        """Initialize the settings with default values from ElectronicStructureSettingsplus PySCF-specific defaults."""
        super().__init__()  # This sets up all the base class defaults


class PyscfScfSolver(ScfSolver):
    """PySCF-based Self-Consistent Field (SCF) solver for quantum chemistry calculations.

    This class provides an interface between QDK/Chemistry and PySCF for performing both Hartree-Fock (HF) and Density
    Functional Theory (DFT) calculations on molecular systems. It handles the conversion between QDK/Chemistry
    structuresand PySCF molecular representations, performs the SCF calculation, and returns the results in
    QDK/Chemistry-compatible format.

    The solver supports:

    * Hartree-Fock (HF): method="hf" uses RHF/UHF/ROHF
    * DFT methods: any other method string is treated as an XC functional (e.g., "b3lyp", "pbe", "m06")

    The solver automatically selects restricted/unrestricted variants based on the
    reference_type setting, returning electronic energy (excluding nuclear
    repulsion) along with molecular orbitals information.

    Examples:
        >>> solver = PyscfScfSolver()
        >>> solver.settings()["basis_set"] = "6-31G*"
        >>> solver.settings()["method"] = "b3lyp"  # Use B3LYP DFT
        >>> energy, orbitals = solver.run(molecule_structure, charge=0, spin_multiplicity=1)
        >>> print(f"Electronic energy: {energy} Hartree")

    See Also:
        ScfSolver : Base class for SCF solvers
        PyscfScfSettings : Settings configuration for PySCF calculations

    """

    def __init__(self):
        """Initialize the PySCF SCF solver with default settings."""
        super().__init__()
        self._settings = PyscfScfSettings()

    def _run_impl(
        self, structure: Structure, charge: int, spin_multiplicity: int, initial_guess: Orbitals | None = None
    ) -> tuple[float, Wavefunction]:
        """Perform a self-consistent field (SCF) calculation using PySCF.

        This method converts the input structure to PySCF format, runs either a Hartree-Fock (HF) or Density Functional
        Theory (DFT) calculation based on the method setting, and returns the electronic energy and molecular orbitals
        in QDK/Chemistry format.

        The method automatically selects:
        * HF variants (RHF/UHF/ROHF) when method="hf"
        * DFT variants (RKS/UKS/ROKS) for any other method string, treating it as an exchange-correlation functional

        Args:
            structure: The molecular or crystal structure to be calculated. Must be compatible with the
                ``structure_to_pyscf_atom_labels`` conversion function.
            charge: The total charge of the molecular system.
                Note: This parameter is not used directly; the charge from ``self._settings`` is used instead.
            spin_multiplicity: The spin multiplicity :math:`(2S+1)` of the system, where :math:`S` is the total spin.
                Note: This parameter is not used directly; the spin_multiplicity from ``self._settings`` is used
                instead.
            initial_guess: Initial orbital guess for the SCF calculation. If provided, the
                molecular orbital coefficients will be used as the starting point
                for the SCF iteration. If None, PySCF's default initial guess will be used.

        Returns:
            * The electronic energy of the system in atomic units (Hartree), excluding nuclear repulsion energy
                for consistency with qdk_chemistry.
            * A single-determinant Slater determinant wavefunction representing the Hartree-Fock ground state.

        Note:
            The calculation uses the basis set and method specified in the settings. For DFT calculations, the method
            string should be a valid PySCF XC functional name (e.g., "b3lyp", "pbe", "m06", "wb97x-d").

        """
        atoms, _, _ = structure_to_pyscf_atom_labels(structure)
        basis_name = self._settings["basis_set"]
        method = self._settings["method"].lower()
        tolerance = self._settings["tolerance"]
        max_iterations = self._settings["max_iterations"]

        # The PySCF convention is 2S not 2S+1
        multiplicity = spin_multiplicity
        spin = (multiplicity - 1) if multiplicity > 0 else None
        open_shell = multiplicity != 1

        # Determine reference type and unrestricted flag
        reference_type = self._settings["reference_type"].lower()

        if reference_type == "auto":
            unrestricted = open_shell
        elif reference_type == "unrestricted":
            unrestricted = True
            if not open_shell and initial_guess is None:
                warnings.warn(
                    "Unrestricted reference requested for closed-shell system. "
                    "Automatic spin symmetry breaking is not supported. "
                    "Consider providing a spin-broken initial guess if desired.",
                    stacklevel=2,
                )
        elif reference_type == "restricted":
            unrestricted = False
            # Note: For open-shell with restricted, we'll use ROHF/ROKS which PySCF supports
        else:
            raise ValueError("reference_type must be one of: auto, restricted, unrestricted")

        # Check effective core potentials (ECPs)
        ecp, ecp_atoms = bse_predefined_ecp(basis_name, atoms)
        if ecp_atoms:
            ecp_dict = dict.fromkeys(ecp_atoms, ecp)

        mol = gto.Mole(
            atom=atoms,
            basis=basis_name,
            charge=charge,
            spin=spin,
            unit="Bohr",
            ecp=ecp_dict if ecp_atoms else None,
        )
        mol.build()

        # Select the appropriate SCF method based on the method setting and reference type
        if method == "hf":
            # Hartree-Fock methods
            if unrestricted:
                mf = scf.UHF(mol)
            elif not open_shell:
                mf = scf.RHF(mol)
            else:
                # Restricted open-shell Hartree-Fock
                mf = scf.ROHF(mol)
        # DFT methods (Kohn-Sham)
        elif unrestricted:
            mf = scf.UKS(mol)
            mf.xc = method  # Set the exchange-correlation functional
        elif not open_shell:
            mf = scf.RKS(mol)
            mf.xc = method  # Set the exchange-correlation functional
        else:
            # Restricted open-shell Kohn-Sham
            mf = scf.ROKS(mol)
            mf.xc = method

        # Configure convergence settings
        mf.conv_tol = tolerance
        mf.max_cycle = max_iterations

        # Set initial guess if provided
        if initial_guess is not None and hasattr(initial_guess, "get_coefficients"):
            # Validate initial guess compatibility with reference type
            initial_guess_is_unrestricted = initial_guess.is_unrestricted()
            if unrestricted and not initial_guess_is_unrestricted:
                warnings.warn(
                    "Unrestricted calculation requested but restricted initial guess provided. ",
                    stacklevel=2,
                )
            if not unrestricted and initial_guess_is_unrestricted:
                raise ValueError("Restricted calculation requested but unrestricted initial guess provided. ")

            # Create occupation arrays based on electron configuration
            norb = initial_guess.get_num_molecular_orbitals()
            num_alpha = mol.nelec[0]  # Number of alpha electrons
            num_beta = mol.nelec[1]  # Number of beta electrons

            occ_alpha = np.array([1.0 if i < num_alpha else 0.0 for i in range(norb)])
            occ_beta = np.array([1.0 if i < num_beta else 0.0 for i in range(norb)])

            # Use utility function to convert qdk-chemistry orbitals to PySCF format
            temp_mf = orbitals_to_scf(initial_guess, occ_alpha, occ_beta, not unrestricted)

            # Extract density matrix from the temporary SCF object
            dm0 = temp_mf.make_rdm1()

            # Run SCF with initial density matrix
            energy = mf.kernel(dm0=dm0)
        else:
            # No initial guess provided, use default
            energy = mf.kernel()

        basis_set = pyscf_mol_to_qdk_basis(mf.mol, structure, basis_name)
        _ovlp = mf.get_ovlp()

        if not unrestricted and mol.spin == 0:
            orbitals = Orbitals(
                mf.mo_coeff,
                mf.mo_energy,
                ao_overlap=_ovlp,
                basis_set=basis_set,
            )
        elif not unrestricted:
            # Restricted open-shell (ROHF/ROKS) - create unrestricted-style orbitals with same coefficients
            orbitals = Orbitals(
                mf.mo_coeff,  # Same coefficients for alpha and beta
                mf.mo_coeff,
                mf.mo_energy,  # Same energies for alpha and beta
                mf.mo_energy,
                ao_overlap=_ovlp,
                basis_set=basis_set,
            )
        elif unrestricted:
            energy_a, energy_b = mf.mo_energy
            coeff_a, coeff_b = mf.mo_coeff

            # Unrestricted case - create Orbitals with alpha/beta arguments
            orbitals = Orbitals(
                coeff_a,
                coeff_b,
                energy_a,
                energy_b,
                ao_overlap=_ovlp,
                basis_set=basis_set,
            )
        else:
            # Create Orbitals with restricted arguments (coeffs, occupations, energies)
            orbitals = Orbitals(
                mf.mo_coeff,
                mf.mo_energy,
                ao_overlap=_ovlp,
                basis_set=basis_set,
            )

        # Create Slater determinant wavefunction
        n_alpha = mol.nelec[0]  # Number of alpha electrons
        n_beta = mol.nelec[1]  # Number of beta electrons
        n_orbitals = orbitals.get_num_molecular_orbitals()

        # Create HF ground state configuration using canonical method
        hf_config = Configuration.canonical_hf_configuration(n_alpha, n_beta, n_orbitals)

        wfn = Wavefunction(SlaterDeterminantContainer(hf_config, orbitals))

        return energy, wfn

    def name(self) -> str:
        """Return the name of the SCF solver."""
        return "pyscf"


register(lambda: PyscfScfSolver())
