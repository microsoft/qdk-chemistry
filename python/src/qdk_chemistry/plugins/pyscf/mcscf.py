"""PySCF-based Multi-Configurational Self-Consistent Field (MCSCF) solver implementation for qdk_chemistry.

This module integrates QDK/Chemistry and PySCF to perform Multi-Configurational Self-Consistent
Field (MCSCF) calculations.
It implements a wrapper that adapts a QDK multi-configuration calculator to PySCF's FCI solver interface,
allowing it to be used in PySCF's MCSCF framework.

The module contains:

* :class:`PyscfMcscfSettings`: Configuration class for MCSCF calculation parameters
* :class:`PyscfMcscfCalculator`: Main calculator class that performs CASSCF/MCSCF calculations
* Registration utilities to integrate the calculator with QDK/Chemistry's plugin system
* Utilities to convert QDK multi-configuration calculators to PySCF FCI solvers

Upon import, this module automatically registers the PySCF MCSCF calculator with QDK/Chemistry's
MCSCF calculator registry under the name "pyscf".

Examples
--------
>>> from qdk_chemistry.plugins.pyscf.mcscf import PyscfMcscfCalculator
>>> mcscf = PyscfMcscfCalculator()
>>> energy, wfn = mcscf.solve(
...     orbitals,
...     hamiltonian_constructor,
...     multi_configuration_calculator,
...     n_active_alpha_electrons,
...     n_active_beta_electrons,
... )
>>> print(f"MCSCF energy: {energy} Hartree")

This module requires both QDK/Chemistry and PySCF to be installed.

Notes
-----
* Restricted orbitals with identical alpha/beta active and inactive spaces are required.
* The Hamiltonian constructor parameter is currently unused but included for interface consistency.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import numpy as np
from pyscf import ao2mo, gto, mcscf

from qdk_chemistry.algorithms import (
    MultiConfigurationCalculator,
    MultiConfigurationScf,
)
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    CasWavefunctionContainer,
    Hamiltonian,
    ModelOrbitals,
    Orbitals,
    SciWavefunctionContainer,
    Settings,
    Wavefunction,
)
from qdk_chemistry.plugins.pyscf.conversion import SCFType, orbitals_to_scf
from qdk_chemistry.utils import Logger

__all__ = ["PyscfMcscfCalculator", "PyscfMcscfSettings"]


class _QdkMcSolverWrapper:
    """Wrapper class to make a QDK MultiConfigurationCalculator compatible with PySCF FCI solver interface.

    This class adapts a QDK MultiConfigurationCalculator to work as a PySCF FCI solver
    that can be used in CASCI and CASSCF calculations. It provides the necessary interface
    methods that PySCF expects from an FCI solver.

    """

    def __init__(self, mol: gto.Mole, mc_calculator: MultiConfigurationCalculator) -> None:
        """Initialize the _QdkMcSolverWrapper.

        Args:
            mol: PySCF molecule object (required by PySCF FCI interface).
            mc_calculator: The QDK multi-configurational calculator to wrap.

        """
        Logger.trace_entering()
        self.mol = mol
        self.mc_calculator = mc_calculator
        self.wavefunction = None

    def kernel(
        self,
        h1e: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: np.ndarray | None = None,  # noqa: ARG002
        ecore: float = 0,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[float, np.ndarray]:
        """Run the FCI calculation.

        This method is called by PySCF's CASCI/CASSCF to solve the CI problem.

        Args:
            h1e: One-body integrals in the active space.
            eri: Two-body integrals in the active space.
            norb: Number of active orbitals.
            nelec: Number of electrons in active space. Can be total electrons or (alpha, beta) tuple.
            ci0: Initial CI guess vector.
            ecore: Core energy contribution.
            kwargs: Additional keyword arguments.

        Returns:
            A tuple containing the total energy and CI coefficients.

        Raises:
            RuntimeError: If the wavefunction is not returned from the MC calculator.

        """
        Logger.trace_entering()
        # Handle nelec format (can be int or tuple)
        if isinstance(nelec, tuple | list):
            n_alpha, n_beta = nelec
        else:
            # If total electrons given, assume equal spin
            n_beta = nelec // 2
            n_alpha = nelec - n_beta

        # Create ModelOrbitals for the active space and use real orbitals only after the calculation
        orbitals = ModelOrbitals(norb, True)

        # eri needs to be completely filled and then flattened to a vector for QDK
        eri = ao2mo.restore(1, eri, norb)
        eri_qdk = eri.flatten()

        # Create empty inactive Fock matrix (no inactive orbitals in active space)
        inactive_fock = np.zeros((0, 0))

        # Create QDK Hamiltonian
        hamiltonian = Hamiltonian(CanonicalFourCenterHamiltonianContainer(h1e, eri_qdk, orbitals, ecore, inactive_fock))

        # Run the multi-configurational calculation
        result = self.mc_calculator.run(hamiltonian, n_alpha, n_beta)
        energy, self.wavefunction = result
        if not self.wavefunction:
            raise RuntimeError("Wavefunction not returned from MC calculator.")

        # get coeffs
        coeffs = None
        if self.wavefunction.get_container_type() == "cas" or self.wavefunction.get_container_type() == "sci":
            coeffs = self.wavefunction.get_container().get_coefficients()
        else:
            raise RuntimeError(
                f"Unsupported wavefunction type ({self.wavefunction.get_container_type()}) for FCI solver wrapper."
            )

        return energy, coeffs

    def make_rdm1(
        self,
        fcivec: Any,
        norb: int,
        nelec: int,
        link_index: Any = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute 1-particle reduced density matrix.

        Args:
            fcivec: CI vector/coefficients (not used, kept for PySCF compatibility).
            norb: Number of orbitals.
            nelec: Number of electrons.
            link_index: Link indices (PySCF internal parameter).
            kwargs: Additional keyword arguments.

        Returns:
            1-particle reduced density matrix.

        """
        Logger.trace_entering()
        return self.make_rdm12(fcivec, norb, nelec, **kwargs)[0]

    def make_rdm12(
        self,
        fcivec: Any,  # noqa: ARG002
        ncas: int,
        nelec: int | tuple[int, int],
        link_index: Any = None,  # noqa: ARG002
        reorder: bool = True,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 1-particle and 2-particle reduced density matrices.

        Args:
            fcivec: CI vector/coefficients (not used, kept for PySCF compatibility).
            ncas: Number of CAS orbitals.
            nelec: Number of electrons (can be total count or (alpha, beta) tuple).
            link_index: Link indices (PySCF internal parameter).
            reorder: Whether to reorder the density matrices.
            kwargs: Additional keyword arguments.

        Returns:
            A tuple containing (rdm1, rdm2) where rdm1 is the 1-particle
                reduced density matrix and rdm2 is the 2-particle reduced density matrix.

        Raises:
            ValueError: If wavefunction is not available (kernel() must be run first).

        """
        Logger.trace_entering()
        if self.wavefunction is None:
            raise ValueError("Wavefunction not available. Run kernel() first.")

        if not isinstance(nelec, int):
            nelec = sum(nelec)  # Convert (n_alpha, n_beta) to total electrons

        two_rdm = self.wavefunction.get_active_two_rdm_spin_traced()
        # convert the RDM from QDK to PySCF format and scale by 2 due to convention in QDK
        two_rdm = np.reshape(two_rdm, (ncas, ncas, ncas, ncas))
        one_rdm = self.wavefunction.get_active_one_rdm_spin_traced()
        return one_rdm, two_rdm


def _mcsolver_to_fcisolver(mol: Any, mc_calculator: MultiConfigurationCalculator) -> _QdkMcSolverWrapper:
    """Convert a QDK MultiConfigurationCalculator to a PySCF-compatible FCI solver.

    This function creates a wrapper that adapts a QDK MultiConfigurationCalculator
    to the PySCF FCI solver interface, allowing it to be used in PySCF's CASCI and
    CASSCF calculations.

    Args:
        mol: PySCF molecule object (required by PySCF interface).
        mc_calculator: The QDK multi-configurational calculator to convert.

    Returns:
        A PySCF-compatible FCI solver wrapper.

    Examples:
        >>> from qdk_chemistry import algorithms
        >>> from pyscf import gto
        >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.5', basis='sto-3g')
        >>> mc_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        >>> fci_solver = _mcsolver_to_fcisolver(mol, mc_calc)
        >>> # Now fci_solver can be used with PySCF CASSCF
        >>> from pyscf import mcscf
        >>> casscf = mcscf.CASSCF(mf, 2, 2)
        >>> casscf.fcisolver = fci_solver

    Note:
        This is a bridge function that enables QDK MC calculators to be used within
        PySCF's MCSCF framework. The wrapper handles the conversion of data formats
        and calling conventions between the two libraries.

    """
    Logger.trace_entering()
    return _QdkMcSolverWrapper(mol, mc_calculator)


class PyscfMcscfSettings(Settings):
    """Configuration settings for PySCF MCSCF calculations.

    This class extends the base Settings class to provide specific configuration
    options for Multi-Configurational Self-Consistent Field calculations using PySCF.
    It includes MCSCF-specific parameters in addition to the standard electronic
    structure calculation settings.

    """

    def __init__(self):
        """Initialize the settings with default values from ElectronicStructureSettings plus MCSCF-specific defaults."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("max_cycle_macro", "int", 50)
        self._set_default("verbose", "int", 0)


class PyscfMcscfCalculator(MultiConfigurationScf):
    """PySCF-based MCSCF calculator for quantum chemistry calculations.

    This class provides an interface between QDK and PySCF for performing
    Multi-Configurational Self-Consistent Field calculations on molecular systems.
    It handles the conversion between QDK Hamiltonian objects and PySCF
    representations, performs the MCSCF calculation, and returns the results
    in QDK-compatible format.

    The calculator uses CASSCF (Complete Active Space SCF) method.

    """

    def __init__(self):
        """Initialize the calculator with default settings."""
        Logger.trace_entering()
        super().__init__()
        self._settings = PyscfMcscfSettings()

    def _run_impl(
        self,
        orbitals: Any,
        ham_ctor: Any,  # noqa: ARG002
        mc_calculator: Any,
        n_active_alpha_electrons: int,
        n_active_beta_electrons: int,
    ) -> tuple[float, Any]:
        """Perform a Multi-Configurational Self-Consistent Field calculation using PySCF.

        This method takes QDK/Chemistry orbitals, converts them to the PySCF format,
        performs a CASSCF calculation using the QDK MC calculator as the FCI solver,
        and returns the results as a pair of energy and QDK Wavefunction object.

        Args:
            orbitals: The QDK Orbitals object containing molecular orbital information.
            ham_ctor: Hamiltonian constructor (not used, kept for interface compatibility).
            mc_calculator: The multi-configurational calculator for handling CI calculations.
            n_active_alpha_electrons: The number of alpha electrons in the active space.
            n_active_beta_electrons: The number of beta electrons in the active space.

        Returns:
            A tuple containing the total MCSCF energy and a QDK Wavefunction
                object containing the optimized orbitals and CI coefficients.

        Raises:
            ValueError: If the orbitals don't meet the requirements (restricted, identical active/inactive spaces).
            RuntimeError: If the MCSCF calculation does not converge.

        """
        Logger.trace_entering()
        # check that alpha and beta active space indices are the same
        if orbitals.get_active_space_indices()[0] != orbitals.get_active_space_indices()[1]:
            raise ValueError("MCSCF implementation only supports identical active spaces for alpha and beta electrons.")
        if orbitals.get_inactive_space_indices()[0] != orbitals.get_inactive_space_indices()[1]:
            raise ValueError(
                "MCSCF implementation only supports identical inactive spaces for alpha and beta electrons."
            )
        if not orbitals.is_restricted():
            raise ValueError("MCSCF implementation only supports restricted orbitals.")

        # Get orbital information from hamiltonian
        active_indices = orbitals.get_active_space_indices()[0]  # Get alpha indices (same for restricted)
        n_active_orbitals = len(active_indices)
        n_active_electrons = n_active_alpha_electrons + n_active_beta_electrons

        # fake alpha and beta occupations with number of electrons in active space
        # occupied orbitals are doubly occupied
        alpha_occ = [0] * orbitals.get_num_molecular_orbitals()
        beta_occ = [0] * orbitals.get_num_molecular_orbitals()
        for i in orbitals.get_inactive_space_indices()[0]:
            alpha_occ[i] = 1
            beta_occ[i] = 1
        for i in active_indices:
            if n_active_alpha_electrons > 0:
                alpha_occ[i] = 1
                n_active_alpha_electrons -= 1
            if n_active_beta_electrons > 0:
                beta_occ[i] = 1
                n_active_beta_electrons -= 1

        # get pyscf scf object
        pyscf_scf = orbitals_to_scf(orbitals, occ_alpha=alpha_occ, occ_beta=beta_occ, scf_type=SCFType.RESTRICTED)

        # Create CASSCF object
        pyscf_mcscf = mcscf.CASSCF(pyscf_scf, n_active_orbitals, n_active_electrons)
        mc_calculator.settings().set("calculate_one_rdm", True)
        mc_calculator.settings().set("calculate_two_rdm", True)
        pyscf_mcscf.fcisolver = _mcsolver_to_fcisolver(
            pyscf_scf.mol,
            mc_calculator,
        )

        # shuffle indices for pyscf and add 1 for one based
        pyscf_active_indices = [i + 1 for i in active_indices]
        mo = pyscf_mcscf.sort_mo(pyscf_active_indices)

        # Apply settings
        self._apply_settings(pyscf_mcscf)

        # Run CASSCF calculation
        pyscf_mcscf.verbose = self._settings.get("verbose")
        energy = pyscf_mcscf.kernel(mo)[0]

        if not pyscf_mcscf.converged:
            raise RuntimeError("MCSCF calculation did not converge")

        return energy, self._make_wavefunction(pyscf_mcscf, orbitals)

    def _make_wavefunction(self, pyscf_mcscf: Any, orbitals: Orbitals) -> Wavefunction:
        """Convert PySCF MCSCF result to QDK Wavefunction object.

        This method extracts the optimized orbitals and CI coefficients from
        the PySCF MCSCF object and constructs a QDK Wavefunction object.

        Args:
            pyscf_mcscf: The PySCF CASSCF object after the calculation.
            orbitals : The QDK orbitals object

        Returns:
            A QDK Wavefunction object containing the optimized orbitals and CI coefficients.

        Raises:
            RuntimeError: If the wavefunction type is not supported.

        """
        Logger.trace_entering()
        # Extract basis set and overlap from PySCF object
        _ovlp = pyscf_mcscf._scf.get_ovlp()  # noqa: SLF001

        # Create Orbitals with restricted arguments (coeffs, occupations, energies)
        wfn = pyscf_mcscf.fcisolver.wavefunction

        orbitals = Orbitals(
            pyscf_mcscf.mo_coeff,
            pyscf_mcscf.mo_energy,
            ao_overlap=_ovlp,
            basis_set=orbitals.get_basis_set(),
            indices=(orbitals.get_active_space_indices()[0], orbitals.get_inactive_space_indices()[0]),
        )

        container = None

        # try to get spin-dependent RDMs, fall back to spin-traced if not available
        use_spin_traced = False
        try:
            one_rdm_aa, one_rdm_bb = wfn.get_active_one_rdm_spin_dependent()
        except RuntimeError:
            use_spin_traced = True
        try:
            two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb = wfn.get_active_two_rdm_spin_dependent()
        except RuntimeError:
            use_spin_traced = True

        if wfn.get_container_type() == "cas":
            if use_spin_traced:
                container = CasWavefunctionContainer(
                    wfn.get_container().get_coefficients(),
                    wfn.get_active_determinants(),
                    orbitals,
                    one_rdm_spin_traced=wfn.get_active_one_rdm_spin_traced(),
                    two_rdm_spin_traced=wfn.get_active_two_rdm_spin_traced(),
                    type=wfn.get_type(),
                )
            else:
                container = CasWavefunctionContainer(
                    wfn.get_container().get_coefficients(),
                    wfn.get_active_determinants(),
                    orbitals,
                    one_rdm_spin_traced=None,
                    one_rdm_aa=one_rdm_aa,
                    one_rdm_bb=one_rdm_bb,
                    two_rdm_spin_traced=None,
                    two_rdm_aabb=two_rdm_aabb,
                    two_rdm_aaaa=two_rdm_aaaa,
                    two_rdm_bbbb=two_rdm_bbbb,
                    type=wfn.get_type(),
                )
        elif wfn.get_container_type() == "sci":
            if use_spin_traced:
                container = SciWavefunctionContainer(
                    wfn.get_container().get_coefficients(),
                    wfn.get_active_determinants(),
                    orbitals,
                    one_rdm_spin_traced=wfn.get_active_one_rdm_spin_traced(),
                    two_rdm_spin_traced=wfn.get_active_two_rdm_spin_traced(),
                    type=wfn.get_type(),
                )
            else:
                container = SciWavefunctionContainer(
                    wfn.get_container().get_coefficients(),
                    wfn.get_active_determinants(),
                    orbitals,
                    one_rdm_spin_traced=None,
                    one_rdm_aa=one_rdm_aa,
                    one_rdm_bb=one_rdm_bb,
                    two_rdm_spin_traced=None,
                    two_rdm_aabb=two_rdm_aabb,
                    two_rdm_aaaa=two_rdm_aaaa,
                    two_rdm_bbbb=two_rdm_bbbb,
                    type=wfn.get_type(),
                )
        else:
            raise RuntimeError(f"Unsupported wavefunction type ({wfn.get_container_type()}) for FCI solver wrapper.")
        return Wavefunction(container)

    def _apply_settings(self, pyscf_mcscf: Any) -> None:
        """Apply QDK settings to PySCF MCSCF object."""
        # Apply core MCSCF settings
        pyscf_mcscf.max_cycle_macro = int(self._settings.get("max_cycle_macro"))

    def name(self) -> str:
        """Return the name of the MCSCF solver."""
        Logger.trace_entering()
        return "pyscf"
