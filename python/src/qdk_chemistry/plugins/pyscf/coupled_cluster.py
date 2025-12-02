"""PySCF-based Coupled Cluster calculator implementation for qdk_chemistry.

This module provides integration between QDK/Chemistry and PySCF for performing
Coupled Cluster (CC) calculations. It implements a CCSD (Coupled Cluster Singles and Doubles)
calculator that can be used within the QDK/Chemistry framework for accurate electronic structure
calculations.

The module contains:

* :class:`PyscfCoupledClusterSettings`: Configuration class for CC calculation parameters
* :class:`PyscfCoupledClusterCalculator`: Main calculator class that performs CC calculations
* Registration utilities to integrate the calculator with QDK/Chemistry's plugin system

The calculator handles automatic conversion between QDK/Chemistry Hamiltonian objects and PySCF
format, performs the CC calculation, and returns results (energy and
excitation amplitudes) in QDK/Chemistry-compatible format.

Upon import, this module automatically registers the PySCF CC calculator with QDK/Chemistry's
CC calculator registry under the name "pyscf".

Examples:
    >>> from qdk_chemistry.plugins.pyscf.coupled_cluster import PyscfCoupledClusterCalculator
    >>> cc_calculator = PyscfCoupledClusterCalculator()
    >>> energy, amplitudes = cc_calculator.calculate(hamiltonian)

This module requires both QDK/Chemistry and PySCF to be installed.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from pyscf import cc

from qdk_chemistry.algorithms import CoupledClusterCalculator, register
from qdk_chemistry.data import CoupledClusterAmplitudes, Settings
from qdk_chemistry.plugins.pyscf.utils import hamiltonian_to_scf

__all__ = ["PyscfCoupledClusterCalculator", "PyscfCoupledClusterSettings"]


class PyscfCoupledClusterSettings(Settings):
    """Settings configuration for the PySCF Coupled Cluster calculator.

    This class manages the configuration parameters for the PySCF Coupled Cluster
    calculator, providing default settings and allowing customization of
    quantum chemistry calculations.

    Attributes:
        verbose (int): Print level. Default value equals to ``Mole.verbose``.
        max_memory (float or int): Allowed memory in MB. Default value equals to ``Mole.max_memory``.
        conv_tol (float): Convergence threshold. Default is 1e-7.
        conv_tol_norm (float): Convergence threshold for norm(t1,t2). Default is 1e-5.
        max_cycle (int): Max number of iterations. Default is 50.
        diis_space (int): DIIS space size. Default is 6.
        diis_start_cycle (int): The step to start DIIS. Default is 0.
        direct (bool): AO-direct CCSD. Default is False.
        async_io (bool): Allow for asynchronous function execution. Default is True.
        incore_complete (bool): Avoid all I/O. Default is True.
        frozen (int or list): If integer is given, the inner-most orbitals are frozen from CC amplitudes.
            Given the orbital indices (0-based) in a list, both occupied and virtual orbitals can be frozen in the
            CC calculation.

    Examples:
        >>> settings = PyscfCoupledClusterSettings()
        >>> settings.set("max_cycle", 100)

    """

    def __init__(self):
        """Initialize the settings with default values."""
        super().__init__()
        self._set_default("conv_tol", "double", 1e-7)
        self._set_default("conv_tol_normt", "double", 1e-5)
        self._set_default("max_cycle", "int", 50)
        self._set_default("diis_space", "int", 6)
        self._set_default("diis_start_cycle", "int", 0)
        self._set_default("direct", "bool", False)
        self._set_default("async_io", "bool", True)
        self._set_default("incore_complete", "bool", True)


class PyscfCoupledClusterCalculator(CoupledClusterCalculator):
    """PySCF-based Coupled Cluster calculator for quantum chemistry calculations.

    This class provides an interface between QDK/Chemistry and PySCF for performing
    Coupled Cluster calculations on molecular systems. It handles the conversion
    between QDK/Chemistry Hamiltonian objects and PySCF representations, performs the
    CC calculation, and returns the results in QDK/Chemistry-compatible format.

    The calculator uses CCSD (Coupled Cluster Singles and Doubles) method.

    """

    def __init__(self):
        """Initialize the calculator with default settings."""
        super().__init__()
        self._settings = PyscfCoupledClusterSettings()

    def _run_impl(self, ansatz):
        """Perform a Coupled Cluster calculation using PySCF.

        This method takes a QDK/Chemistry Hamiltonian object, converts it to the PySCF format,
        performs a CCSD calculation, and returns the results as a pair of energy and
        QDK/Chemistry CoupledClusterAmplitudes object.

        Args:
            ansatz: The QDK/Chemistry :class:`Ansatz` object representing the system to calculate.

        Returns:
            A tuple containing the total energy and a QDK/Chemistry CoupledClusterAmplitudes
            object containing the excitation amplitudes.

        Raises:
            RuntimeError: If the CCSD calculation does not converge.

        """
        hamiltonian = ansatz.get_hamiltonian()
        alpha_occ, beta_occ = ansatz.get_wavefunction().get_total_orbital_occupations()
        pyscf_scf = hamiltonian_to_scf(hamiltonian, alpha_occ, beta_occ)
        pyscf_cc = cc.CCSD(pyscf_scf)
        # Handle Settings
        if self._settings.get_or_default("verbose", -1) != -1:
            pyscf_cc.verbose = int(self._settings.get("verbose"))
        if self._settings.get_or_default("max_memory", -1) != -1:
            pyscf_cc.max_memory = int(self._settings.get("max_memory"))
        if self._settings.get_or_default("frozen", -1) != -1:
            pyscf_cc.frozen = self._settings.get("frozen")
        pyscf_cc.conv_tol = float(self._settings.get("conv_tol"))
        pyscf_cc.conv_tol_normt = float(self._settings.get("conv_tol_normt"))
        pyscf_cc.max_cycle = int(self._settings.get("max_cycle"))
        pyscf_cc.diis_space = int(self._settings.get("diis_space"))
        pyscf_cc.diis_start_cycle = int(self._settings.get("diis_start_cycle"))
        pyscf_cc.direct = bool(self._settings.get("direct"))
        pyscf_cc.async_io = bool(self._settings.get("async_io"))
        pyscf_cc.incore_complete = bool(self._settings.get("incore_complete"))

        pyscf_cc.kernel()
        if not pyscf_cc.converged:
            raise RuntimeError("CCSD calculation did not converge")
        t1 = pyscf_cc.t1
        t2 = pyscf_cc.t2
        t1 = np.reshape(t1, (t1.size, 1))
        t2 = np.reshape(t2, (t2.size, 1))

        # Create amplitudes object without correlation energy
        nalpha, nbeta = ansatz.get_wavefunction().get_total_num_electrons()
        qdk_cc_amplitudes = CoupledClusterAmplitudes(hamiltonian.get_orbitals(), t1, t2, nalpha, nbeta)

        # Total energy = SCF energy + correlation energy
        total_energy = pyscf_scf.e_tot + pyscf_cc.e_corr

        return total_energy, qdk_cc_amplitudes

    def name(self) -> str:
        """Return the name of the Coupled Cluster calculator."""
        return "pyscf"


register(lambda: PyscfCoupledClusterCalculator())
