"""Tests for MCSCF functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry import algorithms as alg
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Structure

from .reference_tolerances import mcscf_energy_tolerance


def create_n2_structure():
    """Create a nitrogen molecule structure."""
    symbols = ["N", "N"]
    coords = np.array([[0.0, 0.0, 2.0 * ANGSTROM_TO_BOHR], [0.0, 0.0, 0.0]])
    return Structure(symbols, coords)


def create_o2_structure():
    """Create an oxygen molecule structure."""
    symbols = ["O", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.208 * ANGSTROM_TO_BOHR]])
    return Structure(symbols, coords)


class TestMCSCF:
    """Test class MCSCF functionality."""

    def test_n2_6_6_ccpvdz_casscf(self):
        """Test PySCF MCSCF for n2 with cc-pvdz basis and CAS(6,6)."""
        # Create N2 structure
        n2 = create_n2_structure()

        # Perform SCF calculation with QDK
        scf_solver = alg.create("scf_solver")
        scf_solver.settings().set("basis_set", "cc-pvdz")
        _, wavefunction = scf_solver.run(n2, 0, 1)

        # Construct QDK Hamiltonian for active space
        ham_calculator = alg.create("hamiltonian_constructor")

        # Create MACIS calculator
        macis_calc = alg.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Select active space
        valence_selector = alg.create("active_space_selector", "qdk_valence")
        valence_selector.settings().set("num_active_electrons", 6)
        valence_selector.settings().set("num_active_orbitals", 6)
        active_orbitals_sd = valence_selector.run(wavefunction)

        # Calculate with QDK/MACIS
        mcscf = alg.create("multi_configuration_scf", "pyscf")
        mcscf_energy, _ = mcscf.run(active_orbitals_sd.get_orbitals(), ham_calculator, macis_calc, 3, 3)

        assert np.isclose(mcscf_energy, -108.78966139913287, atol=mcscf_energy_tolerance)

    def test_o2_6_6_ccpvdz_casscf_triplet(self):
        """Test PySCF MCSCF for o2 with cc-pvdz basis and CAS(6,6)."""
        # Create O2 structure
        o2 = create_o2_structure()

        # Perform SCF calculation with PySCF
        scf_solver = alg.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "cc-pvdz")
        scf_solver.settings().set("force_restricted", True)
        _, wavefunction = scf_solver.run(o2, 0, 3)

        # Construct QDK Hamiltonian for active space
        ham_calculator = alg.create("hamiltonian_constructor")

        # Create MACIS calculator
        macis_calc = alg.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)
        macis_calc.settings().set("ci_residual_tolerance", 1e-10)

        # Select active space: 6 orbitals, 6 electrons
        valence_selector = alg.create("active_space_selector", "qdk_valence")
        valence_selector.settings().set("num_active_electrons", 6)
        valence_selector.settings().set("num_active_orbitals", 6)
        active_orbitals_sd = valence_selector.run(wavefunction)

        # Calculate with QDK/MACIS
        mcscf = alg.create("multi_configuration_scf", "pyscf")
        mcscf_energy, _ = mcscf.run(active_orbitals_sd.get_orbitals(), ham_calculator, macis_calc, 4, 2)

        assert np.isclose(mcscf_energy, -149.68131616317658, atol=mcscf_energy_tolerance)
