"""Tests for MP2 energy validation against PySCF reference."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Ansatz, Structure

from .reference_tolerances import mp2_energy_tolerance


def create_o2_molecule(bond_length=2.3):
    """Create O2 molecule for testing.

    Args:
        bond_length: O-O bond length in Bohr atomic units.

    Returns:
        Structure: O2 molecule structure.

    """
    coords = np.array([[0.0, 0.0, 0.0], [bond_length, 0.0, 0.0]])
    atomic_numbers = [8, 8]
    return Structure(coords, atomic_numbers)


class TestMP2Validation:
    """Test class for MP2 energy validation against PySCF."""

    def test_o2_rmp2_energy_validation(self):
        """Test restricted MP2 energy for singlet O2 against PySCF."""
        pyscf_mp2_corr_energy = -0.38428662586339435
        # Create O2 molecule
        o2_molecule = create_o2_molecule(bond_length=2.3)

        # QDK Chemistry HF calculation
        qdk_scf_solver = create("scf_solver")
        qdk_scf_solver.settings().set("method", "hf")

        _, hf_wavefunction = qdk_scf_solver.run(o2_molecule, 0, 1, "cc-pvdz")
        qdk_orbitals = hf_wavefunction.get_orbitals()

        # Create Hamiltonian for energy evaluation
        ham_constructor = create("hamiltonian_constructor")
        qdk_hamiltonian = ham_constructor.run(qdk_orbitals)

        # Create ansatz from Hamiltonian and wavefunction
        ansatz = Ansatz(qdk_hamiltonian, hf_wavefunction)

        # QDK MP2 calculation
        mp2_calculator = create("dynamical_correlation_calculator", "qdk_mp2_calculator")
        qdk_mp2_total_energy, _ = mp2_calculator.run(ansatz)
        reference_energy = ansatz.calculate_energy()
        qdk_mp2_corr_energy = qdk_mp2_total_energy - reference_energy

        assert abs(qdk_mp2_corr_energy - pyscf_mp2_corr_energy) < mp2_energy_tolerance, (
            f"MP2 correlation energy mismatch: QDK={qdk_mp2_corr_energy:.8f}, "
            f"PySCF={pyscf_mp2_corr_energy:.8f}, "
            f"diff={abs(qdk_mp2_corr_energy - pyscf_mp2_corr_energy):.2e}"
        )

    def test_o2_ump2_energy_validation(self):
        """Test unrestricted MP2 energies for O2."""
        ref = -0.3509471012518759
        # Create O2 molecule
        o2_molecule = create_o2_molecule(bond_length=2.3)

        # QDK Chemistry UHF calculation
        qdk_scf_solver = create("scf_solver")
        qdk_scf_solver.settings().set("method", "hf")

        _, hf_wavefunction = qdk_scf_solver.run(o2_molecule, 0, 3, "cc-pvdz")
        qdk_orbitals = hf_wavefunction.get_orbitals()

        # Create Hamiltonian for energy evaluation
        ham_constructor = create("hamiltonian_constructor")
        qdk_hamiltonian = ham_constructor.run(qdk_orbitals)

        # Create ansatz and use MP2Calculator
        ansatz = Ansatz(qdk_hamiltonian, hf_wavefunction)

        # QDK UMP2 calculation
        mp2_calculator = create("dynamical_correlation_calculator", "qdk_mp2_calculator")
        qdk_ump2_total_energy, _ = mp2_calculator.run(ansatz)
        reference_energy = ansatz.calculate_energy()
        qdk_ump2_corr_energy = qdk_ump2_total_energy - reference_energy

        # Check energy equality for unrestricted
        assert abs(qdk_ump2_corr_energy - ref) < mp2_energy_tolerance, (
            f"Unrestricted MP2 correlation energy mismatch: QDK={qdk_ump2_corr_energy:.8f}, "
            f"Ref={ref:.8f}, "
            f"diff={abs(qdk_ump2_corr_energy - ref):.2e}"
        )
