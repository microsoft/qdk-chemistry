"""Tests for valence active space parameter utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Configuration, Orbitals, SlaterDeterminantContainer, Structure, Wavefunction
from qdk_chemistry.utils import compute_valence_space_parameters


def solve_wavefunction(structure, charge, multiplicity):
    scf_solver = create("scf_solver")
    _, wavefunction = scf_solver.run(structure, charge, multiplicity, "STO-3G")
    return wavefunction


class TestValenceParameters:
    def test_compute_valence_space_parameters(self):
        symbols = ["O", "H", "H"]
        coords = np.array(
            [[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000], [-0.757000, 0.586000, 0.000000]]
        )
        water = Structure(symbols, coords)
        wavefunction_sd = solve_wavefunction(water, 0, 1)

        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction_sd, 0)
        assert num_active_electrons == 8  # number of valence electrons
        assert num_active_orbitals == 6  # number of valence orbitals

    def test_compute_valence_space_parameters_truncated(self):
        symbols = ["O", "H", "H"]
        coords = np.array(
            [[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000], [-0.757000, 0.586000, 0.000000]]
        )
        water = Structure(symbols, coords)
        wavefunction_sd = solve_wavefunction(water, 0, 1)

        det_truncated = Configuration("22222")
        initial_orbitals = wavefunction_sd.get_orbitals()
        basis_set = initial_orbitals.get_basis_set()
        coeffs_truncated = np.eye(10, 5)
        orbitals_truncated = Orbitals(coeffs_truncated, None, None, basis_set)
        container_truncated = SlaterDeterminantContainer(det_truncated, orbitals_truncated)
        wavefun_truncated = Wavefunction(container_truncated)

        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_truncated, 0)
        assert num_active_electrons == 8  # number of valence electrons
        assert num_active_orbitals == 4  # number of valence orbitals

    def test_compute_valence_space_parameters_helium(self):
        # Create a single Helium atom structure
        symbols = ["He"]
        coords = np.array([[0.000000, 0.000000, 0.000000]])  # He at origin
        helium = Structure(symbols, coords)

        wavefun_he = solve_wavefunction(helium, 0, 1)

        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_he, 0)

        assert num_active_electrons == 2  # number of valence electrons
        assert num_active_orbitals == 1  # number of valence orbitals

    def test_compute_valence_space_parameters_oxygen_hydrogen(self):
        # Create an Oxygen-Hydrogen molecule structure (OH)
        symbols = ["O", "H"]
        coords = np.array([[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000]])
        oh_molecule = Structure(symbols, coords)

        wavefun_oh = solve_wavefunction(oh_molecule, 0, 2)

        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_oh, 0)

        assert num_active_electrons == 7  # number of valence electrons
        assert num_active_orbitals == 5  # number of valence orbitals

    def test_compute_valence_space_parameters_positive_oxygen_hydrogen(self):
        symbols = ["O", "H"]
        coords = np.array([[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000]])
        oh_molecule = Structure(symbols, coords)

        wavefun_ohp = solve_wavefunction(oh_molecule, 1, 1)

        # Test with +1 charge
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_ohp, 1)

        assert num_active_electrons == 6  # number of valence electrons
        assert num_active_orbitals == 5  # number of valence orbitals

    def test_compute_valence_space_parameters_negative_oxygen_hydrogen(self):
        symbols = ["O", "H"]
        coords = np.array([[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000]])
        oh_molecule = Structure(symbols, coords)

        wavefun_ohn = solve_wavefunction(oh_molecule, -1, 1)

        # Test with -1 charge
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_ohn, -1)

        assert num_active_electrons == 8  # number of valence electrons
        assert num_active_orbitals == 5  # number of valence orbitals
