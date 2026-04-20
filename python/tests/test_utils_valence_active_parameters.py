"""Tests for valence active space parameter utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Configuration, Orbitals, SlaterDeterminantContainer, Structure, Wavefunction
from qdk_chemistry.utils import compute_valence_space_parameters


def solve_wavefunction(structure, charge, multiplicity, basis="STO-3G"):
    """Run HF SCF and return the converged wavefunction for a given basis set."""
    scf_solver = create("scf_solver")
    _, wavefunction = scf_solver.run(structure, charge, multiplicity, basis)
    return wavefunction


class TestValenceParameters:
    """Tests for valence active space parameter computation."""

    def test_compute_valence_space_parameters(self):
        """Water molecule: 8 valence electrons, 6 valence orbitals."""
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
        """Truncated water wavefunction: valence orbitals capped by basis."""
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
        """Helium atom: 2 valence electrons, 1 valence orbital."""
        # Create a single Helium atom structure
        symbols = ["He"]
        coords = np.array([[0.000000, 0.000000, 0.000000]])  # He at origin
        helium = Structure(symbols, coords)

        wavefun_he = solve_wavefunction(helium, 0, 1)

        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_he, 0)

        assert num_active_electrons == 2  # number of valence electrons
        assert num_active_orbitals == 1  # number of valence orbitals

    def test_compute_valence_space_parameters_oxygen_hydrogen(self):
        """OH radical: 7 valence electrons, 5 valence orbitals."""
        # Create an Oxygen-Hydrogen molecule structure (OH)
        symbols = ["O", "H"]
        coords = np.array([[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000]])
        oh_molecule = Structure(symbols, coords)

        wavefun_oh = solve_wavefunction(oh_molecule, 0, 2)

        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_oh, 0)

        assert num_active_electrons == 7  # number of valence electrons
        assert num_active_orbitals == 5  # number of valence orbitals

    def test_compute_valence_space_parameters_positive_oxygen_hydrogen(self):
        """OH+ cation: charge reduces valence electrons."""
        symbols = ["O", "H"]
        coords = np.array([[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000]])
        oh_molecule = Structure(symbols, coords)

        wavefun_ohp = solve_wavefunction(oh_molecule, 1, 1)

        # Test with +1 charge
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_ohp, 1)

        assert num_active_electrons == 6  # number of valence electrons
        assert num_active_orbitals == 5  # number of valence orbitals

    def test_compute_valence_space_parameters_negative_oxygen_hydrogen(self):
        """OH- anion: negative charge increases valence electrons."""
        symbols = ["O", "H"]
        coords = np.array([[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000]])
        oh_molecule = Structure(symbols, coords)

        wavefun_ohn = solve_wavefunction(oh_molecule, -1, 1)

        # Test with -1 charge
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefun_ohn, -1)

        assert num_active_electrons == 8  # number of valence electrons
        assert num_active_orbitals == 5  # number of valence orbitals


class TestTransitionMetalValenceParameters:
    """Tests for the double-d-shell valence space in transition metals.

    Periods 4-5 include a correlating d' shell: 14 valence orbitals per TM atom
    (ns + 5*(n-1)d + 5*nd' + 3*np) instead of 9 (ns + 5*(n-1)d + 3*np).

    Note: These tests assume the valence orbital constants in
    valence_space.cpp have been updated accordingly.
    """

    def test_copper_atom_def2svp(self):
        """Cu atom (Z=29, period 4): 11 valence electrons, 14 valence orbitals."""
        symbols = ["Cu"]
        coords = np.array([[0.0, 0.0, 0.0]])
        structure = Structure(symbols, coords)

        wavefunction = solve_wavefunction(structure, 0, 2, "def2-svp")
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 11  # 29 - 18 (Ar core)
        assert num_active_orbitals == 14  # 4s + 5*3d + 5*4d' + 3*4p

    def test_nickel_atom_def2svp(self):
        """Ni atom (Z=28, period 4): 10 valence electrons, 14 valence orbitals."""
        symbols = ["Ni"]
        coords = np.array([[0.0, 0.0, 0.0]])
        structure = Structure(symbols, coords)

        wavefunction = solve_wavefunction(structure, 0, 3, "def2-svp")
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 10  # 28 - 18
        assert num_active_orbitals == 14

    def test_zinc_full_d_shell(self):
        """Zn (Z=30, period 4, d10): 12 valence electrons, 14 orbitals."""
        symbols = ["Zn"]
        coords = np.array([[0.0, 0.0, 0.0]])
        structure = Structure(symbols, coords)

        wavefunction = solve_wavefunction(structure, 0, 1, "def2-svp")
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 12  # 30 - 18
        assert num_active_orbitals == 14

    def test_silver_hydride(self):
        """AgH: Ag (Z=47, period 5) also gets 14 valence orbitals with double-d-shell."""
        symbols = ["Ag", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.617]])
        structure = Structure(symbols, coords)

        wavefunction = solve_wavefunction(structure, 0, 1, "def2-svp")
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 12  # 11 (Ag: 47-36) + 1 (H)
        assert num_active_orbitals == 15  # 14 (Ag) + 1 (H)

    def test_period3_unaffected(self):
        """Period 3 elements (Na, Cl) should still get 4 valence orbitals each."""
        symbols = ["Na", "Cl"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.361]])
        structure = Structure(symbols, coords)

        wavefunction = solve_wavefunction(structure, 0, 1)
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 8  # 1 (Na) + 7 (Cl)
        assert num_active_orbitals == 8  # 4 (Na) + 4 (Cl)

    def test_potassium_period4_main_group(self):
        """K (Z=19, period 4 main group): only 9 valence orbitals (no d' shell)."""
        symbols = ["K"]
        coords = np.array([[0.0, 0.0, 0.0]])
        structure = Structure(symbols, coords)

        wavefunction = solve_wavefunction(structure, 0, 2, "def2-svp")
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 1  # 19 - 18
        assert num_active_orbitals == 9  # period 4 main-group: no d' shell

    def test_platinum_period6_d_block(self):
        """Pt (Z=78, period 6 d-block): 24 valence electrons, 21 valence orbitals."""
        symbols = ["Pt"]
        coords = np.array([[0.0, 0.0, 0.0]])
        structure = Structure(symbols, coords)

        wavefunction = solve_wavefunction(structure, 0, 1, "def2-svp")
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 24  # 78 - 54 (Xe core)
        assert num_active_orbitals == 21  # 6s + 7*4f + 5*5d + 5*6d' + 3*6p
