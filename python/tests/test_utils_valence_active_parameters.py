"""Tests for valence active space parameter utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    AOType,
    BasisSet,
    Configuration,
    Orbitals,
    OrbitalType,
    Shell,
    SlaterDeterminantContainer,
    Structure,
    Wavefunction,
)
from qdk_chemistry.utils import compute_valence_space_parameters


def solve_wavefunction(structure, charge, multiplicity, basis="STO-3G"):
    """Run HF SCF and return the converged wavefunction for a given basis set."""
    scf_solver = create("scf_solver")
    _, wavefunction = scf_solver.run(structure, charge, multiplicity, basis)
    return wavefunction


def make_minimal_wavefunction(symbols, coords, n_alpha, n_beta, num_molecular_orbitals):
    """Build a minimal Wavefunction directly, without running SCF.

    ``compute_valence_space_parameters`` only consults the structure (atom
    elements), the total electron count, and ``num_molecular_orbitals``. We
    therefore wrap a dummy single-s-shell BasisSet (so it validates), an
    orbital coefficient matrix with the documented
    ``(num_atomic_orbitals, num_molecular_orbitals)`` shape, and a
    Configuration string that yields exactly ``(n_alpha, n_beta)`` electrons.
    This avoids basis set dependencies and SCF convergence flakiness in tests
    that only validate counting logic.
    """
    structure = Structure(symbols, np.asarray(coords, dtype=float))

    exps = np.array([1.0])
    coefs = np.array([1.0])
    shells = [Shell(i, OrbitalType.S, exps, coefs) for i in range(len(symbols))]
    basis_set = BasisSet("dummy", shells, structure, AOType.Spherical)

    # Coefficients are (num_atomic_orbitals, num_molecular_orbitals); their
    # values are unused by compute_valence_space_parameters, only the shape
    # matters here.
    num_atomic_orbitals = basis_set.get_num_atomic_orbitals()
    coeffs = np.zeros((num_atomic_orbitals, num_molecular_orbitals))
    orbitals = Orbitals(coeffs, None, None, basis_set)

    pair_count = min(n_alpha, n_beta)
    chars = ["0"] * num_molecular_orbitals
    for i in range(pair_count):
        chars[i] = "2"
    if n_alpha > n_beta:
        for i in range(n_alpha - n_beta):
            chars[pair_count + i] = "u"
    elif n_beta > n_alpha:
        for i in range(n_beta - n_alpha):
            chars[pair_count + i] = "d"
    config = Configuration("".join(chars))
    container = SlaterDeterminantContainer(config, orbitals)
    return Wavefunction(container)


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
    """Tests for the optional double-d-shell valence space in transition metals.

    When ``include_double_d_shell=True``, periods 4-6 d-block elements (Sc-Zn,
    Y-Cd, Hf-Hg) get a correlating d' shell: 14 valence orbitals per period
    4-5 TM atom (ns + 5*(n-1)d + 5*nd' + 3*np) instead of 9, and 21 valence
    orbitals per period 6 TM atom (6s + 7*4f + 5*5d + 5*6d' + 3*6p) instead
    of 16.

    These tests only validate the valence-space sizing logic in
    ``compute_valence_space_parameters``; that function only consults the
    structure, the total electron count, and ``num_molecular_orbitals``. We
    therefore build a minimal Wavefunction directly via
    :func:`make_minimal_wavefunction` and skip SCF entirely. This keeps the
    suite fast and removes any dependence on a particular basis set being
    available.
    """

    def test_copper_atom(self):
        """Cu atom (Z=29, period 4): 11 valence electrons, 14 valence orbitals."""
        # 29 electrons doublet: 15 alpha + 14 beta. 50 MOs >> num_core + 14.
        wavefunction = make_minimal_wavefunction(["Cu"], [[0.0, 0.0, 0.0]], 15, 14, 50)
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(
            wavefunction, 0, include_double_d_shell=True
        )

        assert num_active_electrons == 11  # 29 - 18 (Ar core)
        assert num_active_orbitals == 14  # 4s + 5*3d + 5*4d' + 3*4p

    def test_nickel_atom(self):
        """Ni atom (Z=28, period 4): 10 valence electrons, 14 valence orbitals."""
        # 28 electrons triplet: 15 alpha + 13 beta.
        wavefunction = make_minimal_wavefunction(["Ni"], [[0.0, 0.0, 0.0]], 15, 13, 50)
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(
            wavefunction, 0, include_double_d_shell=True
        )

        assert num_active_electrons == 10  # 28 - 18
        assert num_active_orbitals == 14

    def test_zinc_full_d_shell(self):
        """Zn (Z=30, period 4, d10): 12 valence electrons, 14 orbitals."""
        wavefunction = make_minimal_wavefunction(["Zn"], [[0.0, 0.0, 0.0]], 15, 15, 50)
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(
            wavefunction, 0, include_double_d_shell=True
        )

        assert num_active_electrons == 12  # 30 - 18
        assert num_active_orbitals == 14

    def test_silver_hydride(self):
        """AgH: Ag (Z=47, period 5) also gets 14 valence orbitals with double-d-shell."""
        # Ag (47) + H (1) = 48 electrons, singlet.
        wavefunction = make_minimal_wavefunction(["Ag", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.617]], 24, 24, 60)
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(
            wavefunction, 0, include_double_d_shell=True
        )

        assert num_active_electrons == 12  # 11 (Ag: 47-36) + 1 (H)
        assert num_active_orbitals == 15  # 14 (Ag) + 1 (H)

    def test_period3_unaffected(self):
        """Period 3 elements (Na, Cl) should still get 4 valence orbitals each."""
        # Na (11) + Cl (17) = 28 electrons, singlet.
        wavefunction = make_minimal_wavefunction(["Na", "Cl"], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.361]], 14, 14, 30)
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(
            wavefunction, 0, include_double_d_shell=True
        )

        assert num_active_electrons == 8  # 1 (Na) + 7 (Cl)
        assert num_active_orbitals == 8  # 4 (Na) + 4 (Cl)

    def test_potassium_period4_main_group(self):
        """K (Z=19, period 4 main group): only 9 valence orbitals (no d' shell)."""
        # 19 electrons doublet: 10 alpha + 9 beta.
        wavefunction = make_minimal_wavefunction(["K"], [[0.0, 0.0, 0.0]], 10, 9, 30)
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(
            wavefunction, 0, include_double_d_shell=True
        )

        assert num_active_electrons == 1  # 19 - 18
        assert num_active_orbitals == 9  # period 4 main-group: no d' shell

    def test_platinum_period6_d_block(self):
        """Pt (Z=78, period 6 d-block): 24 valence electrons, 21 valence orbitals."""
        # 78 electrons singlet.
        wavefunction = make_minimal_wavefunction(["Pt"], [[0.0, 0.0, 0.0]], 39, 39, 100)
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(
            wavefunction, 0, include_double_d_shell=True
        )

        assert num_active_electrons == 24  # 78 - 54 (Xe core)
        assert num_active_orbitals == 21  # 6s + 7*4f + 5*5d + 5*6d' + 3*6p

    def test_copper_default_no_double_d_shell(self):
        """With the default (include_double_d_shell=False) Cu has no d' shell."""
        wavefunction = make_minimal_wavefunction(["Cu"], [[0.0, 0.0, 0.0]], 15, 14, 50)
        # Default: include_double_d_shell=False.
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 11  # 29 - 18
        assert num_active_orbitals == 9  # 4s + 5*3d + 3*4p (no d' shell)

    def test_platinum_default_no_double_d_shell(self):
        """With the default (include_double_d_shell=False) Pt has no d' shell."""
        wavefunction = make_minimal_wavefunction(["Pt"], [[0.0, 0.0, 0.0]], 39, 39, 100)
        # Default: include_double_d_shell=False.
        (num_active_electrons, num_active_orbitals) = compute_valence_space_parameters(wavefunction, 0)

        assert num_active_electrons == 24  # 78 - 54
        assert num_active_orbitals == 16  # 6s + 7*4f + 5*5d + 3*6p (no d' shell)
