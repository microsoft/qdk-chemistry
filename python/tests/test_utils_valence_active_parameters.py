"""Tests for valence active space parameter utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

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
    pair_count = min(n_alpha, n_beta)
    single_count = abs(n_alpha - n_beta)
    if pair_count + single_count > num_molecular_orbitals:
        raise ValueError(
            f"make_minimal_wavefunction: num_molecular_orbitals ({num_molecular_orbitals}) "
            f"is too small for the requested electron count "
            f"(n_alpha={n_alpha}, n_beta={n_beta})."
        )

    structure = Structure(symbols, np.asarray(coords, dtype=float))

    exps = np.array([1.0])
    coefs = np.array([1.0])
    shells = [Shell(i, OrbitalType.S, exps, coefs) for i in range(len(symbols))]
    basis_set = BasisSet("dummy", shells, structure, AOType.Spherical)

    coeffs = np.zeros((basis_set.get_num_atomic_orbitals(), num_molecular_orbitals))
    orbitals = Orbitals(coeffs, None, None, basis_set)

    chars = ["0"] * num_molecular_orbitals
    for i in range(pair_count):
        chars[i] = "2"
    unpaired = "u" if n_alpha > n_beta else "d"
    for i in range(single_count):
        chars[pair_count + i] = unpaired

    config = Configuration("".join(chars))
    container = SlaterDeterminantContainer(config, orbitals)
    return Wavefunction(container)


def make_single_atom_wavefunction(symbol, charge=0, multiplicity=1, num_molecular_orbitals=None):
    """Single-atom Wavefunction at the origin sized from ``charge`` / ``multiplicity``.

    ``num_molecular_orbitals`` defaults to ``4 * Z`` -- comfortably larger than
    any valence space these tests check, so the cap in
    ``compute_valence_space_parameters`` does not bind unless tests pass an
    explicit smaller value.
    """
    z = int(Structure.symbol_to_element(symbol))
    n_electrons = z - charge
    n_unpaired = multiplicity - 1  # 2S = mult - 1
    n_alpha = (n_electrons + n_unpaired) // 2
    n_beta = n_electrons - n_alpha
    return make_minimal_wavefunction(
        [symbol],
        [[0.0, 0.0, 0.0]],
        n_alpha,
        n_beta,
        num_molecular_orbitals if num_molecular_orbitals is not None else 4 * z,
    )


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


class TestMakeMinimalWavefunction:
    """Self-tests for the no-SCF Wavefunction helper used by the TM tests."""

    @pytest.mark.parametrize(
        ("symbol", "charge", "multiplicity"),
        [("He", 0, 1), ("Li", 0, 2), ("O", 0, 3), ("O", -1, 2), ("Cu", 0, 2), ("Pt", 0, 1)],
    )
    def test_produces_requested_electron_counts(self, symbol, charge, multiplicity):
        """Helper round-trips ``(charge, multiplicity)`` through the Wavefunction."""
        wfn = make_single_atom_wavefunction(symbol, charge=charge, multiplicity=multiplicity)
        n_alpha, n_beta = wfn.get_total_num_electrons()
        z = int(Structure.symbol_to_element(symbol))
        assert n_alpha + n_beta == z - charge
        assert n_alpha - n_beta == multiplicity - 1

    def test_rejects_too_few_molecular_orbitals(self):
        """Passing a Configuration that won't fit the requested electrons raises."""
        # Triplet on a single atom needs >= 2 alpha (or beta) MO slots.
        with pytest.raises(ValueError, match="num_molecular_orbitals"):
            make_minimal_wavefunction(["He"], [[0.0, 0.0, 0.0]], 2, 0, 1)


# Parameterized table of single-atom transition-metal cases.
# (symbol, multiplicity, expected_active_electrons,
#  expected_active_orbitals_with_dshell, expected_active_orbitals_default)
#   - Period 4 d-block: 4s + 5*3d + 5*4d' + 3*4p = 14 (vs default 9).
#   - Period 4 main-group control (K): no d' shell either way.
#   - Period 6 d-block: 6s + 7*4f + 5*5d + 5*6d' + 3*6p = 21 (vs 16).
TM_SINGLE_ATOM_CASES = [
    ("Cu", 2, 11, 14, 9),
    ("Ni", 3, 10, 14, 9),
    ("Zn", 1, 12, 14, 9),
    ("K", 2, 1, 9, 9),
    ("Pt", 1, 24, 21, 16),
]


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

    @pytest.mark.parametrize(
        ("symbol", "multiplicity", "expected_nele", "expected_norb_on"),
        [(s, m, e, n_on) for (s, m, e, n_on, _n_off) in TM_SINGLE_ATOM_CASES],
    )
    def test_toggle_on_adds_d_prime_shell_for_d_block_only(self, symbol, multiplicity, expected_nele, expected_norb_on):
        """include_double_d_shell=True yields the expected sizing per element."""
        wfn = make_single_atom_wavefunction(symbol, charge=0, multiplicity=multiplicity)
        nele, norb = compute_valence_space_parameters(wfn, 0, include_double_d_shell=True)
        assert nele == expected_nele
        assert norb == expected_norb_on

    @pytest.mark.parametrize(
        ("symbol", "multiplicity", "expected_nele", "expected_norb_off"),
        [(s, m, e, n_off) for (s, m, e, _n_on, n_off) in TM_SINGLE_ATOM_CASES],
    )
    def test_toggle_off_preserves_historical_sizing(self, symbol, multiplicity, expected_nele, expected_norb_off):
        """Default (include_double_d_shell=False) preserves the pre-toggle sizing."""
        wfn = make_single_atom_wavefunction(symbol, charge=0, multiplicity=multiplicity)
        nele, norb = compute_valence_space_parameters(wfn, 0)
        assert nele == expected_nele
        assert norb == expected_norb_off

    def test_silver_hydride_adds_d_prime_on_ag_only(self):
        """AgH: Ag (period 5) gets the d' shell; the H spectator does not."""
        wfn = make_minimal_wavefunction(["Ag", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.617]], 24, 24, 60)
        nele_on, norb_on = compute_valence_space_parameters(wfn, 0, include_double_d_shell=True)
        assert nele_on == 12  # Ag valence (11) + H (1)
        assert norb_on == 15  # Ag period-5 with d' (14) + H (1)

        nele_off, norb_off = compute_valence_space_parameters(wfn, 0)
        assert nele_off == 12
        assert norb_off == 10  # Ag period-5 default (9) + H (1)

    def test_toggle_adds_exactly_five_per_d_block_atom(self):
        """Toggle delta is +5 * (number of d-block atoms), zero else; electrons unchanged."""
        # Zero d-block atoms -> zero delta.
        nacl = make_minimal_wavefunction(["Na", "Cl"], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.361]], 14, 14, 30)
        nacl_e_off, nacl_o_off = compute_valence_space_parameters(nacl, 0)
        nacl_e_on, nacl_o_on = compute_valence_space_parameters(nacl, 0, include_double_d_shell=True)
        assert nacl_e_off == nacl_e_on
        assert nacl_o_on - nacl_o_off == 0

        # One d-block atom -> +5.
        agh = make_minimal_wavefunction(["Ag", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.617]], 24, 24, 60)
        agh_e_off, agh_o_off = compute_valence_space_parameters(agh, 0)
        agh_e_on, agh_o_on = compute_valence_space_parameters(agh, 0, include_double_d_shell=True)
        assert agh_e_off == agh_e_on
        assert agh_o_on - agh_o_off == 5

    def test_toggle_respects_basis_cap(self):
        """When the basis is too small for the d' shell, the cap clips num_active_orbitals."""
        # Cu doublet with only 12 MOs total. num_core_mos = 9, so the active
        # space is capped at 12 - 9 = 3, far below the 14 the toggle would
        # otherwise ask for.
        wfn = make_single_atom_wavefunction("Cu", charge=0, multiplicity=2, num_molecular_orbitals=12)
        nele, norb = compute_valence_space_parameters(wfn, 0, include_double_d_shell=True)
        assert nele == 11  # electron count is unaffected by the orbital cap
        assert norb == 3  # capped at num_molecular_orbitals - num_core_mos
