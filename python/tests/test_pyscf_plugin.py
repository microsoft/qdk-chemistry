"""Tests for PySCF plugin functionality and basis set conversion utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms, data
from qdk_chemistry.data import Ansatz, Settings, Structure

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
    mcscf_energy_tolerance,
    orthonormality_error_tolerance,
    plain_text_tolerance,
    scf_energy_tolerance,
    scf_orbital_tolerance,
    unitarity_error_tolerance,
)

try:
    import pyscf
    import pyscf.gto
    import pyscf.lo
    import pyscf.mcscf
    import pyscf.scf

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

if PYSCF_AVAILABLE:
    from qdk_chemistry.constants import ANGSTROM_TO_BOHR
    from qdk_chemistry.data import AOType, BasisSet, OrbitalType, Shell
    from qdk_chemistry.plugins.pyscf.conversion import (
        basis_to_pyscf_mol,
        hamiltonian_to_scf,
        hamiltonian_to_scf_from_n_electrons_and_multiplicity,
        orbitals_to_scf,
        orbitals_to_scf_from_n_electrons_and_multiplicity,
        pyscf_mol_to_qdk_basis,
        structure_to_pyscf_atom_labels,
    )
    from qdk_chemistry.plugins.pyscf.mcscf import _mcsolver_to_fcisolver


pytestmark = pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")


def create_n2_structure():
    """Create a nitrogen molecule structure."""
    symbols = ["N", "N"]
    coords = np.array(
        [
            [0.000000000, 0.0000000000, 2.000000000000 * ANGSTROM_TO_BOHR],
            [0.000000000, 0.0000000000, 0.000000000000],
        ]
    )
    return Structure(symbols, coords)


def create_water_structure():
    """Create a water molecule structure.

    Crawford geometry - same as used in C++ tests.
    """
    symbols = ["O", "H", "H"]
    coords = np.array(
        [
            [0.000000000 * ANGSTROM_TO_BOHR, -0.0757918436 * ANGSTROM_TO_BOHR, 0.000000000000],
            [0.866811829 * ANGSTROM_TO_BOHR, 0.6014357793 * ANGSTROM_TO_BOHR, -0.000000000000],
            [-0.866811829 * ANGSTROM_TO_BOHR, 0.6014357793 * ANGSTROM_TO_BOHR, -0.000000000000],
        ]
    )
    return Structure(symbols, coords)


def create_helium_structure():
    """Create a helium atom structure."""
    return Structure(["He"], np.array([[0.0, 0.0, 0.0]]))


def create_li_structure():
    """Create a lithium atom structure."""
    return Structure(["Li"], np.array([[0.0, 0.0, 0.0]]))


def create_o2_structure():
    """Create an oxygen molecule (O2) structure."""
    symbols = ["O", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.208 * ANGSTROM_TO_BOHR]])
    return Structure(symbols, coords)


def create_uo2_structure():
    """Create a uranyl ion (UO2) structure."""
    symbols = ["U", "O", "O"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.85213 * ANGSTROM_TO_BOHR],
            [0.0, 0.0, -1.85213 * ANGSTROM_TO_BOHR],
        ]
    )
    return Structure(symbols, coords)


def pipek_objective_function(orbitals, mos):
    """Calculate the Pipek-Mezey objective function."""
    mf = orbitals_to_scf(orbitals, 0, 0)
    mol = mf.mol
    pm = pyscf.lo.PM(mol, mos, mf, pop_method="mulliken")
    return pm.cost_function(None)


def boys_objective_function(orbitals, mos):
    """Calculate the Foster-Boys objective function."""
    mf = orbitals_to_scf(orbitals, 0, 0)
    mol = mf.mol
    fb = pyscf.lo.Boys(mol, mos)
    return fb.cost_function(None)


def er_objective_function(orbitals, mos):
    """Calculate the Edmiston-Ruedenberg objective function."""
    mf = orbitals_to_scf(orbitals, 0, 0)  # true electron count not needed
    mol = mf.mol
    er = pyscf.lo.ER(mol, mos)
    return er.cost_function(np.eye(mos.shape[1]))


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestPyscfPlugin:
    """Test class for PySCF plugin functionality."""

    def test_pyscf_plugin_registration(self):
        """Test that PySCF plugin is properly registered."""
        available_solvers = algorithms.available("scf_solver")
        assert "pyscf" in available_solvers

        available_localizers = algorithms.available("orbital_localizer")
        assert "pyscf_multi" in available_localizers

        available_selectors = algorithms.available("active_space_selector")
        assert "pyscf_avas" in available_selectors

        available_stability_checkers = algorithms.available("stability_checker")
        assert "pyscf" in available_stability_checkers

    def test_pyscf_scf_solver_creation(self):
        """Test creating PySCF SCF solver."""
        scf_solver = algorithms.create("scf_solver", "pyscf")
        assert scf_solver is not None

    def test_pyscf_localizer_creation(self):
        """Test creating PySCF localizer."""
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        assert localizer is not None

    def test_pyscf_avas_selector_creation(self):
        """Test creating PySCF AVAS Active Space Selector."""
        avas_selector = algorithms.create("active_space_selector", "pyscf_avas")
        assert avas_selector is not None

    def test_pyscf_cc_calculator(self):
        """Test creating PySCF Coupled Cluster module."""
        cc = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        assert cc is not None

    def test_pyscf_stability_checker_creation(self):
        """Test creating PySCF stability checker."""
        stability_checker = algorithms.create("stability_checker", "pyscf")
        assert stability_checker is not None

    def test_pyscf_scf_solver_settings(self):
        """Test PySCF SCF solver settings interface."""
        scf_solver = algorithms.create("scf_solver", "pyscf")
        settings = scf_solver.settings()

        # Test that settings object exists
        assert settings is not None

        assert settings.get("max_iterations") == 50

        # Test setting max iterations
        settings.set("max_iterations", 100)
        assert settings.get("max_iterations") == 100

        # Test setting other parameters
        settings.set("scf_type", "restricted")
        assert settings.get("scf_type") == "restricted"

    def test_pyscf_localizer_settings(self):
        """Test PySCF localizer settings interface."""
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        settings = localizer.settings()

        # Test that settings object exists
        assert settings is not None

        # Test default values
        assert settings.get("method") == "pipek-mezey"
        assert settings.get("population_method") == "mulliken"

        # Test setting method
        settings.set("method", "foster-boys")
        assert settings.get("method") == "foster-boys"

        # Test setting population method
        settings.set("population_method", "lowdin")
        assert settings.get("population_method") == "lowdin"

    def test_pyscf_avas_selector_settings(self):
        """Test PySCF AVAS selector settings interface."""
        avas_selector = algorithms.create("active_space_selector", "pyscf_avas")
        settings = avas_selector.settings()

        # Test that settings object exists
        assert settings is not None

        ao_labels = settings.get("ao_labels")
        assert len(ao_labels) == 0

        canonicalize = settings.get("canonicalize")
        assert canonicalize is False

        ref_labels = ["1s", "2s", "2p"]
        settings.set("ao_labels", ref_labels)
        ao_labels = settings.get("ao_labels")
        assert ao_labels == ref_labels

    def test_pyscf_cc_settings(self):
        """Test PySCF Coupled Cluster settings interface."""
        cc = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        settings = cc.settings()

        # Test that settings object exists
        assert settings is not None

        # Since the (T) setting was removed, these assertions are no longer valid
        # Instead, let's test that settings is a proper Settings object
        assert isinstance(settings, Settings)

        # Add a test setting
        settings.set("conv_tol", 1e-8)
        assert settings.get("conv_tol") == 1e-8

    def test_pyscf_stability_checker_settings(self):
        """Test PySCF stability checker settings interface."""
        stability_checker = algorithms.create("stability_checker", "pyscf")
        settings = stability_checker.settings()

        # Test that settings object exists
        assert settings is not None

        # Test default settings
        assert settings.get("internal") is True
        assert settings.get("external") is True
        assert settings.get("with_symmetry") is True
        assert settings.get("nroots") == 3
        assert settings.get("davidson_tolerance") == 1e-4
        assert settings.get("stability_tolerance") == -1e-4

        # Test setting parameters
        settings.set("internal", False)
        assert settings.get("internal") is False

        settings.set("nroots", 5)
        assert settings.get("nroots") == 5

        settings.set("stability_tolerance", 1e-6)
        assert settings.get("stability_tolerance") == 1e-6

    def test_pyscf_water_scf_def2svp(self):
        """Test PySCF SCF solver on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -75.9229032346701, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

        # Check basis set populated
        basis_set = orbitals.get_basis_set()
        assert basis_set is not None

    def test_pyscf_water_scf_def2tzvp(self):
        """Test PySCF SCF solver on water molecule with def2-tzvp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-tzvp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -76.02057765181318, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

        # Check basis set populated
        basis_set = orbitals.get_basis_set()
        assert basis_set is not None

    def test_pyscf_li_scf_def2svp(self):
        """Test PySCF SCF solver on lithium atom with def2-svp basis."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(lithium, 0, 2, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(energy, -7.4250663561, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted() is False

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_li_rohf_def2svp(self):
        """Test PySCF SCF solver on lithium atom with ROHF/def2-svp basis."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        energy, wavefunction = scf_solver.run(lithium, 0, 2, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -7.42506404463744, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_li_plus_scf_def2svp(self):
        """Test PySCF SCF solver on lithium ion (Li+) with def2-svp basis."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(lithium, 1, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -7.23289811389006, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_o2_triplet_scf_def2svp(self):
        """Test PySCF SCF solver on O2 triplet state with def2-svp basis."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -149.49029917454197, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted() is False

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 9.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 7.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_water_dft_b3lyp_def2svp(self):
        """Test PySCF DFT solver on water molecule with B3LYP/def2-svp."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")

        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -76.33342033646656, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_water_dft_pbe_def2svp(self):
        """Test PySCF DFT solver on water molecule with PBE/def2-svp."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "pbe")

        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -76.2511269787294, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()

    def test_pyscf_li_dft_b3lyp_def2svp(self):
        """Test PySCF DFT solver on lithium atom with B3LYP/def2-svp (UKS)."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")

        energy, wavefunction = scf_solver.run(lithium, 0, 2, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -7.484980651804635, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted() is False  # Should be UKS (unrestricted)

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_li_dft_roks_b3lyp_def2svp(self):
        """Test PySCF DFT solver on lithium atom with ROKS B3LYP/def2-svp."""
        lithium = create_li_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")
        scf_solver.settings().set("scf_type", "restricted")

        energy, wavefunction = scf_solver.run(lithium, 0, 2, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable ROKS DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -7.484979697016255, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted()  # Should be restricted (ROKS)

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 2.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_o2_triplet_dft_b3lyp_def2svp(self):
        """Test PySCF DFT solver on O2 triplet state with B3LYP/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")

        energy, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable DFT results
        assert isinstance(energy, float)
        assert np.isclose(
            energy, -150.204697358644, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )
        assert orbitals is not None
        assert orbitals.has_basis_set()
        assert orbitals.has_overlap_matrix()
        assert orbitals.is_restricted() is False  # Should be UKS
        assert orbitals.is_unrestricted()

        # Check occupations
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 9.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            np.sum(occ_b), 7.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_dft_method_case_insensitive(self):
        """Test that DFT method names are case insensitive."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        # Test uppercase
        scf_solver.settings().set("method", "B3LYP")
        energy_upper, _ = scf_solver.run(water, 0, 1, "sto-3g")

        # Test lowercase
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "b3lyp")
        energy_lower, _ = scf_solver.run(water, 0, 1, "sto-3g")

        # Should give the same result
        assert np.isclose(
            energy_upper, energy_lower, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

    def test_pyscf_uo2_lanl2dz(self):
        """Test PySCF SCF solver on UO2 with LANL2DZ basis and ECP."""
        uo2 = create_uo2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        energy, _ = scf_solver.run(uo2, 0, 1, "lanl2dz")
        ref_energy = -200.29749139183
        assert np.isclose(energy, ref_energy, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)

    def test_pyscf_scf_solver_initial_guess_restart(self):
        """Test PySCF SCF solver with initial guess from converged orbitals."""
        # Water as restricted test
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("method", "hf")

        # First calculation - let it converge normally
        energy_first, wfn_first = scf_solver.run(water, 0, 1, "def2-tzvp")
        orbitals_first = wfn_first.get_orbitals()

        # Verify we get the expected energy for HF/def2-tzvp
        assert np.isclose(
            energy_first, -76.0205776518, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # Now restart with the converged orbitals as initial guess
        # Create a new solver instance since settings are locked after run
        scf_solver2 = algorithms.create("scf_solver", "pyscf")
        scf_solver2.settings().set("method", "hf")
        scf_solver2.settings().set("max_iterations", 2)  # 2 is minimum as need to check energy difference

        # Second calculation with initial guess
        energy_second, _ = scf_solver2.run(water, 0, 1, orbitals_first)

        # Should get the same energy (within tight tolerance)
        assert np.isclose(
            energy_first, energy_second, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # Oxygen Triplet Initial Guess Test
        o2 = create_o2_structure()
        scf_solver3 = algorithms.create("scf_solver", "pyscf")
        scf_solver3.settings().set("method", "hf")

        # First calculation - let triplet converge normally
        energy_o2_first, wfn_o2_first = scf_solver3.run(o2, 0, 3, "sto-3g")
        orbitals_o2_first = wfn_o2_first.get_orbitals()

        # Verify we get the expected energy for HF/STO-3G triplet
        assert np.isclose(
            energy_o2_first, -147.633969608498, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # Now restart with the converged orbitals as initial guess
        # Create a new solver instance since settings are locked after run
        scf_solver4 = algorithms.create("scf_solver", "pyscf")
        scf_solver4.settings().set("method", "hf")
        scf_solver4.settings().set("max_iterations", 2)  # 2 is minimum as need to check energy difference

        # Second calculation with initial guess
        energy_o2_second, _ = scf_solver4.run(o2, 0, 3, orbitals_o2_first)

        # Should get the same energy (within tight tolerance)
        assert np.isclose(
            energy_o2_first, energy_o2_second, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

    def test_pyscf_water_pm_localization_def2svp(self):
        """Test PySCF Pipek-Mezey localization on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_objective_value = pipek_objective_function(orbitals, orbitals.get_coefficients()[0])

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals

        # Prepare for Test 2: Calculate can_random_objective_value before localizer creation
        random_occ_indices = [1, 3, 4]  # Random subset of occupied orbitals
        ca_can, _ = orbitals.get_coefficients()
        ca_selected = ca_can[:, random_occ_indices]
        can_random_objective_value = pipek_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = pipek_objective_function(localized_virt, mos_final)
        assert final_objective_value > can_objective_value

        # Test 2: Randomly choose indices from occupied orbitals only
        localized_random = localizer.run(wavefunction, random_occ_indices, random_occ_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_occ_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_occ_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_occ_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = pipek_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value >= can_random_objective_value

    def test_pyscf_water_fb_localization_def2svp(self):
        """Test PySCF Foster-Boys localization on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_objective_value = boys_objective_function(orbitals, orbitals.get_coefficients()[0])

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals

        # Prepare for Test 2: Calculate can_random_objective_value before localizer creation
        random_virt_indices = [5, 7, 9]  # Random subset of virtual orbitals (indices >= num_occupied_orbitals)
        ca_can, _ = orbitals.get_coefficients()
        ca_selected = ca_can[:, random_virt_indices]
        can_random_objective_value = boys_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "foster-boys")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = boys_objective_function(localized_virt, mos_final)
        assert final_objective_value < can_objective_value

        # Test 2: Randomly choose indices from virtual orbitals only
        localized_random = localizer.run(wavefunction, random_virt_indices, random_virt_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_virt_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_virt_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_virt_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = boys_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value <= can_random_objective_value

    def test_pyscf_water_er_localization_def2svp(self):
        """Test PySCF Edmiston-Ruedenberg localization on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()
        ca_can, _ = orbitals.get_coefficients()
        # Compute the objective function for the canonical orbitals
        can_objective_value = er_objective_function(orbitals, orbitals.get_coefficients()[0])
        # Random subset of occupied orbitals, must include 0 (O 1s), possibly due to numerical instability
        random_occ_indices = [0, 1, 4]
        # Prepare for Test 2: Calculate can_random_objective_value before localizer creation
        ca_selected = ca_can[:, random_occ_indices]
        can_random_objective_value = er_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "edmiston-ruedenberg")

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals
        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = er_objective_function(localized_virt, mos_final)
        assert final_objective_value > can_objective_value

        # Test 2: Randomly choose indices from occupied orbitals only
        localized_random = localizer.run(wavefunction, random_occ_indices, random_occ_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_occ_indices]
        # Test that the objective function improved for the random selection
        random_objective_value = er_objective_function(orbitals, ca_loc_selected)
        assert random_objective_value >= can_random_objective_value

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_occ_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_occ_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

    def test_pyscf_o2_pm_localization_def2svp_uhf(self):
        """Test PySCF Pipek-Mezey localization on O2 molecule with UHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        [can_a, can_b] = orbitals.get_coefficients()
        can_objective_value_a = pipek_objective_function(orbitals, can_a)
        can_objective_value_b = pipek_objective_function(orbitals, can_b)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_alpha, num_beta = num_elec[0], num_elec[1]

        # Prepare for Test 2: Randomly choose indices from occupied orbitals only for both spin channels
        random_occ_indices_alpha = [2, 3]  # Random subset of occupied alpha orbitals
        random_occ_indices_beta = [0, 1]  # Random subset of occupied beta orbitals
        ca_selected = can_a[:, random_occ_indices_alpha]
        cb_selected = can_b[:, random_occ_indices_beta]
        can_random_objective_value_a = pipek_objective_function(orbitals, ca_selected)
        can_random_objective_value_b = pipek_objective_function(orbitals, cb_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices_alpha = list(range(num_alpha))
        virt_indices_alpha = list(range(num_alpha, orbitals.get_num_molecular_orbitals()))
        occ_indices_beta = list(range(num_beta))
        virt_indices_beta = list(range(num_beta, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices_alpha, occ_indices_beta)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices_alpha, virt_indices_beta)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final_a, mos_final_b = localized_virt.get_coefficients()
        final_objective_value_a = pipek_objective_function(localized_virt, mos_final_a)
        final_objective_value_b = pipek_objective_function(localized_virt, mos_final_b)
        assert final_objective_value_a > can_objective_value_a
        assert final_objective_value_b > can_objective_value_b

        # Test 2: Randomly choose indices to localize for both spin channels
        localized_random = localizer.run(wavefunction, random_occ_indices_alpha, random_occ_indices_beta)
        mos_rand_a, mos_rand_b = localized_random.get_orbitals().get_coefficients()

        # Test alpha channel
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand_a[:, random_occ_indices_alpha]

        # Check that the transformation for selected indices is unitary - alpha
        u_selected_a = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error_a = np.linalg.norm(u_selected_a @ u_selected_a.T - np.eye(len(random_occ_indices_alpha)))
        assert unitarity_error_a < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - alpha
        overlap_check_a = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error_a = np.linalg.norm(overlap_check_a - np.eye(len(random_occ_indices_alpha)))
        assert orthonormality_error_a < orthonormality_error_tolerance

        # Test beta channel
        cb_loc_selected = mos_rand_b[:, random_occ_indices_beta]

        # Check that the transformation for selected indices is unitary - beta
        u_selected_b = cb_selected.T @ s_matrix @ cb_loc_selected
        unitarity_error_b = np.linalg.norm(u_selected_b @ u_selected_b.T - np.eye(len(random_occ_indices_beta)))
        assert unitarity_error_b < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - beta
        overlap_check_b = cb_loc_selected.T @ s_matrix @ cb_loc_selected
        orthonormality_error_b = np.linalg.norm(overlap_check_b - np.eye(len(random_occ_indices_beta)))
        assert orthonormality_error_b < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value_a = pipek_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        random_objective_value_b = pipek_objective_function(localized_random.get_orbitals(), cb_loc_selected)
        # Allow tiny numerical tolerance due to floating point precision
        assert random_objective_value_a >= can_random_objective_value_a - 5e-14
        assert random_objective_value_b >= can_random_objective_value_b - 5e-14

    def test_pyscf_o2_fb_localization_def2svp_uhf(self):
        """Test PySCF Foster-Boys localization on O2 molecule with UHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        [can_a, can_b] = orbitals.get_coefficients()
        can_objective_value_a = boys_objective_function(orbitals, can_a)
        can_objective_value_b = boys_objective_function(orbitals, can_b)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_alpha, num_beta = num_elec[0], num_elec[1]

        # Prepare for Test 2: Randomly choose indices from virtual orbitals only for both spin channels
        random_virt_indices_alpha = [
            num_alpha + 1,
            num_alpha + 3,
        ]  # Random subset of virtual alpha orbitals
        random_virt_indices_beta = [
            num_beta + 2,
            num_beta + 4,
        ]  # Random subset of virtual beta orbitals
        ca_selected = can_a[:, random_virt_indices_alpha]
        cb_selected = can_b[:, random_virt_indices_beta]
        can_random_objective_value_a = boys_objective_function(orbitals, ca_selected)
        can_random_objective_value_b = boys_objective_function(orbitals, cb_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "foster-boys")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices_alpha = list(range(num_alpha))
        virt_indices_alpha = list(range(num_alpha, orbitals.get_num_molecular_orbitals()))
        occ_indices_beta = list(range(num_beta))
        virt_indices_beta = list(range(num_beta, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices_alpha, occ_indices_beta)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices_alpha, virt_indices_beta)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final_a, mos_final_b = localized_virt.get_coefficients()
        final_objective_value_a = boys_objective_function(localized_virt, mos_final_a)
        final_objective_value_b = boys_objective_function(localized_virt, mos_final_b)
        assert final_objective_value_a < can_objective_value_a
        assert final_objective_value_b < can_objective_value_b

        # Test 2: Randomly choose indices from virtual orbitals only for both spin channels
        localized_random = localizer.run(wavefunction, random_virt_indices_alpha, random_virt_indices_beta)
        mos_rand_a, mos_rand_b = localized_random.get_orbitals().get_coefficients()

        # Test alpha channel
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand_a[:, random_virt_indices_alpha]

        # Check that the transformation for selected indices is unitary - alpha
        u_selected_a = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error_a = np.linalg.norm(u_selected_a @ u_selected_a.T - np.eye(len(random_virt_indices_alpha)))
        assert unitarity_error_a < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - alpha
        overlap_check_a = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error_a = np.linalg.norm(overlap_check_a - np.eye(len(random_virt_indices_alpha)))
        assert orthonormality_error_a < orthonormality_error_tolerance

        # Test beta channel
        cb_loc_selected = mos_rand_b[:, random_virt_indices_beta]

        # Check that the transformation for selected indices is unitary - beta
        u_selected_b = cb_selected.T @ s_matrix @ cb_loc_selected
        unitarity_error_b = np.linalg.norm(u_selected_b @ u_selected_b.T - np.eye(len(random_virt_indices_beta)))
        assert unitarity_error_b < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - beta
        overlap_check_b = cb_loc_selected.T @ s_matrix @ cb_loc_selected
        orthonormality_error_b = np.linalg.norm(overlap_check_b - np.eye(len(random_virt_indices_beta)))
        assert orthonormality_error_b < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value_a = boys_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        random_objective_value_b = boys_objective_function(localized_random.get_orbitals(), cb_loc_selected)
        # Allow tiny numerical tolerance due to floating point precision
        assert random_objective_value_a <= can_random_objective_value_a + 5e-14
        assert random_objective_value_b <= can_random_objective_value_b + 5e-14

    def test_pyscf_o2_er_localization_def2svp_uhf(self):
        """Test PySCF Edmiston-Ruedenberg localization on O2 molecule with UHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        [can_a, can_b] = orbitals.get_coefficients()
        can_objective_value_a = er_objective_function(orbitals, can_a)
        can_objective_value_b = er_objective_function(orbitals, can_b)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_alpha, num_beta = num_elec[0], num_elec[1]

        # Prepare for Test 2: Randomly choose indices from occupied orbitals only for both spin channels
        random_occ_indices_alpha = [2, 3]  # Random subset of occupied alpha orbitals
        random_occ_indices_beta = [0, 1, 4]  # Random subset of occupied beta orbitals
        ca_selected = can_a[:, random_occ_indices_alpha]
        cb_selected = can_b[:, random_occ_indices_beta]
        can_random_objective_value_a = er_objective_function(orbitals, ca_selected)
        can_random_objective_value_b = er_objective_function(orbitals, cb_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices_alpha = list(range(num_alpha))
        virt_indices_alpha = list(range(num_alpha, orbitals.get_num_molecular_orbitals()))
        occ_indices_beta = list(range(num_beta))
        virt_indices_beta = list(range(num_beta, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices_alpha, occ_indices_beta)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices_alpha, virt_indices_beta)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final_a, mos_final_b = localized_virt.get_coefficients()
        final_objective_value_a = er_objective_function(localized_virt, mos_final_a)
        final_objective_value_b = er_objective_function(localized_virt, mos_final_b)
        assert final_objective_value_a > can_objective_value_a
        assert final_objective_value_b > can_objective_value_b

        # Test 2: Randomly choose indices to localize for both spin channels
        localized_random = localizer.run(wavefunction, random_occ_indices_alpha, random_occ_indices_beta)
        mos_rand_a, mos_rand_b = localized_random.get_orbitals().get_coefficients()

        # Test alpha channel
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand_a[:, random_occ_indices_alpha]

        # Check that the transformation for selected indices is unitary - alpha
        u_selected_a = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error_a = np.linalg.norm(u_selected_a @ u_selected_a.T - np.eye(len(random_occ_indices_alpha)))
        assert unitarity_error_a < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - alpha
        overlap_check_a = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error_a = np.linalg.norm(overlap_check_a - np.eye(len(random_occ_indices_alpha)))
        assert orthonormality_error_a < orthonormality_error_tolerance

        # Test beta channel
        cb_loc_selected = mos_rand_b[:, random_occ_indices_beta]

        # Check that the transformation for selected indices is unitary - beta
        u_selected_b = cb_selected.T @ s_matrix @ cb_loc_selected
        unitarity_error_b = np.linalg.norm(u_selected_b @ u_selected_b.T - np.eye(len(random_occ_indices_beta)))
        assert unitarity_error_b < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal - beta
        overlap_check_b = cb_loc_selected.T @ s_matrix @ cb_loc_selected
        orthonormality_error_b = np.linalg.norm(overlap_check_b - np.eye(len(random_occ_indices_beta)))
        assert orthonormality_error_b < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value_a = er_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        random_objective_value_b = er_objective_function(localized_random.get_orbitals(), cb_loc_selected)
        # Allow tiny numerical tolerance due to floating point precision
        assert random_objective_value_a >= can_random_objective_value_a - 5e-14
        assert random_objective_value_b >= can_random_objective_value_b - 5e-14

    def test_pyscf_o2_pm_localization_def2svp_rohf(self):
        """Test PySCF Pipek-Mezey localization on O2 molecule with ROHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_mos, _ = orbitals.get_coefficients()
        can_objective_value = pipek_objective_function(orbitals, can_mos)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals (same as alpha for ROHF)

        # Prepare for Test 2: Randomly choose indices from occupied orbitals only
        random_occ_indices = [0, 3, 4, 7]  # Random subset of occupied orbitals
        ca_selected = can_mos[:, random_occ_indices]
        can_random_objective_value = pipek_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = pipek_objective_function(localized_virt, mos_final)
        assert final_objective_value > can_objective_value

        # Test 2: Randomly choose indices from occupied orbitals only
        localized_random = localizer.run(wavefunction, random_occ_indices, random_occ_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_occ_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_occ_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_occ_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = pipek_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value >= can_random_objective_value

    def test_pyscf_o2_fb_localization_def2svp_rohf(self):
        """Test PySCF Foster-Boys localization on O2 molecule with ROHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_mos, _ = orbitals.get_coefficients()
        can_objective_value = boys_objective_function(orbitals, can_mos)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals (same as alpha for ROHF)

        # Prepare for Test 2: Randomly choose indices from virtual orbitals only
        random_virt_indices = [
            num_occupied_orbitals + 1,
            num_occupied_orbitals + 3,
            num_occupied_orbitals + 5,
            num_occupied_orbitals + 7,
        ]  # Random subset of virtual orbitals
        ca_selected = can_mos[:, random_virt_indices]
        can_random_objective_value = boys_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "foster-boys")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = boys_objective_function(localized_virt, mos_final)
        assert final_objective_value < can_objective_value

        # Test 2: Randomly choose indices from virtual orbitals only
        localized_random = localizer.run(wavefunction, random_virt_indices, random_virt_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_virt_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_virt_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_virt_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = boys_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value <= can_random_objective_value

    def test_pyscf_o2_er_localization_def2svp_rohf(self):
        """Test PySCF Edmiston-Ruedenberg localization on O2 molecule with ROHF/def2-svp."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Compute the objective function for the canonical orbitals
        can_mos, _ = orbitals.get_coefficients()
        can_objective_value = er_objective_function(orbitals, can_mos)

        # Get orbital occupancy information
        num_elec = wavefunction.get_total_num_electrons()
        num_occupied_orbitals = num_elec[0]  # Number of occupied orbitals (same as alpha for ROHF)

        # Prepare for Test 2: Randomly choose indices from occupied orbitals only
        random_occ_indices = [0, 3, 4]  # Random subset of occupied orbitals
        ca_selected = can_mos[:, random_occ_indices]
        can_random_objective_value = er_objective_function(orbitals, ca_selected)

        # Localize orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "edmiston-ruedenberg")

        # Test 1: Localize occupied first, then virtual orbitals on the localized occupied orbitals
        occ_indices = list(range(num_occupied_orbitals))
        virt_indices = list(range(num_occupied_orbitals, orbitals.get_num_molecular_orbitals()))

        # Localize occupied orbitals first
        localized_occ_wfn = localizer.run(wavefunction, occ_indices, occ_indices)
        assert localized_occ_wfn is not None

        # Localize virtual orbitals on the occupied-localized orbitals
        localized_virt_wfn = localizer.run(localized_occ_wfn, virt_indices, virt_indices)
        assert localized_virt_wfn is not None

        # Check that the objective function improved for the final orbitals
        localized_virt = localized_virt_wfn.get_orbitals()
        mos_final, _ = localized_virt.get_coefficients()
        final_objective_value = er_objective_function(localized_virt, mos_final)
        assert final_objective_value > can_objective_value

        # Test 2: Randomly choose indices from occupied orbitals only
        localized_random = localizer.run(wavefunction, random_occ_indices, random_occ_indices)
        mos_rand, _ = localized_random.get_orbitals().get_coefficients()

        # Extract the submatrix for the localized indices
        s_matrix = orbitals.get_overlap_matrix()
        ca_loc_selected = mos_rand[:, random_occ_indices]

        # Check that the transformation for selected indices is unitary
        u_selected = ca_selected.T @ s_matrix @ ca_loc_selected
        unitarity_error = np.linalg.norm(u_selected @ u_selected.T - np.eye(len(random_occ_indices)))
        assert unitarity_error < unitarity_error_tolerance

        # Check that localized orbitals are orthonormal
        overlap_check = ca_loc_selected.T @ s_matrix @ ca_loc_selected
        orthonormality_error = np.linalg.norm(overlap_check - np.eye(len(random_occ_indices)))
        assert orthonormality_error < orthonormality_error_tolerance

        # Test that the objective function improved for the random selection
        random_objective_value = er_objective_function(localized_random.get_orbitals(), ca_loc_selected)
        assert random_objective_value >= can_random_objective_value

    # =============================================================================
    # Tests for active space preservation after localization
    # Regression tests for bug: active space indices lost after orbital localization
    # =============================================================================

    def _verify_active_space_preserved(self, wfn_before, wfn_after, localizer_name):
        """Helper to verify active space indices are preserved after localization."""
        orbitals_before = wfn_before.get_orbitals()
        orbitals_after = wfn_after.get_orbitals()

        assert orbitals_before.has_active_space()
        assert orbitals_after.has_active_space(), f"Active space lost after {localizer_name} localization"

        alpha_before, beta_before = orbitals_before.get_active_space_indices()
        alpha_after, beta_after = orbitals_after.get_active_space_indices()

        assert list(alpha_before) == list(alpha_after), f"{localizer_name}: alpha indices changed"
        assert list(beta_before) == list(beta_after), f"{localizer_name}: beta indices changed"

    @pytest.mark.parametrize("method", ["pipek-mezey", "foster-boys", "edmiston-ruedenberg", "cholesky"])
    def test_pyscf_localization_preserves_active_space_restricted(self, method):
        """Test that PySCF localization preserves active space indices (restricted)."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")

        # Select an active space
        selector = algorithms.create("active_space_selector", "qdk_valence")
        selector.settings().set("num_active_electrons", 6)
        selector.settings().set("num_active_orbitals", 5)
        active_wfn = selector.run(wavefunction)

        active_alpha, active_beta = active_wfn.get_orbitals().get_active_space_indices()

        # Localize
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", method)
        localized_wfn = localizer.run(active_wfn, list(active_alpha), list(active_beta))

        self._verify_active_space_preserved(active_wfn, localized_wfn, f"pyscf_{method}")

    def test_pyscf_localization_preserves_active_space_unrestricted(self):
        """Test that PySCF localization preserves active space indices (unrestricted)."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "unrestricted")
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")

        # Manually set active space indices (ValenceActiveSpaceSelector doesn't support UHF)
        orbitals = wavefunction.get_orbitals()
        num_mo = orbitals.get_num_molecular_orbitals()

        # Define active space: frozen core (first 2 are inactive), rest are active
        # Must include all occupied orbitals in active space for SlaterDeterminantContainer
        active_alpha = list(range(2, num_mo))
        active_beta = list(range(2, num_mo))
        inactive_alpha = [0, 1]
        inactive_beta = [0, 1]

        # Create orbitals with active space
        coeffs_alpha, coeffs_beta = orbitals.get_coefficients()
        active_orbitals = data.Orbitals(
            coefficients_alpha=coeffs_alpha,
            coefficients_beta=coeffs_beta,
            ao_overlap=orbitals.get_overlap_matrix() if orbitals.has_overlap_matrix() else None,
            basis_set=orbitals.get_basis_set(),
            indices=(active_alpha, active_beta, inactive_alpha, inactive_beta),
        )

        active_wfn = data.Wavefunction(
            data.SlaterDeterminantContainer(wavefunction.get_active_determinants()[0], active_orbitals)
        )

        # Localize only the active orbitals
        localizer = algorithms.create("orbital_localizer", "pyscf_multi")
        localizer.settings().set("method", "pipek-mezey")
        localized_wfn = localizer.run(active_wfn, active_alpha, active_beta)

        self._verify_active_space_preserved(active_wfn, localized_wfn, "pyscf_pipek_mezey_unrestricted")

    def test_pyscf_avas_selector_water_def2svp(self):
        """Test PySCF AVAS selector on water molecule with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")

        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")

        # Select active space using AVAS
        avas_selector = algorithms.create("active_space_selector", "pyscf_avas")
        avas_selector.settings().set("ao_labels", ["O 2s", "O 2p", "H 1s"])
        active_wfn = avas_selector.run(wavefunction)

        act_a, act_b = active_wfn.get_orbitals().get_active_space_indices()
        assert act_a == act_b
        assert act_a == [0, 1, 2, 3, 4]

        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        assert np.isclose(
            np.sum(occ_a), 5.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.allclose(
            occ_a, occ_b, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_pyscf_avas_selector_o2_triplet_def2svp(self):
        """Test PySCF AVAS selector on O2 molecule (triplet ROHF) with def2-svp basis."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")

        # Select active space using AVAS
        avas_selector = algorithms.create("active_space_selector", "pyscf_avas")
        avas_selector.settings().set("ao_labels", ["O 2s", "O 2p"])
        active_wfn = avas_selector.run(wavefunction)

        act_a, act_b = active_wfn.get_orbitals().get_active_space_indices()
        assert act_a == act_b
        assert act_a == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_pyscf_ccsd_water_def2svp(self):
        """Test PySCF CCSD on water with def2-svp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")

        ham_calculator = algorithms.create("hamiltonian_constructor", "qdk")
        hamiltonian = ham_calculator.run(wavefunction.get_orbitals())

        # Compute CC energy
        cc_calculator = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        cc_calculator.settings().set("store_amplitudes", True)
        ansatz_object = Ansatz(hamiltonian, wavefunction)
        cc_energy, updated_wavefunction = cc_calculator.run(ansatz_object)
        reference_energy = -76.14613724756676
        assert np.isclose(cc_energy, reference_energy), f"{cc_energy=} should match total energy {reference_energy=}"

        # Get amplitudes from the wavefunction container
        cc_container = updated_wavefunction.get_container()
        assert cc_container.has_t1_amplitudes()
        assert cc_container.has_t2_amplitudes()
        t1_amplitudes = cc_container.get_t1_amplitudes()
        t2_amplitudes = cc_container.get_t2_amplitudes()
        assert t1_amplitudes is not None
        assert t2_amplitudes is not None

    def test_pyscf_uccsd_o2_triplet_def2svp(self):
        """Test PySCF UCCSD on O2 triplet with def2-svp basis."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")

        # Verify we have unrestricted orbitals
        orbitals = wavefunction.get_orbitals()
        assert orbitals.is_unrestricted(), "O2 triplet should have unrestricted orbitals"
        assert wavefunction.get_container_type() == "sd"
        assert wavefunction.size() == 1, "single determinant"

        # Create Hamiltonian
        ham_calculator = algorithms.create("hamiltonian_constructor", "qdk")
        hamiltonian = ham_calculator.run(orbitals)

        # Compute UCCSD energy
        cc_calculator = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        cc_calculator.settings().set("store_amplitudes", True)
        ansatz_object = Ansatz(hamiltonian, wavefunction)
        cc_energy, updated_wavefunction = cc_calculator.run(ansatz_object)
        reference_energy = -149.8417973596817
        assert np.isclose(cc_energy, reference_energy), (
            f"cc energy {cc_energy} should match reference {reference_energy}"
        )

        # Get amplitudes from the wavefunction container
        cc_container = updated_wavefunction.get_container()
        assert cc_container.has_t1_amplitudes(), "should have T1 amplitudes"
        assert cc_container.has_t2_amplitudes(), "should have T2 amplitudes"

        # For unrestricted, we should get separate alpha and beta amplitudes
        t1_alpha, t1_beta = cc_container.get_t1_amplitudes()
        t2_abab, t2_aaaa, t2_bbbb = cc_container.get_t2_amplitudes()

        # Verify all amplitudes are present
        assert t1_alpha is not None, "T1 alpha amplitudes should not be None"
        assert t1_beta is not None, "T1 beta amplitudes should not be None"
        assert t2_abab is not None, "T2 alpha-beta amplitudes should not be None"
        assert t2_aaaa is not None, "T2 alpha-alpha amplitudes should not be None"
        assert t2_bbbb is not None, "T2 beta-beta amplitudes should not be None"

        # Verify the amplitudes have the expected shapes (stored as column vectors)
        # T1: (nocc * nvirt, 1) for each spin
        # T2: (nocc * nocc * nvirt * nvirt, 1) for each component

        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        nocc_alpha = int(np.sum(occ_a))
        nocc_beta = int(np.sum(occ_b))
        nvirt_alpha = orbitals.get_num_molecular_orbitals() - nocc_alpha
        nvirt_beta = orbitals.get_num_molecular_orbitals() - nocc_beta

        t1_alpha_array = np.array(t1_alpha) if not isinstance(t1_alpha, np.ndarray) else t1_alpha
        t1_beta_array = np.array(t1_beta) if not isinstance(t1_beta, np.ndarray) else t1_beta

        assert t1_alpha_array.shape[0] == nocc_alpha * nvirt_alpha, "T1 alpha shape mismatch"
        assert t1_beta_array.shape[0] == nocc_beta * nvirt_beta, "T1 beta shape mismatch"

    def test_pyscf_uccsd_wavefunction_serialization_roundtrip(self, tmp_path):
        """Test that UCCSD wavefunction can be serialized and deserialized via JSON and HDF5."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(o2, 0, 3, "sto-3g")

        ham_calculator = algorithms.create("hamiltonian_constructor", "qdk")
        hamiltonian = ham_calculator.run(wavefunction.get_orbitals())

        # Compute UCCSD energy with amplitudes stored
        cc_calculator = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        cc_calculator.settings().set("store_amplitudes", True)
        ansatz_object = Ansatz(hamiltonian, wavefunction)
        _, cc_wavefunction = cc_calculator.run(ansatz_object)

        # Verify original wavefunction properties
        assert cc_wavefunction.get_container_type() == "coupled_cluster"

        # Get original container and check it has amplitudes
        original_container = cc_wavefunction.get_container()
        assert original_container.has_t1_amplitudes()
        assert original_container.has_t2_amplitudes()

        # Get original amplitudes - unrestricted has separate alpha/beta
        orig_t1_alpha, orig_t1_beta = original_container.get_t1_amplitudes()
        orig_t2_abab, orig_t2_aaaa, orig_t2_bbbb = original_container.get_t2_amplitudes()

        # Verify all amplitudes are present
        assert orig_t1_alpha is not None
        assert orig_t1_beta is not None
        assert orig_t2_abab is not None
        assert orig_t2_aaaa is not None
        assert orig_t2_bbbb is not None

        # Get original orbitals properties
        orig_orbitals = cc_wavefunction.get_orbitals()
        orig_num_orbs = orig_orbitals.get_num_molecular_orbitals()
        orig_is_unrestricted = orig_orbitals.is_unrestricted()

        orig_num_elec = cc_wavefunction.get_total_num_electrons()

        # Test 1: JSON serialization
        wf_json = cc_wavefunction.to_json()
        restored_json = data.Wavefunction.from_json(wf_json)

        # Verify JSON restored wavefunction
        assert restored_json.get_container_type() == "coupled_cluster"

        json_container = restored_json.get_container()
        assert json_container.has_t1_amplitudes()
        assert json_container.has_t2_amplitudes()

        # Verify amplitudes are preserved - unrestricted version
        json_t1_alpha, json_t1_beta = json_container.get_t1_amplitudes()
        json_t2_abab, json_t2_aaaa, json_t2_bbbb = json_container.get_t2_amplitudes()

        assert np.allclose(
            np.array(orig_t1_alpha),
            np.array(json_t1_alpha),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t1_beta),
            np.array(json_t1_beta),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_abab),
            np.array(json_t2_abab),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_aaaa),
            np.array(json_t2_aaaa),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_bbbb),
            np.array(json_t2_bbbb),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify orbitals properties preserved
        json_orbitals = restored_json.get_orbitals()
        assert json_orbitals.get_num_molecular_orbitals() == orig_num_orbs
        assert json_orbitals.is_unrestricted() == orig_is_unrestricted

        # Verify electron count preserved
        json_num_elec = restored_json.get_total_num_electrons()
        assert json_num_elec == orig_num_elec

        # Test 2: HDF5 serialization
        filename = tmp_path / "uccsd_wf.hdf5"
        cc_wavefunction.to_hdf5_file(str(filename))
        restored_hdf5 = data.Wavefunction.from_hdf5_file(str(filename))

        # Verify HDF5 restored wavefunction
        assert restored_hdf5.get_container_type() == "coupled_cluster"

        hdf5_container = restored_hdf5.get_container()
        assert hdf5_container.has_t1_amplitudes()
        assert hdf5_container.has_t2_amplitudes()

        # Verify amplitudes are preserved - unrestricted version
        hdf5_t1_alpha, hdf5_t1_beta = hdf5_container.get_t1_amplitudes()
        hdf5_t2_abab, hdf5_t2_aaaa, hdf5_t2_bbbb = hdf5_container.get_t2_amplitudes()

        assert np.allclose(
            np.array(orig_t1_alpha),
            np.array(hdf5_t1_alpha),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t1_beta),
            np.array(hdf5_t1_beta),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_abab),
            np.array(hdf5_t2_abab),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_aaaa),
            np.array(hdf5_t2_aaaa),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2_bbbb),
            np.array(hdf5_t2_bbbb),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify orbitals properties preserved
        hdf5_orbitals = restored_hdf5.get_orbitals()
        assert hdf5_orbitals.get_num_molecular_orbitals() == orig_num_orbs
        assert hdf5_orbitals.is_unrestricted() == orig_is_unrestricted

        # Verify electron count preserved
        hdf5_num_elec = restored_hdf5.get_total_num_electrons()
        assert hdf5_num_elec == orig_num_elec

    def test_pyscf_ccsd_wavefunction_serialization_roundtrip(self, tmp_path):
        """Test that CCSD wavefunction can be serialized and deserialized with json and hdf5."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")

        ham_calculator = algorithms.create("hamiltonian_constructor", "qdk")
        hamiltonian = ham_calculator.run(wavefunction.get_orbitals())

        # Compute CC energy with amplitudes stored
        cc_calculator = algorithms.create("dynamical_correlation_calculator", "pyscf_coupled_cluster")
        cc_calculator.settings().set("store_amplitudes", True)
        ansatz_object = Ansatz(hamiltonian, wavefunction)
        _, cc_wavefunction = cc_calculator.run(ansatz_object)

        # Verify original wavefunction properties
        assert cc_wavefunction.get_container_type() == "coupled_cluster"

        # Get original container and check it has amplitudes
        original_container = cc_wavefunction.get_container()
        assert original_container.has_t1_amplitudes()
        assert original_container.has_t2_amplitudes()

        # Get original amplitudes
        orig_t1 = original_container.get_t1_amplitudes()
        orig_t2 = original_container.get_t2_amplitudes()
        assert orig_t1 is not None
        assert orig_t2 is not None

        # Get original orbitals properties
        orig_orbitals = cc_wavefunction.get_orbitals()
        orig_num_orbs = orig_orbitals.get_num_molecular_orbitals()
        orig_is_restricted = orig_orbitals.is_restricted()

        orig_num_elec = cc_wavefunction.get_total_num_electrons()

        # Test 1: JSON serialization
        wf_json = cc_wavefunction.to_json()
        restored_json = data.Wavefunction.from_json(wf_json)

        # Verify JSON restored wavefunction
        assert restored_json.get_container_type() == "coupled_cluster"

        json_container = restored_json.get_container()
        assert json_container.has_t1_amplitudes()
        assert json_container.has_t2_amplitudes()

        # Verify amplitudes are preserved
        json_t1 = json_container.get_t1_amplitudes()
        json_t2 = json_container.get_t2_amplitudes()
        assert np.allclose(
            np.array(orig_t1),
            np.array(json_t1),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2),
            np.array(json_t2),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify orbitals properties preserved
        json_orbitals = restored_json.get_orbitals()
        assert json_orbitals.get_num_molecular_orbitals() == orig_num_orbs
        assert json_orbitals.is_restricted() == orig_is_restricted

        # Verify electron count preserved
        json_num_elec = restored_json.get_total_num_electrons()
        assert json_num_elec == orig_num_elec

        # Test 2: HDF5 serialization
        filename = tmp_path / "cc_wf.hdf5"
        cc_wavefunction.to_hdf5_file(str(filename))
        restored_hdf5 = data.Wavefunction.from_hdf5_file(str(filename))

        # Verify HDF5 restored wavefunction
        assert restored_hdf5.get_container_type() == "coupled_cluster"

        hdf5_container = restored_hdf5.get_container()
        assert hdf5_container.has_t1_amplitudes()
        assert hdf5_container.has_t2_amplitudes()

        # Verify amplitudes are preserved
        hdf5_t1 = hdf5_container.get_t1_amplitudes()
        hdf5_t2 = hdf5_container.get_t2_amplitudes()
        assert np.allclose(
            np.array(orig_t1),
            np.array(hdf5_t1),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            np.array(orig_t2),
            np.array(hdf5_t2),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify orbitals properties preserved
        hdf5_orbitals = restored_hdf5.get_orbitals()
        assert hdf5_orbitals.get_num_molecular_orbitals() == orig_num_orbs
        assert hdf5_orbitals.is_restricted() == orig_is_restricted

        # Verify electron count preserved
        hdf5_num_elec = restored_hdf5.get_total_num_electrons()
        assert hdf5_num_elec == orig_num_elec

    def test_pyscf_mcscf_singlet(self):
        """Test PySCF MCSCF for n2 with cc-pvdz basis and CAS(6,6)."""
        # Create N2 structure
        n2 = create_n2_structure()

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(n2, 0, 1, "cc-pvdz")

        # Construct qdk-chemistry Hamiltonian for active space
        ham_calculator = algorithms.create("hamiltonian_constructor")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Select active space: 6 orbitals, 6 electrons
        valence_selector = algorithms.create("active_space_selector", "qdk_valence")
        valence_selector.settings().set("num_active_electrons", 6)
        valence_selector.settings().set("num_active_orbitals", 6)
        active_orbitals_sd = valence_selector.run(wavefunction)

        # Calculate with qdk-chemistry/MACIS
        pyscf_mcscf = algorithms.create("multi_configuration_scf", "pyscf")
        pyscf_mcscf_energy, _ = pyscf_mcscf.run(active_orbitals_sd.get_orbitals(), ham_calculator, macis_calc, 3, 3)

        assert np.isclose(
            pyscf_mcscf_energy,
            -108.78966139913287,
            rtol=float_comparison_relative_tolerance,
            atol=mcscf_energy_tolerance,
        )

    def test_pyscf_mcscf_triplet(self):
        """Test PySCF MCSCF for o2 triplet with cc-pvdz basis and CAS(6,6)."""
        # Create O2 structure
        o2 = create_o2_structure()

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")
        _, wavefunction = scf_solver.run(o2, 0, 3, "cc-pvdz")

        # Construct qdk-chemistry Hamiltonian for active space
        ham_calculator = algorithms.create("hamiltonian_constructor")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)
        macis_calc.settings().set("ci_residual_tolerance", 1e-10)

        # Select active space: 6 orbitals, 6 electrons
        valence_selector = algorithms.create("active_space_selector", "qdk_valence")
        valence_selector.settings().set("num_active_electrons", 6)
        valence_selector.settings().set("num_active_orbitals", 6)
        active_orbitals_sd = valence_selector.run(wavefunction)

        # Calculate with qdk-chemistry/MACIS
        pyscf_mcscf = algorithms.create("multi_configuration_scf", "pyscf")
        pyscf_mcscf_energy, _ = pyscf_mcscf.run(active_orbitals_sd.get_orbitals(), ham_calculator, macis_calc, 4, 2)

        assert np.isclose(
            pyscf_mcscf_energy,
            -149.68131616317658,
            rtol=float_comparison_relative_tolerance,
            atol=mcscf_energy_tolerance,
        )

    def test_pyscf_fciwrapper_casci_singlet(self):
        """Test MC wrapper for n2 with cc-pvdz basis and CAS(6,6)."""
        # Create N2 structure
        n2 = create_n2_structure()
        pyscf_mol = pyscf.gto.M(
            atom=structure_to_pyscf_atom_labels(n2)[0], basis="cc-pvdz", unit="Bohr", charge=0, spin=0
        )

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(n2, 0, 1, "cc-pvdz")

        # Get pyscf object from hf orbitals
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        pyscf_scf = orbitals_to_scf(wavefunction.get_orbitals(), occ_alpha=occ_a, occ_beta=occ_b, scf_type="restricted")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Create PySCF CASCI calculation with macis
        casci = pyscf.mcscf.CASCI(pyscf_scf, 6, 6)
        casci.fcisolver = _mcsolver_to_fcisolver(pyscf_mol, macis_calc)
        casci.verbose = 0
        casci_energy = casci.kernel()[0]

        assert np.isclose(
            casci_energy, -108.74113344655625, rtol=float_comparison_relative_tolerance, atol=mcscf_energy_tolerance
        )

    def test_pyscf_fciwrapper_casci_triplet(self):
        """Test MC wrapper for o2 triplet with cc-pvdz basis and CAS(6,6)."""
        # Create O2 structure
        o2 = create_o2_structure()
        pyscf_mol = pyscf.gto.M(atom=structure_to_pyscf_atom_labels(o2)[0], basis="cc-pvdz", charge=0, spin=2)

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(o2, 0, 3, "cc-pvdz")

        # Get pyscf object from hf orbitals
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        pyscf_scf = orbitals_to_scf(wavefunction.get_orbitals(), occ_alpha=occ_a, occ_beta=occ_b, scf_type="restricted")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Create PySCF CASCI calculation with macis
        casci = pyscf.mcscf.CASCI(pyscf_scf, 8, 6)
        casci.fcisolver = _mcsolver_to_fcisolver(pyscf_mol, macis_calc)
        casci.verbose = 0
        casci_energy = casci.kernel()[0]

        assert np.isclose(
            casci_energy, -149.661310389037, rtol=float_comparison_relative_tolerance, atol=mcscf_energy_tolerance
        )

    def test_pyscf_fciwrapper_casscf_singlet(self):
        """Test MC wrapper in casscf for n2 with cc-pvdz basis and CAS(6,6)."""
        # Create N2 structure
        n2 = create_n2_structure()
        pyscf_mol = pyscf.gto.M(
            atom=structure_to_pyscf_atom_labels(n2)[0], basis="cc-pvdz", unit="Bohr", charge=0, spin=0
        )

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(n2, 0, 1, "cc-pvdz")

        # Get pyscf object from hf orbitals
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        pyscf_scf = orbitals_to_scf(wavefunction.get_orbitals(), occ_alpha=occ_a, occ_beta=occ_b, scf_type="restricted")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)

        # Create PySCF CASSCF calculation with macis
        casscf = pyscf.mcscf.CASSCF(pyscf_scf, 6, 6)
        casscf.fcisolver = _mcsolver_to_fcisolver(pyscf_mol, macis_calc)
        casscf.verbose = 0
        casscf_energy = casscf.kernel()[0]

        assert np.isclose(
            casscf_energy, -108.78966139913287, rtol=float_comparison_relative_tolerance, atol=mcscf_energy_tolerance
        )

    def test_pyscf_fciwrapper_casscf_triplet(self):
        """Test MC wrapper in casscf for o2 triplet with cc-pvdz basis and CAS(6,6)."""
        # Create O2 structure
        o2 = create_o2_structure()
        pyscf_mol = pyscf.gto.M(atom=structure_to_pyscf_atom_labels(o2)[0], basis="cc-pvdz", charge=0, spin=2)

        # Perform SCF calculation with qdk-chemistry
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")
        _, wavefunction = scf_solver.run(o2, 0, 3, "cc-pvdz")

        # Get pyscf object from hf orbitals
        occ_a, occ_b = wavefunction.get_total_orbital_occupations()
        pyscf_scf = orbitals_to_scf(wavefunction.get_orbitals(), occ_alpha=occ_a, occ_beta=occ_b, scf_type="restricted")

        # Create MACIS calculator
        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)
        macis_calc.settings().set("ci_residual_tolerance", 1e-10)

        # Create PySCF CASSCF calculation with macis
        casscf = pyscf.mcscf.CASSCF(pyscf_scf, 6, 6)
        casscf.fcisolver = _mcsolver_to_fcisolver(pyscf_mol, macis_calc)
        casscf.verbose = 0
        casscf_energy = casscf.kernel()[0]

        assert np.isclose(
            casscf_energy, -149.68131616317658, rtol=float_comparison_relative_tolerance, atol=mcscf_energy_tolerance
        )

    def test_pyscf_occupations_from_n_electrons_and_multiplicity(self):
        """Test occupations from n_electrons and multiplicity on water with def2-svp basis."""
        # Get orbitals and Hamiltonian
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()
        hamiltonian_calculator = algorithms.create("hamiltonian_constructor")
        hamiltonian = hamiltonian_calculator.run(orbitals)

        # Check orbitals to SCF for singlet state
        occupation_singlet = [np.concatenate((np.ones(5), np.zeros(19))), np.concatenate((np.ones(5), np.zeros(19)))]
        scf_1 = orbitals_to_scf(orbitals, occupation_singlet[0], occupation_singlet[1])
        scf_2 = orbitals_to_scf_from_n_electrons_and_multiplicity(orbitals, 10, 1)
        assert np.allclose(
            scf_1.mo_occ,
            scf_2.mo_occ,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check orbitals to SCF for doublet state
        occupation_doublet = [np.concatenate((np.ones(6), np.zeros(18))), np.concatenate((np.ones(5), np.zeros(19)))]
        scf_1 = orbitals_to_scf(orbitals, occupation_doublet[0], occupation_doublet[1])
        scf_2 = orbitals_to_scf_from_n_electrons_and_multiplicity(orbitals, 11, 2)
        assert np.allclose(
            scf_1.mo_occ,
            scf_2.mo_occ,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check orbitals to SCF for triplet state
        occupation_triplet = [np.concatenate((np.ones(6), np.zeros(18))), np.concatenate((np.ones(4), np.zeros(20)))]
        scf_1 = orbitals_to_scf(orbitals, occupation_triplet[0], occupation_triplet[1])
        scf_2 = orbitals_to_scf_from_n_electrons_and_multiplicity(orbitals, 10, 3)
        assert np.allclose(
            scf_1.mo_occ,
            scf_2.mo_occ,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check Hamiltonian to SCF for singlet state
        scf_1 = hamiltonian_to_scf(hamiltonian, occupation_singlet[0], occupation_singlet[1])
        scf_2 = hamiltonian_to_scf_from_n_electrons_and_multiplicity(hamiltonian, 10, 1)
        assert np.allclose(
            scf_1.mo_occ,
            scf_2.mo_occ,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_orbitals_to_scf_charge_and_multiplicity_handling(self):
        """Test that orbitals_to_scf correctly sets charge and multiplicity in the PySCF molecule."""
        scf_solver = algorithms.create("scf_solver", "pyscf")

        # Test singlet state (closed-shell)
        water = create_water_structure()
        _, wavefunction_singlet = scf_solver.run(water, 0, 1, "sto-3g")
        orbitals_singlet = wavefunction_singlet.get_orbitals()
        n_orbitals_singlet = orbitals_singlet.get_num_molecular_orbitals()

        # Singlet: 5 alpha, 5 beta electrons (charge = 0, multiplicity = 1)
        occ_alpha_singlet = np.concatenate((np.ones(5), np.zeros(n_orbitals_singlet - 5)))
        occ_beta_singlet = np.concatenate((np.ones(5), np.zeros(n_orbitals_singlet - 5)))
        scf_singlet = orbitals_to_scf(orbitals_singlet, occ_alpha_singlet, occ_beta_singlet)

        assert scf_singlet.mol.charge == 0
        assert scf_singlet.mol.spin == 0
        assert scf_singlet.mol.multiplicity == 1

        # Test doublet state (open-shell)
        lithium = create_li_structure()
        _, wavefunction_doublet = scf_solver.run(lithium, 0, 2, "sto-3g")
        orbitals_doublet = wavefunction_doublet.get_orbitals()
        n_orbitals_doublet = orbitals_doublet.get_num_molecular_orbitals()

        # Doublet: 2 alpha, 1 beta electrons (charge = 0, multiplicity = 2)
        occ_alpha_doublet = np.concatenate((np.ones(2), np.zeros(n_orbitals_doublet - 2)))
        occ_beta_doublet = np.concatenate((np.ones(1), np.zeros(n_orbitals_doublet - 1)))
        scf_doublet = orbitals_to_scf(orbitals_doublet, occ_alpha_doublet, occ_beta_doublet)

        assert scf_doublet.mol.charge == 0
        assert scf_doublet.mol.spin == 1
        assert scf_doublet.mol.multiplicity == 2

        # Test triplet state (open-shell)
        o2 = create_o2_structure()
        _, wavefunction_triplet = scf_solver.run(o2, 0, 3, "sto-3g")
        orbitals_triplet = wavefunction_triplet.get_orbitals()
        n_orbitals_triplet = orbitals_triplet.get_num_molecular_orbitals()

        # Triplet: 9 alpha, 7 beta electrons (charge = 0, multiplicity = 3)
        occ_alpha_triplet = np.concatenate((np.ones(9), np.zeros(n_orbitals_triplet - 9)))
        occ_beta_triplet = np.concatenate((np.ones(7), np.zeros(n_orbitals_triplet - 7)))
        scf_triplet = orbitals_to_scf(orbitals_triplet, occ_alpha_triplet, occ_beta_triplet)

        assert scf_triplet.mol.charge == 0
        assert scf_triplet.mol.spin == 2
        assert scf_triplet.mol.multiplicity == 3

        # Test cation (open-shell)
        water = create_water_structure()
        _, wavefunction_cation = scf_solver.run(water, 1, 2, "sto-3g")
        orbitals_cation = wavefunction_cation.get_orbitals()
        n_orbitals_cation = orbitals_cation.get_num_molecular_orbitals()

        # Doublet: 5 alpha, 4 beta electrons (charge = 1, multiplicity = 2)
        occ_alpha_cation = np.concatenate((np.ones(5), np.zeros(n_orbitals_cation - 5)))
        occ_beta_cation = np.concatenate((np.ones(4), np.zeros(n_orbitals_cation - 4)))
        scf_cation = orbitals_to_scf(orbitals_cation, occ_alpha_cation, occ_beta_cation)

        assert scf_cation.mol.charge == 1
        assert scf_cation.mol.spin == 1
        assert scf_cation.mol.multiplicity == 2
        assert scf_cation.mol.nelectron == 9

        # Test with ECP electrons
        ag = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))
        _, wavefunction_ecp = scf_solver.run(ag, 0, 2, "lanl2dz")
        orbitals_ecp = wavefunction_ecp.get_orbitals()
        n_orbitals_ecp = orbitals_ecp.get_num_molecular_orbitals()

        # Doublet: 10 alpha, 9 beta electrons (charge = 0, multiplicity = 2)
        occ_alpha_ecp = np.concatenate((np.ones(10), np.zeros(n_orbitals_ecp - 10)))
        occ_beta_ecp = np.concatenate((np.ones(9), np.zeros(n_orbitals_ecp - 9)))
        scf_ecp = orbitals_to_scf(orbitals_ecp, occ_alpha_ecp, occ_beta_ecp)

        assert hasattr(scf_ecp.mol, "ecp")
        assert scf_ecp.mol.ecp
        assert scf_ecp.mol.charge == 0
        assert scf_ecp.mol.spin == 1
        assert scf_ecp.mol.multiplicity == 2
        assert scf_ecp.mol.nelectron == 19

    def test_hamiltonian_to_scf_rerouting_and_error_handling(self):
        """Test hamiltonian_to_scf rerouting and error handling.

        This test validates three scenarios:
        1. Rerouting: Non-model Hamiltonians should reroute to orbitals_to_scf
        2. Error throwing: Invalid configurations should raise ValueError
        3. Non-rerouting: Valid model Hamiltonians should create fake SCF objects
        """
        # 1: Rerouting for non-model Hamiltonian
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")
        orbitals = wavefunction.get_orbitals()

        # Create a Hamiltonian from these orbitals (non-model)
        hamiltonian_calculator = algorithms.create("hamiltonian_constructor")
        hamiltonian = hamiltonian_calculator.run(orbitals)

        # Verify the orbitals have coefficients (non-model)
        coeff_a, coeff_b = orbitals.get_coefficients()
        assert coeff_a is not None
        assert coeff_b is not None

        # Create occupation arrays for a singlet state
        norb = orbitals.get_num_molecular_orbitals()
        occupation_alpha = np.concatenate((np.ones(5), np.zeros(norb - 5)))
        occupation_beta = np.concatenate((np.ones(5), np.zeros(norb - 5)))

        # Call hamiltonian_to_scf - should re-route to orbitals_to_scf
        scf_from_hamiltonian = hamiltonian_to_scf(hamiltonian, occupation_alpha, occupation_beta)

        # MO coefficients should NOT be identity matrix
        assert not np.allclose(scf_from_hamiltonian.mo_coeff, np.eye(norb)), (
            "MO coefficients should not be identity matrix for non-model Hamiltonian (rerouted case)"
        )

        # 2. Unrestricted model Hamiltonian should throw
        model_orbitals_unrestricted = data.ModelOrbitals(4, False)  # unrestricted
        one_body_alpha = np.eye(4)
        one_body_beta = np.eye(4) * 1.1
        two_body_aaaa = np.zeros(4**4)
        two_body_aabb = np.zeros(4**4)
        two_body_bbbb = np.zeros(4**4)
        h_unrestricted_model = data.Hamiltonian(
            data.CanonicalFourCenterHamiltonianContainer(
                one_body_alpha,
                one_body_beta,
                two_body_aaaa,
                two_body_aabb,
                two_body_bbbb,
                model_orbitals_unrestricted,
                0.0,
                np.eye(4),
                np.eye(4),
            )
        )

        occupation_alpha_test = np.array([1.0, 1.0, 0.0, 0.0])
        occupation_beta_test = np.array([1.0, 1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="You cannot pass an unrestricted model Hamiltonian here"):
            hamiltonian_to_scf(h_unrestricted_model, occupation_alpha_test, occupation_beta_test)

        # 3. Non-rerouting for valid model Hamiltonian

        # Create a model Hamiltonian (restricted, closed-shell, full active space)
        model_orbitals_proper = data.ModelOrbitals(4, True)  # All orbitals are active by default
        one_body_model = np.eye(4) * 0.5
        two_body_model = np.zeros(4**4)
        h_model = data.Hamiltonian(
            data.CanonicalFourCenterHamiltonianContainer(
                one_body_model, two_body_model, model_orbitals_proper, 0.5, np.eye(4)
            )
        )

        # Closed-shell occupations
        occupation_alpha_closed = np.array([1.0, 1.0, 0.0, 0.0])
        occupation_beta_closed = np.array([1.0, 1.0, 0.0, 0.0])

        # Call hamiltonian_to_scf - should create fake SCF
        scf_from_model = hamiltonian_to_scf(h_model, occupation_alpha_closed, occupation_beta_closed)

        # For model Hamiltonian - MO coefficients should be identity matrix (fake SCF)
        assert np.allclose(scf_from_model.mo_coeff, np.eye(4)), (
            "MO coefficients should be identity matrix for model Hamiltonian (fake SCF object)"
        )

        # Verify core energy matches
        assert np.isclose(scf_from_model.energy_nuc(), 0.5), "Core energy should match the model Hamiltonian"

        # Verify occupations are set correctly (total = alpha + beta for restricted)
        expected_total_occ = occupation_alpha_closed + occupation_beta_closed
        assert np.allclose(scf_from_model.mo_occ, expected_total_occ), (
            "Occupations should be correctly set in fake SCF object"
        )

        # Verify electron count
        assert scf_from_model.mol.nelectron == 4, "Total electron count should be 4"


class TestQDKChemistryPySCFBasisConversion:
    """Test suite for QDK/Chemistry-PySCF basis set conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple helium structure
        self.he_structure = Structure(["He"], np.array([[0.0, 0.0, 0.0]]))

        # Create a water structure
        self.h2o_structure = Structure(
            ["O", "H", "H"],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.757, 0.586, 0.0],
                    [-0.757, 0.586, 0.0],
                ]
            ),
        )

        # Create a hydrogen molecule structure
        self.h2_structure = Structure(
            ["H", "H"],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.4, 0.0, 0.0],
                ]
            ),
        )

    def create_simple_basis_set(self, structure, basis_name="STO-3G"):
        """Create a simple basis set for testing."""
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(structure, 0, 1, basis_name.lower())
        return wavefunction.get_orbitals().get_basis_set()

    def test_qdk_to_pyscf_conversion_helium(self):
        """Test converting QDK/Chemistry basis set to PySCF for helium."""
        qdk_basis = self.create_simple_basis_set(self.he_structure)

        # Convert to PySCF
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)

        # Verify structure
        assert pyscf_mol.natm == 1
        assert pyscf_mol.atom_charges()[0] == 2  # Helium
        assert pyscf_mol.nao_nr() == qdk_basis.get_num_atomic_orbitals()

        # Verify coordinates match
        coords = pyscf_mol.atom_coords()
        assert np.allclose(
            coords[0], [0.0, 0.0, 0.0], rtol=float_comparison_relative_tolerance, atol=plain_text_tolerance
        )

    def test_qdk_to_pyscf_conversion_water(self):
        """Test converting QDK/Chemistry basis set to PySCF for water."""
        qdk_basis = self.create_simple_basis_set(self.h2o_structure)

        # Convert to PySCF
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)

        # Verify structure
        assert pyscf_mol.natm == 3
        assert pyscf_mol.atom_charges()[0] == 8  # Oxygen
        assert pyscf_mol.atom_charges()[1] == 1  # Hydrogen
        assert pyscf_mol.atom_charges()[2] == 1  # Hydrogen
        assert pyscf_mol.nao_nr() == qdk_basis.get_num_atomic_orbitals()

    def test_pyscf_to_qdk_conversion_helium(self):
        """Test converting PySCF molecule to QDK/Chemistry basis set for helium."""
        # Create PySCF molecule
        pyscf_mol = pyscf.gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)

        # Convert to QDK/Chemistry
        qdk_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.he_structure)

        # Verify basic properties
        assert qdk_basis.get_num_atoms() == 1
        assert qdk_basis.get_num_atomic_orbitals() == pyscf_mol.nao_nr()
        assert qdk_basis.has_structure()

    def test_pyscf_to_qdk_conversion_water(self):
        """Test converting PySCF molecule to QDK/Chemistry basis set for water."""
        # Create PySCF molecule
        pyscf_mol = pyscf.gto.M(
            atom="""
            O  0.0      0.0      0.0
            H  0.757    0.586    0.0
            H -0.757    0.586    0.0
            """,
            basis="sto-3g",
            verbose=0,
        )

        # Convert to QDK/Chemistry
        qdk_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.h2o_structure)

        # Verify basic properties
        assert qdk_basis.get_num_atoms() == 3
        assert qdk_basis.get_num_atomic_orbitals() == pyscf_mol.nao_nr()
        assert qdk_basis.has_structure()

    def test_pyscf_to_qdk_conversion_water_generally_contracted(self):
        """Test converting Pyscf molecule to QDK/Chemistry basis set for water with a generally contracted basis."""
        # Create PySCF molecule
        pyscf_mol = pyscf.gto.M(
            atom="""
             O  0.0      0.0      0.0
             H  0.757    0.586    0.0
             H -0.757    0.586    0.0
             """,
            basis="cc-pvdz",
            verbose=0,
        )

        # Convert to QDK/Chemistry
        qdk_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.h2o_structure)

        # Verify basic properties
        assert qdk_basis.get_num_atoms() == 3
        assert qdk_basis.get_num_atomic_orbitals() == pyscf_mol.nao_nr()
        assert qdk_basis.has_structure()

    def test_round_trip_conversion_helium(self):
        """Test round-trip conversion: QDK/Chemistry -> PySCF -> QDK/Chemistry for helium."""
        # Start with QDK/Chemistry basis
        original_basis = self.create_simple_basis_set(self.he_structure)

        # Convert to PySCF
        pyscf_mol = basis_to_pyscf_mol(original_basis)

        # Convert back to QDK/Chemistry
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.he_structure)

        # Compare key properties
        assert original_basis.get_num_atoms() == converted_basis.get_num_atoms()
        assert original_basis.get_num_shells() == converted_basis.get_num_shells()
        assert original_basis.get_num_atomic_orbitals() == converted_basis.get_num_atomic_orbitals()

        # Compare shell-by-shell (with more relaxed tolerances for numerical differences)
        orig_shells = original_basis.get_shells()
        conv_shells = converted_basis.get_shells()

        for i, (orig_shell, conv_shell) in enumerate(zip(orig_shells, conv_shells, strict=True)):
            assert orig_shell.atom_index == conv_shell.atom_index, f"Shell {i}: atom index mismatch"
            assert orig_shell.orbital_type == conv_shell.orbital_type, f"Shell {i}: orbital type mismatch"
            assert orig_shell.get_num_primitives() == conv_shell.get_num_primitives(), (
                f"Shell {i}: primitive count mismatch"
            )
            # Compare exponents and coefficients within tolerance (more relaxed for conversion)
            try:
                assert np.allclose(
                    orig_shell.exponents,
                    conv_shell.exponents,
                    rtol=plain_text_tolerance / 100,
                    atol=plain_text_tolerance,
                ), f"Shell {i}: exponent mismatch"
                assert np.allclose(
                    orig_shell.coefficients,
                    conv_shell.coefficients,
                    rtol=plain_text_tolerance / 100,
                    atol=plain_text_tolerance,
                ), f"Shell {i}: coefficient mismatch"
            except AssertionError:
                # If exact comparison fails, just verify they're reasonably close
                exp_diff = np.max(np.abs(orig_shell.exponents - conv_shell.exponents))
                coeff_diff = np.max(np.abs(orig_shell.coefficients - conv_shell.coefficients))
                assert exp_diff < plain_text_tolerance, f"Shell {i}: large exponent difference {exp_diff}"
                assert coeff_diff < plain_text_tolerance, f"Shell {i}: large coefficient difference {coeff_diff}"

    def test_round_trip_conversion_water(self):
        """Test round-trip conversion: QDK/Chemistry -> PySCF -> QDK/Chemistry for water."""
        # Start with QDK/Chemistry basis
        original_basis = self.create_simple_basis_set(self.h2o_structure)

        # Convert to PySCF
        pyscf_mol = basis_to_pyscf_mol(original_basis)

        # Convert back to QDK/Chemistry
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.h2o_structure)

        # Compare key properties
        assert original_basis.get_num_atoms() == converted_basis.get_num_atoms()
        assert original_basis.get_num_shells() == converted_basis.get_num_shells()
        assert original_basis.get_num_atomic_orbitals() == converted_basis.get_num_atomic_orbitals()

    def test_atomic_orbital_type_handling(self):
        """Test handling of spherical vs cartesian basis types."""
        # Create basis with spherical functions
        qdk_basis = self.create_simple_basis_set(self.he_structure)
        original_type = qdk_basis.get_atomic_orbital_type()

        # Convert to PySCF and back
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.he_structure)

        # Basis type should be preserved
        assert converted_basis.get_atomic_orbital_type() == original_type

    def test_shell_ordering_consistency(self):
        """Test that shell ordering is consistent after conversion."""
        qdk_basis = self.create_simple_basis_set(self.h2o_structure)

        # Get original shell ordering
        orig_shells = qdk_basis.get_shells()
        orig_shell_types = [shell.orbital_type for shell in orig_shells]
        orig_atom_indices = [shell.atom_index for shell in orig_shells]

        # Convert to PySCF and back
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.h2o_structure)

        # Get converted shell ordering
        conv_shells = converted_basis.get_shells()
        conv_shell_types = [shell.orbital_type for shell in conv_shells]
        conv_atom_indices = [shell.atom_index for shell in conv_shells]

        # Shell ordering should match
        assert orig_shell_types == conv_shell_types
        assert orig_atom_indices == conv_atom_indices

    def test_molecular_orbital_consistency(self):
        """Test that molecular orbitals remain consistent after basis conversion."""
        # Get QDK/Chemistry solution
        scf_solver = algorithms.create("scf_solver", "pyscf")
        qdk_energy, qdk_wavefunction = scf_solver.run(self.he_structure, 0, 1, "sto-3g")
        qdk_orbitals = qdk_wavefunction.get_orbitals()
        qdk_mos = qdk_orbitals.get_coefficients()[0]

        # Convert basis and solve with PySCF
        qdk_basis = qdk_orbitals.get_basis_set()
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)
        mf = pyscf.scf.RHF(pyscf_mol)
        mf.verbose = 0
        pyscf_energy = mf.kernel()
        pyscf_mos = mf.mo_coeff

        # Energies should match
        qdk_total_energy = qdk_energy + self.he_structure.calculate_nuclear_repulsion_energy()
        assert np.allclose(
            qdk_total_energy, pyscf_energy, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # MO coefficients should be similar (up to phase)
        # For helium with STO-3G, we expect 1 occupied orbital
        qdk_homo = qdk_mos[:, 0]
        pyscf_homo = pyscf_mos[:, 0]

        # Check if coefficients match (considering possible sign flip)
        overlap = np.abs(np.dot(qdk_homo, pyscf_homo))
        assert np.allclose(overlap, 1.0, rtol=float_comparison_relative_tolerance, atol=scf_orbital_tolerance)

    def test_basis_set_metadata_preservation(self):
        """Test that basis set metadata is preserved during conversion."""
        qdk_basis = self.create_simple_basis_set(self.he_structure)
        original_name = qdk_basis.get_name()

        # Convert to PySCF and back
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.he_structure)

        # Name should be preserved
        converted_name = converted_basis.get_name()
        assert original_name == converted_name

    def test_ecp_extraction_and_metadata(self):
        """Test that ECP shells and metadata are properly extracted from PySCF."""
        # Create a structure with heavy atoms that use ECPs
        ag_structure = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))

        # Create PySCF molecule with ECP
        pyscf_mol = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp="lanl2dz", verbose=0)

        # Convert to QDK/Chemistry basis
        qdk_basis = pyscf_mol_to_qdk_basis(pyscf_mol, ag_structure)

        # Verify ECP shells were extracted with radial powers
        assert qdk_basis.has_ecp_shells()
        assert qdk_basis.get_num_ecp_shells() > 0

        ecp_shells = qdk_basis.get_ecp_shells()
        for shell in ecp_shells:
            assert shell.has_radial_powers()
            assert len(shell.rpowers) > 0
            assert len(shell.rpowers) == len(shell.exponents) == len(shell.coefficients)

        # Verify ECP metadata
        assert qdk_basis.has_ecp_electrons()
        assert qdk_basis.get_ecp_name() == "lanl2dz"

        # Check ECP electron counts
        ecp_electrons = qdk_basis.get_ecp_electrons()
        assert len(ecp_electrons) == 1
        assert ecp_electrons[0] == 28  # LANL2DZ ECP for Ag removes 28 core electrons

    def test_ecp_roundtrip_conversion(self):
        """Test round-trip conversion of ECP shells and metadata: QDK -> PySCF -> QDK."""
        ag_structure = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))

        # Create PySCF molecule with ECP
        pyscf_mol_orig = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp="lanl2dz", verbose=0)
        qdk_basis_orig = pyscf_mol_to_qdk_basis(pyscf_mol_orig, ag_structure)

        # Store original data
        orig_ecp_shells = qdk_basis_orig.get_ecp_shells()
        orig_ecp_name = qdk_basis_orig.get_ecp_name()

        # Convert to PySCF (need to preserve charge and multiplicity)
        pyscf_mol_converted = basis_to_pyscf_mol(qdk_basis_orig, charge=0, multiplicity=2)

        # Verify qdk_ecp_name attribute was stored
        assert hasattr(pyscf_mol_converted, "qdk_ecp_name")
        assert pyscf_mol_converted.qdk_ecp_name == orig_ecp_name

        # Convert back to QDK/Chemistry
        qdk_basis_roundtrip = pyscf_mol_to_qdk_basis(pyscf_mol_converted, ag_structure)

        # Verify ECP name preserved
        assert qdk_basis_roundtrip.get_ecp_name() == orig_ecp_name

        # Verify ECP shells preserved with high precision
        assert qdk_basis_roundtrip.has_ecp_shells()
        assert qdk_basis_roundtrip.get_num_ecp_shells() == len(orig_ecp_shells)

        roundtrip_ecp_shells = qdk_basis_roundtrip.get_ecp_shells()
        for orig_shell, rt_shell in zip(orig_ecp_shells, roundtrip_ecp_shells, strict=True):
            assert orig_shell.atom_index == rt_shell.atom_index
            assert orig_shell.orbital_type == rt_shell.orbital_type
            assert orig_shell.has_radial_powers() == rt_shell.has_radial_powers()
            assert np.allclose(
                orig_shell.exponents,
                rt_shell.exponents,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                orig_shell.coefficients,
                rt_shell.coefficients,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.array_equal(orig_shell.rpowers, rt_shell.rpowers)

    def test_ecp_multi_atom_and_formats(self):
        """Test ECP handling in multi-atom systems and different PySCF ECP formats."""
        # Test 1: Multi-atom system with mixed ECP/no-ECP atoms
        agh_structure = Structure(["Ag", "H"], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        pyscf_mol_mixed = pyscf.gto.M(
            atom="Ag 0 0 0; H 2 0 0", basis={"Ag": "lanl2dz", "H": "sto-3g"}, ecp={"Ag": "lanl2dz"}, verbose=0
        )
        qdk_basis_mixed = pyscf_mol_to_qdk_basis(pyscf_mol_mixed, agh_structure)

        assert qdk_basis_mixed.has_ecp_electrons()
        assert qdk_basis_mixed.has_ecp_shells()
        ecp_electrons = qdk_basis_mixed.get_ecp_electrons()
        assert ecp_electrons == [28, 0]  # Ag with ECP, H without
        assert len(qdk_basis_mixed.get_ecp_shells_for_atom(0)) > 0  # Ag has ECP shells
        assert len(qdk_basis_mixed.get_ecp_shells_for_atom(1)) == 0  # H has no ECP shells

        # Test 2: Different ECP specification formats (string vs dict)
        ag_structure = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))
        pyscf_mol_str = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp="lanl2dz", verbose=0)
        pyscf_mol_dict = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp={"Ag": "lanl2dz"}, verbose=0)

        qdk_basis_str = pyscf_mol_to_qdk_basis(pyscf_mol_str, ag_structure)
        qdk_basis_dict = pyscf_mol_to_qdk_basis(pyscf_mol_dict, ag_structure)

        # Both formats should give identical results
        assert qdk_basis_str.get_ecp_name() == qdk_basis_dict.get_ecp_name() == "lanl2dz"
        assert qdk_basis_str.get_ecp_electrons() == qdk_basis_dict.get_ecp_electrons()
        assert qdk_basis_str.get_num_ecp_shells() == qdk_basis_dict.get_num_ecp_shells()

    def test_ecp_edge_cases(self):
        """Test ECP edge cases: shells without metadata and full structure format."""
        ag_structure = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))

        # Edge case 1: ECP shells exist without ECP metadata
        shells = [Shell(0, OrbitalType.S, [1.0], [1.0])]
        ecp_shells = [Shell(0, OrbitalType.S, [10.0, 5.0], [50.0, 20.0], [0, 2])]
        qdk_basis_no_meta = BasisSet("test-basis", shells, ecp_shells, ag_structure, AOType.Spherical)

        assert qdk_basis_no_meta.has_ecp_shells()
        assert qdk_basis_no_meta.get_num_ecp_shells() == 1
        assert not qdk_basis_no_meta.has_ecp_electrons()  # No metadata set
        assert qdk_basis_no_meta.get_ecp_name() == "none"

        # Edge case 2: Full ECP structure format roundtrip
        pyscf_mol_orig = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp="lanl2dz", verbose=0)
        qdk_basis_1 = pyscf_mol_to_qdk_basis(pyscf_mol_orig, ag_structure)

        # Convert to PySCF (creates full ECP structure dict, need to preserve charge and multiplicity)
        pyscf_mol_1 = basis_to_pyscf_mol(qdk_basis_1, charge=0, multiplicity=2)

        # Verify full structure format: [ncore, [[l, terms], ...]]
        assert isinstance(pyscf_mol_1.ecp, dict)
        assert "Ag" in pyscf_mol_1.ecp
        ecp_data = pyscf_mol_1.ecp["Ag"]
        assert isinstance(ecp_data, list)
        assert len(ecp_data) >= 2
        assert isinstance(ecp_data[0], int)  # ncore
        assert isinstance(ecp_data[1], list)  # [[l, terms], ...]

        # Convert back (should handle full structure format)
        qdk_basis_2 = pyscf_mol_to_qdk_basis(pyscf_mol_1, ag_structure)

        # Verify complete preservation
        assert qdk_basis_2.has_ecp_electrons()
        assert qdk_basis_2.has_ecp_shells()
        assert qdk_basis_2.get_ecp_name() == qdk_basis_1.get_ecp_name()
        assert qdk_basis_2.get_ecp_electrons() == qdk_basis_1.get_ecp_electrons()
        assert qdk_basis_2.get_num_ecp_shells() == qdk_basis_1.get_num_ecp_shells()

    def test_agh_def2svp_roundtrip(self):
        """Test AgH with def2-svp and ECP round-trip conversion."""
        # Setup AgH molecule with PySCF
        mol1 = pyscf.gto.Mole()
        mol1.atom = "Ag 0.0 0.0 0.0; H 0.0 0.0 1.617"
        mol1.basis = "def2-svp"
        mol1.ecp = "def2-svp"
        mol1.unit = "Angstrom"
        mol1.build()

        # Run SCF calculation
        scf1 = pyscf.scf.RHF(mol1)
        scf1.verbose = 0
        energy1 = scf1.kernel()
        assert hasattr(mol1, "_ecp")
        assert mol1._ecp

        # Create QDK Structure
        structure = Structure(symbols=["Ag", "H"], coordinates=mol1.atom_coords())

        # Convert PySCF Mole to QDK BasisSet
        qdk_basis = pyscf_mol_to_qdk_basis(mol1, structure, basis_name="def2-svp")

        # Convert QDK BasisSet back to PySCF Mole
        mol2 = basis_to_pyscf_mol(qdk_basis)
        assert hasattr(mol2, "_ecp")
        assert mol2._ecp

        # Run SCF calculation with converted basis
        scf2 = pyscf.scf.RHF(mol2)
        scf2.verbose = 0
        energy2 = scf2.kernel()

        # Verify round-trip conversion
        assert mol1.nao == mol2.nao
        assert mol1.nelectron == mol2.nelectron
        assert np.isclose(energy1, energy2, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)
