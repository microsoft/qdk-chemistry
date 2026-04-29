"""Tests for SCF solver functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import BasisSet, Structure
from qdk_chemistry.utils import Logger

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    scf_energy_tolerance,
)


def create_water_structure():
    """Create a water molecule structure.

    Crawford geometry - same as used in C++ tests.
    """
    symbols = ["O", "H", "H"]
    coords = (
        np.array(
            [
                [0.000000000, -0.0757918436, 0.000000000000],
                [0.866811829, 0.6014357793, -0.000000000000],
                [-0.866811829, 0.6014357793, -0.000000000000],
            ]
        )
        * ANGSTROM_TO_BOHR
    )
    return Structure(symbols, coords)


def create_o2_structure():
    """Create an O2 molecule structure.

    Same geometry as used in C++ tests.
    """
    symbols = ["O", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.208 * ANGSTROM_TO_BOHR]])
    return Structure(symbols, coords)


def create_oh_structure():
    """Create an OH molecule structure.

    Same geometry as used in C++ tests.
    """
    symbols = ["O", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.9697 * ANGSTROM_TO_BOHR, 0.0, 0.0]])
    return Structure(symbols, coords)


def create_oxygen_structure():
    """Create an oxygen atom structure.

    Single oxygen atom at origin - same as used in C++ tests.
    """
    symbols = ["O"]
    coords = np.array([[0.00000000000, 0.00000000000, 0.00000000000]])
    return Structure(symbols, coords)


class TestScfSolver:
    """Test class for SCF solver functionality."""

    def test_scf_solver_factory(self):
        """Test SCF solver factory functionality."""
        available_solvers = algorithms.available("scf_solver")
        assert isinstance(available_solvers, list)
        assert len(available_solvers) >= 1

        # Test creating default solver
        scf_solver = algorithms.create("scf_solver")
        assert scf_solver is not None

        # Test creating solver by name
        scf_solver_default = algorithms.create("scf_solver", "qdk")
        assert scf_solver_default is not None

        # Test that nonexistent solver raises error
        with pytest.raises(KeyError):
            algorithms.create("scf_solver", "nonexistent_solver")

    def test_scf_solver_water_default_settings(self):
        """Test SCF solver on water molecule with default settings."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")

        # Solve with default settings
        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert orbitals is not None

        # Compare with expected energy from C++ test
        assert np.isclose(energy, -75.9229032345, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)

        # Check that orbitals have expected properties
        coeffs = orbitals.get_coefficients()
        assert coeffs is not None

        energies = orbitals.get_energies()
        assert energies is not None

    def test_scf_solver_water_def2_tzvp(self):
        """Test SCF solver on water molecule with def2-tzvp basis."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")

        # Change basis set to def2-tzvp
        energy, wavefunction = scf_solver.run(water, 0, 1, "def2-tzvp")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert orbitals is not None

        # Compare with expected energy from C++ test
        assert np.isclose(energy, -76.0205776518, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)

    def test_scf_solver_emits_iteration_logs_when_info_enabled(self, capfd):
        """Test that SCF iteration diagnostics respect the shared global logger level."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")
        scf_solver.settings().set("method", "hf")

        previous_level = Logger.get_global_level()
        try:
            Logger.set_global_level("info")
            energy, _ = scf_solver.run(water, 0, 1, "sto-3g")
            captured = capfd.readouterr()
        finally:
            Logger.set_global_level(previous_level)

        assert isinstance(energy, float)
        combined_output = captured.out + captured.err

        assert re.search(r"\bStep\s+\d+:", combined_output)
        assert "SCF converged: steps=" in combined_output

    def test_scf_solver_settings_edge_cases(self):
        """Test SCF solver with various invalid settings."""
        water = create_water_structure()

        # Test invalid basis set - should throw during solve
        scf_solver = algorithms.create("scf_solver")
        with pytest.raises(ValueError, match=r"Basis set file does not exist:"):
            scf_solver.run(water, 0, 1, "not_a_basis")

        # Should solve successfully with valid settings
        scf_solver = algorithms.create("scf_solver")
        energy, _ = scf_solver.run(water, 0, 1, "def2-tzvp")
        assert np.isclose(energy, -76.0205776518, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)

    def test_scf_solver_initial_guess_restart(self):
        """Test SCF solver with initial guess from converged orbitals."""
        # Water as restricted test
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")
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
        scf_solver2 = algorithms.create("scf_solver")
        scf_solver2.settings().set("method", "hf")
        scf_solver2.settings().set("max_iterations", 2)  # 2 is minimum as need to check energy difference

        # Second calculation with initial guess
        energy_second, _ = scf_solver2.run(water, 0, 1, orbitals_first)

        # Should get the same energy (within tight tolerance)
        assert np.isclose(
            energy_first, energy_second, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

    def test_scf_solver_oxygen_triplet_initial_guess(self):
        """Test SCF solver with initial guess for oxygen triplet state."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver")
        scf_solver.settings().set("method", "hf")

        # First calculation - let triplet converge normally
        energy_o2_first, wfn_o2_first = scf_solver.run(o2, 0, 3, "sto-3g")
        orbitals_o2_first = wfn_o2_first.get_orbitals()

        # Verify we get the expected energy for HF/STO-3G triplet
        assert np.isclose(
            energy_o2_first, -147.63396964335112, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # Now restart with the converged orbitals as initial guess
        # Create a new solver instance since settings are locked after run
        scf_solver2 = algorithms.create("scf_solver")
        scf_solver2.settings().set("method", "hf")
        scf_solver2.settings().set("max_iterations", 2)  # 2 is minimum as need to check energy difference

        # Second calculation with initial guess
        energy_o2_second, _ = scf_solver2.run(o2, 0, 3, orbitals_o2_first)

        # Should get the same energy (within tight tolerance)
        assert np.isclose(
            energy_o2_first, energy_o2_second, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

    def test_h2_scan_diis_numerical_stability(self):
        """Test that SCF handles numerical edge cases in H2 bond scans.

        This reproduces issues found with exact floating-point values from linspace
        where b_max can become zero in DIIS extrapolation.
        """
        # Test different bond lengths to trigger edge cases
        full_linspace = np.linspace(0.5, 5.0, 100)
        test_lengths = [
            full_linspace[3],  # b_max = 0 in DIIS
            np.round(full_linspace[3], 15),  # b_max approx 0 in DIIS
            full_linspace[0],  # b_max != 0 in DIIS
        ]

        expected_energies = [
            -0.7383108980408086,  # full_linspace[3]
            -0.7383108980408086,  # rounded full_linspace[3]
            -0.4033264392907958,  # full_linspace[0]
        ]

        scf_solver = algorithms.create("scf_solver")

        for i, length in enumerate(test_lengths):
            h2 = Structure(
                ["H", "H"],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]]),
            )
            energy, _ = scf_solver.run(h2, 0, 1, "sto-3g")
            assert np.isclose(
                energy,
                expected_energies[i],
                rtol=float_comparison_relative_tolerance,
                atol=scf_energy_tolerance,
            )

    def test_scf_solver_oh_rohf_diis(self):
        """Test SCF solver on OH system with ROHF/sto-3g."""
        oh_structure = create_oh_structure()
        scf_solver = algorithms.create("scf_solver")

        scf_solver.settings().set("method", "hf")
        scf_solver.settings().set("scf_type", "restricted")
        scf_solver.settings().set("enable_gdm", False)

        energy, wavefunction = scf_solver.run(oh_structure, 0, 2, "sto-3g")
        orbitals = wavefunction.get_orbitals()

        assert abs(energy - (-74.361530753176)) < scf_energy_tolerance
        assert orbitals.is_restricted()

    def test_scf_solver_oxygen_atom_gdm(self):
        """Test SCF solver on oxygen atom with PBE/cc-pvdz."""
        oxygen = create_oxygen_structure()
        scf_solver = algorithms.create("scf_solver")

        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("enable_gdm", True)

        energy, wavefunction = scf_solver.run(oxygen, 0, 1, "cc-pvdz")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert orbitals is not None

        # Compare with expected energy from C++ test
        assert abs(energy - (-74.873106298)) < scf_energy_tolerance

    def test_scf_solver_oxygen_atom_history_size_limit_gdm(self):
        """Test SCF solver on oxygen atom with GDM and history size limit 20."""
        oxygen = create_oxygen_structure()
        scf_solver = algorithms.create("scf_solver")

        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("enable_gdm", True)
        scf_solver.settings().set("gdm_bfgs_history_size_limit", 20)

        energy, wavefunction = scf_solver.run(oxygen, 0, 1, "cc-pvdz")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert orbitals is not None

        # Compare with expected energy from C++ test
        assert abs(energy - (-74.873106298)) < scf_energy_tolerance

    def test_scf_solver_oxygen_atom_one_diis_step_gdm(self):
        """Test SCF solver on oxygen atom with PBE/cc-pvdz, with only 1 diis step."""
        oxygen = create_oxygen_structure()
        scf_solver = algorithms.create("scf_solver")

        # Set method and basis set to match C++ test
        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("enable_gdm", True)
        scf_solver.settings().set("gdm_max_diis_iteration", 1)

        energy, wavefunction = scf_solver.run(oxygen, 0, 1, "cc-pvdz")
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert orbitals is not None

        # Compare with expected energy from C++ test
        assert abs(energy - (-74.873106298)) < scf_energy_tolerance

    def test_scf_solver_water_triplet_gdm(self):
        """Test SCF solver on water molecule triplet with GDM enabled."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver")

        # Set method and basis set to match C++ test
        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("enable_gdm", True)

        energy, wavefunction = scf_solver.run(water, 0, 3, "def2-svp")  # triplet state
        orbitals = wavefunction.get_orbitals()

        # Check that we get reasonable results
        assert isinstance(energy, float)
        assert orbitals is not None

        nuclear_repulsion = water.calculate_nuclear_repulsion_energy()
        expected_total_energy = -84.036674819 + nuclear_repulsion
        assert abs(energy - expected_total_energy) < scf_energy_tolerance

        # Check that orbitals are unrestricted (not restricted)
        assert not orbitals.is_restricted()

    def test_scf_solver_oxygen_atom_charged_doublet_gdm(self):
        """Test SCF solver on charged oxygen atom doublet with GDM enabled."""
        oxygen = create_oxygen_structure()
        scf_solver = algorithms.create("scf_solver")

        # Set method and basis set to match C++ test
        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("enable_gdm", True)
        scf_solver.settings().set("max_iterations", 100)

        energy, wavefunction = scf_solver.run(oxygen, 1, 2, "cc-pvdz")  # +1 charge, doublet state
        orbitals = wavefunction.get_orbitals()

        assert isinstance(energy, float)
        assert orbitals is not None

        # Compare with expected energy from C++ test
        assert abs(energy - (-74.416994299)) < scf_energy_tolerance

        # Check that orbitals are unrestricted (not restricted) for the doublet state
        assert not orbitals.is_restricted()

    def test_scf_solver_oxygen_atom_invalid_energy_thresh_diis_switch_gdm(self):
        """Test SCF solver on oxygen atom with GDM - invalid energy_thresh_diis_switch."""
        oxygen = create_oxygen_structure()
        scf_solver = algorithms.create("scf_solver")

        # Set method and basis set to match C++ test
        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("enable_gdm", True)
        scf_solver.settings().set("energy_thresh_diis_switch", -2e-4)

        # Test that negative energy_thresh_diis_switch throws a ValueError (std::invalid_argument in C++)
        with pytest.raises(ValueError, match="energy_thresh_diis_switch must be greater than"):
            scf_solver.run(oxygen, 0, 1, "cc-pvdz")  # singlet state

    def test_scf_solver_oxygen_atom_invalid_bfgs_history_size_limit_gdm(self):
        """Test SCF solver on oxygen atom with GDM - invalid BFGS history size limit."""
        oxygen = create_oxygen_structure()
        scf_solver = algorithms.create("scf_solver")

        # Set method and basis set to match C++ test
        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("enable_gdm", True)
        scf_solver.settings().set("gdm_bfgs_history_size_limit", 0)

        # Test that invalid history size limit throws a ValueError (std::invalid_argument in C++)
        with pytest.raises(ValueError, match="GDM history size limit must be at least"):
            scf_solver.run(oxygen, 0, 1, "cc-pvdz")  # singlet state


_REF_BOHR_TO_ANG = 0.52917721092


def _create_h2o_dfj_structure():
    """Create H2O structure matching the DFJ reference calculation geometry."""
    coords = np.array(
        [
            [0.00, 0.49 / _REF_BOHR_TO_ANG, -0.79 / _REF_BOHR_TO_ANG],
            [0.00, 0.49 / _REF_BOHR_TO_ANG, 0.79 / _REF_BOHR_TO_ANG],
            [0.00, -0.12 / _REF_BOHR_TO_ANG, 0.00],
        ]
    )
    return Structure(["H", "H", "O"], coords)


def _create_o2_dfj_structure():
    """Create O2 structure matching the DFJ reference calculation geometry (bond distance 1.21 Å)."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.21 / _REF_BOHR_TO_ANG],
        ]
    )
    return Structure(["O", "O"], coords)


def _create_bf_structure():
    """Create BF (boron fluoride) structure matching the DFJ reference geometry.

    Coordinates are already in Bohr.
    """
    coords = np.array(
        [
            [0.0, 0.0, 0.85543933],
            [0.0, 0.0, -1.53978853],
        ]
    )
    return Structure(["F", "B"], coords)


class TestScfSolverDfj:
    """DFJ (Density-Fitted Coulomb) SCF tests."""

    def test_water_rhf_dfj(self):
        """Test RHF-DFJ on water with def2-svp / def2-universal-jfit."""
        water = _create_h2o_dfj_structure()
        scf_solver = algorithms.create("scf_solver")
        scf_solver.settings().set("method", "hf")
        scf_solver.settings().set("eri_method", "incore")

        basis = BasisSet.from_basis_name("def2-svp", "def2-universal-jfit", water)
        energy, wfn = scf_solver.run(water, 0, 1, basis)

        assert abs(energy - (-75.955848898587732)) < scf_energy_tolerance

    def test_water_rks_dfj_pbe_m06_2x(self):
        """Test RKS-DFJ/PBE then use it as guess for RKS-DFJ/M06-2X on water with def2-svp / def2-universal-jfit."""
        water = _create_h2o_dfj_structure()
        scf_solver = algorithms.create("scf_solver")
        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("eri_method", "incore")

        basis = BasisSet.from_basis_name("def2-svp", "def2-universal-jfit", water)
        energy, wfn = scf_solver.run(water, 0, 1, basis)

        assert abs(energy - (-76.271464794036)) < scf_energy_tolerance

        # use the PBE orbitals as the initial guess for M06-2X
        m06_solver = algorithms.create("scf_solver")
        m06_solver.settings().set("method", "m06-2x")
        m06_solver.settings().set("eri_method", "incore")
        energy, m06_wfn = m06_solver.run(water, 0, 1, wfn.get_orbitals())

        assert abs(energy - (-76.320941901587)) < scf_energy_tolerance

    def test_o2_triplet_uhf_dfj(self):
        """Test UHF-DFJ on O2 triplet with def2-svp / def2-universal-jfit."""
        o2 = _create_o2_dfj_structure()
        scf_solver = algorithms.create("scf_solver")
        scf_solver.settings().set("method", "hf")
        scf_solver.settings().set("eri_method", "incore")

        basis = BasisSet.from_basis_name("def2-svp", "def2-universal-jfit", o2)
        energy, wfn = scf_solver.run(o2, 0, 3, basis)

        assert abs(energy - (-149.489993170463)) < scf_energy_tolerance

    def test_bf_uks_dfj_pbe(self):
        """Test UKS-DFJ/PBE on BF with sto-3g / def2-universal-jfit."""
        bf = _create_bf_structure()
        scf_solver = algorithms.create("scf_solver")
        scf_solver.settings().set("method", "pbe")
        scf_solver.settings().set("scf_type", "unrestricted")
        scf_solver.settings().set("eri_method", "incore")

        basis = BasisSet.from_basis_name("sto-3g", "def2-universal-jfit", bf)
        energy, wfn = scf_solver.run(bf, 0, 1, basis)

        assert abs(energy - (-122.732943463018)) < scf_energy_tolerance

    def test_dfj_without_aux_basis_raises(self):
        """Test that requesting DFJ without an auxiliary basis raises ValueError."""
        water = _create_h2o_dfj_structure()
        scf_solver = algorithms.create("scf_solver")
        scf_solver.settings().set("method", "hf")
        scf_solver.settings().set("eri_method", "incore")
        scf_solver.settings().set("integral_type", "dfj")

        # Basis without auxiliary shells
        basis = BasisSet.from_basis_name("def2-svp", water)
        with pytest.raises(ValueError, match="DFJ requested but no auxiliary"):
            scf_solver.run(water, 0, 1, basis)
        with pytest.raises(ValueError, match="DFJ requested but no auxiliary"):
            scf_solver.run(water, 0, 1, "def2-svp")