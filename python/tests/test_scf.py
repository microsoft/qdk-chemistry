"""Tests for SCF solver functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Structure

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

    def test_scf_solver_settings_edge_cases(self):
        """Test SCF solver with various invalid settings."""
        water = create_water_structure()

        # Test invalid basis set - should throw during solve
        scf_solver = algorithms.create("scf_solver")
        with pytest.raises(ValueError, match=r".*basis.*not.*supported"):
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
