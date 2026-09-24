"""Tests for the SymmetryShifter (fermionic low-rank BLISS) algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import rebuild_shifted_hamiltonian
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Structure

from .reference_tolerances import (
    ci_energy_tolerance,
    float_comparison_relative_tolerance,
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


@pytest.fixture(scope="module")
def water_hamiltonian():
    """Build a water/STO-3G Hamiltonian, shared across tests in this module."""
    water = create_water_structure()
    scf_solver = algorithms.create("scf_solver")
    _, wfn_hf = scf_solver.run(water, 0, 1, "sto-3g")
    ham_constructor = algorithms.create("hamiltonian_constructor")
    return ham_constructor.run(wfn_hf.get_orbitals())


@pytest.fixture(scope="module")
def water_factorized(water_hamiltonian):
    """The shifter consumes an already double-factorized Hamiltonian."""
    factorizer = algorithms.create("hamiltonian_factorization", "double_factorization")
    return factorizer.run(water_hamiltonian)


class TestSymmetryShifterFactory:
    """Test factory registration and settings hygiene."""

    def test_factory(self):
        available = algorithms.available("symmetry_shifter")
        assert isinstance(available, list)
        assert "fermionic_low_rank" in available

        shifter = algorithms.create("symmetry_shifter")
        assert shifter is not None
        assert shifter.name() == "fermionic_low_rank"

        shifter_named = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        assert shifter_named.name() == "fermionic_low_rank"

        with pytest.raises(KeyError):
            algorithms.create("symmetry_shifter", "nonexistent")

    def test_has_no_settings(self):
        """Truncation and the decomposition belong to double_factorization, not here."""
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        assert shifter.settings().keys() == []

    def test_rejects_non_factorized_hamiltonian(self, water_hamiltonian):
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        with pytest.raises(ValueError):
            shifter.run(water_hamiltonian, 5, 5)


class TestSymmetryShifterCorrectness:
    """Physics correctness tests: energy invariance and 1-norm reduction."""

    def test_energy_invariant_under_shift(self, water_hamiltonian, water_factorized):
        """The correctness check: FCI energy cannot change after the fermionic low-rank BLISS shift."""
        mc = algorithms.create("multi_configuration_calculator", "macis_cas")
        e_before, _ = mc.run(water_hamiltonian, 5, 5)

        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shifted_ham = shifter.run(water_factorized, 5, 5)
        assert shifted_ham is not None

        mc_after = algorithms.create("multi_configuration_calculator", "macis_cas")
        e_after, _ = mc_after.run(shifted_ham, 5, 5)

        assert np.isclose(
            e_before,
            e_after,
            rtol=float_comparison_relative_tolerance,
            atol=ci_energy_tolerance,
        )

    def test_output_is_canonical_four_center(self, water_factorized):
        """Factorized in, dense out: the shift is applied to the dense integrals."""
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shifted = shifter.run(water_factorized, 5, 5)

        assert water_factorized.get_container_type() == "factorized"
        assert shifted.get_container_type() == "canonical_four_center"

    def test_compute_shift_then_rebuild_matches_run(self, water_factorized):
        """compute_shift() + rebuild_shifted_hamiltonian() must reproduce run()."""
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shifted_run = shifter.run(water_factorized, 5, 5)
        assert shifted_run is not None

        shifter2 = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shift = shifter2.compute_shift(water_factorized, 5, 5)
        shifted_manual = rebuild_shifted_hamiltonian(water_factorized, shift, 10)
        assert shifted_manual is not None

        h_run = shifted_run.get_one_body_integrals()[0]
        h_manual = shifted_manual.get_one_body_integrals()[0]
        assert np.allclose(h_run, h_manual, atol=1e-12)

        g_run = shifted_run.get_two_body_integrals()[0]
        g_manual = shifted_manual.get_two_body_integrals()[0]
        assert np.allclose(g_run, g_manual, atol=1e-12)

        assert np.isclose(shifted_run.get_core_energy(), shifted_manual.get_core_energy(), atol=1e-12)
