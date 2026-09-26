"""Tests for the SymmetryShifter (fermionic low-rank BLISS) algorithm."""

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

    def test_output_is_factorized(self, water_factorized):
        """Factorized in, factorized out: the shift is absorbed into the fragments."""
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shifted = shifter.run(water_factorized, 5, 5)

        assert water_factorized.get_container_type() == "factorized"
        assert shifted.get_container_type() == "factorized"

    def test_shift_reduces_reported_lambda(self, water_factorized):
        """Because the output is still factorized, it reports its own 1-norm."""
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shifted = shifter.run(water_factorized, 5, 5)

        assert shifted.get_container().get_lambda() < water_factorized.get_container().get_lambda()

    def test_shift_preserves_rotations(self, water_factorized):
        """Only the fragment eigenvalues move; the rotations and rank do not."""
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        before = water_factorized.get_container()
        after = shifter.run(water_factorized, 5, 5).get_container()

        assert after.get_num_ranks() == before.get_num_ranks()
        assert after.get_num_bases() == before.get_num_bases()
        assert after.get_num_copies() == before.get_num_copies()
        # rtol=0: the rotations must be bitwise-carried-through, so the
        # default relative term would swamp the absolute bound being asserted.
        assert np.allclose(after.get_u_matrices(), before.get_u_matrices(), rtol=0, atol=1e-15)
        assert not np.allclose(after.get_w_matrices(), before.get_w_matrices(), rtol=0, atol=1e-12)

    def test_last_shift_reports_the_applied_parameters(self, water_factorized):
        """run() applies a shift; last_shift() is how a caller reads it back."""
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        assert shifter.last_shift() is None

        shifted = shifter.run(water_factorized, 5, 5)
        shift = shifter.last_shift()
        assert shift is not None

        # The reported parameters must be the ones that were actually folded
        # in: h~ = h + (Ne-1)*xi - (mu1+mu2)*I.
        h_before = water_factorized.get_one_body_integrals()[0]
        h_after = shifted.get_one_body_integrals()[0]
        h_expected = h_before + 9.0 * shift.xi - (shift.mu1 + shift.mu2) * np.eye(
            h_before.shape[0]
        )
        assert np.allclose(h_after, h_expected, rtol=0, atol=1e-12)
