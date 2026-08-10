"""Tests for the HamiltonianRegularizer (FLR-BLISS) algorithm."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Structure
from qdk_chemistry.utils import double_factorize, hamiltonian_one_norm

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


class TestHamiltonianRegularizerFactory:
    """Test factory registration and settings hygiene."""

    def test_factory(self):
        available = algorithms.available("hamiltonian_regularizer")
        assert isinstance(available, list)
        assert "fermionic_low_rank" in available

        regularizer = algorithms.create("hamiltonian_regularizer")
        assert regularizer is not None
        assert regularizer.name() == "fermionic_low_rank"

        regularizer_named = algorithms.create("hamiltonian_regularizer", "fermionic_low_rank")
        assert regularizer_named.name() == "fermionic_low_rank"

        with pytest.raises(KeyError):
            algorithms.create("hamiltonian_regularizer", "nonexistent")

    def test_default_truncation_threshold_is_zero(self):
        regularizer = algorithms.create("hamiltonian_regularizer", "fermionic_low_rank")
        assert regularizer.settings().get("df_truncation_threshold") == 0.0


class TestHamiltonianRegularizerCorrectness:
    """Physics correctness tests: energy invariance and 1-norm reduction."""

    def test_energy_invariant_under_shift(self, water_hamiltonian):
        """The correctness check: FCI energy cannot change after FLR-BLISS shift."""

        mc = algorithms.create("multi_configuration_calculator", "macis_cas")
        e_before, _ = mc.run(water_hamiltonian, 5, 5)

        for threshold in (0.0, 1e-6):
            regularizer = algorithms.create("hamiltonian_regularizer", "fermionic_low_rank")
            regularizer.settings().set("df_truncation_threshold", threshold)
            shifted_ham = regularizer.run(water_hamiltonian, 5, 5)
            assert shifted_ham is not None

            mc_after = algorithms.create("multi_configuration_calculator", "macis_cas")
            e_after, _ = mc_after.run(shifted_ham, 5, 5)

            assert np.isclose(
                e_before,
                e_after,
                rtol=float_comparison_relative_tolerance,
                atol=ci_energy_tolerance,
            ), f"Energy not invariant at df_truncation_threshold={threshold}"

    def test_reduces_one_norm(self, water_hamiltonian):
        norm_before = hamiltonian_one_norm(water_hamiltonian, 0.0)

        regularizer = algorithms.create("hamiltonian_regularizer", "fermionic_low_rank")
        shifted_ham = regularizer.run(water_hamiltonian, 5, 5)

        norm_after = hamiltonian_one_norm(shifted_ham, 0.0)

        assert norm_after.total <= norm_before.total + 1e-10


class TestDoubleFactorizationUtils:
    """Tests double_factorize and hamiltonian_one_norm."""

    def test_double_factorize_default_no_truncation(self, water_hamiltonian):
        g_aaaa, _, _ = water_hamiltonian.get_two_body_integrals()
        norb = water_hamiltonian.get_orbitals().get_num_molecular_orbitals()

        fragments_default = double_factorize(g_aaaa, norb)
        fragments_explicit_zero = double_factorize(g_aaaa, norb, 0.0)
        assert len(fragments_default) == len(fragments_explicit_zero)

        fragments_loose = double_factorize(g_aaaa, norb, 1e-2)
        assert len(fragments_loose) <= len(fragments_explicit_zero)

    def test_hamiltonian_one_norm_standalone(self, water_hamiltonian):
        norm = hamiltonian_one_norm(water_hamiltonian)
        assert norm.one_body > 0.0
        assert norm.two_body > 0.0
        assert np.isclose(norm.total, norm.one_body + norm.two_body)
