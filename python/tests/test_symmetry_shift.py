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
from qdk_chemistry.utils import (
    DoubleFactorizationMethod,
    double_factorize,
    hamiltonian_one_norm,
)

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

    def test_default_truncation_threshold_is_zero(self):
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        assert shifter.settings().get("df_truncation_threshold") == 0.0


class TestSymmetryShifterCorrectness:
    """Physics correctness tests: energy invariance and 1-norm reduction."""

    def test_energy_invariant_under_shift(self, water_hamiltonian):
        """The correctness check: FCI energy cannot change after the fermionic low-rank BLISS shift."""
        mc = algorithms.create("multi_configuration_calculator", "macis_cas")
        e_before, _ = mc.run(water_hamiltonian, 5, 5)

        for threshold in (0.0, 1e-6):
            shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
            shifter.settings().set("df_truncation_threshold", threshold)
            shifted_ham = shifter.run(water_hamiltonian, 5, 5)
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

        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shifted_ham = shifter.run(water_hamiltonian, 5, 5)

        norm_after = hamiltonian_one_norm(shifted_ham, 0.0)

        assert norm_after.total <= norm_before.total + 1e-10

    def test_compute_shift_then_rebuild_matches_run(self, water_hamiltonian):
        """compute_shift() + rebuild_shifted_hamiltonian() must reproduce run()."""
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shifted_run = shifter.run(water_hamiltonian, 5, 5)
        assert shifted_run is not None

        shifter2 = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        shift = shifter2.compute_shift(water_hamiltonian, 5, 5)
        shifted_manual = rebuild_shifted_hamiltonian(water_hamiltonian, shift, 10)
        assert shifted_manual is not None

        h_run = shifted_run.get_one_body_integrals()[0]
        h_manual = shifted_manual.get_one_body_integrals()[0]
        assert np.allclose(h_run, h_manual, atol=1e-12)

        g_run = shifted_run.get_two_body_integrals()[0]
        g_manual = shifted_manual.get_two_body_integrals()[0]
        assert np.allclose(g_run, g_manual, atol=1e-12)

        assert np.isclose(shifted_run.get_core_energy(), shifted_manual.get_core_energy(), atol=1e-12)


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

    def test_double_factorize_defaults_to_cholesky(self, water_hamiltonian):
        g_aaaa, _, _ = water_hamiltonian.get_two_body_integrals()
        norb = water_hamiltonian.get_orbitals().get_num_molecular_orbitals()

        fragments_default = double_factorize(g_aaaa, norb, 0.0)
        fragments_cholesky = double_factorize(g_aaaa, norb, 0.0, DoubleFactorizationMethod.CHOLESKY)
        assert len(fragments_default) == len(fragments_cholesky)
        assert np.allclose(
            [f.lambda_df for f in fragments_default],
            [f.lambda_df for f in fragments_cholesky],
        )

    def test_double_factorize_both_methods_reconstruct_tensor(self, water_hamiltonian):
        """Both methods factor the same operator; only the 1-norm is gauge dependent."""
        g_aaaa, _, _ = water_hamiltonian.get_two_body_integrals()
        norb = water_hamiltonian.get_orbitals().get_num_molecular_orbitals()

        def reconstruct(fragments):
            g = np.zeros((norb, norb, norb, norb))
            for fragment in fragments:
                m = fragment.U @ np.diag(fragment.eps) @ fragment.U.T
                g += fragment.sign * np.einsum("ij,kl->ijkl", m, m)
            return g.reshape(-1)

        for method in (
            DoubleFactorizationMethod.CHOLESKY,
            DoubleFactorizationMethod.EIGEN,
        ):
            fragments = double_factorize(g_aaaa, norb, 0.0, method)
            assert len(fragments) > 0
            assert np.allclose(reconstruct(fragments), g_aaaa, atol=1e-10), method

        # Cholesky never needs a negative fragment for physical integrals, and
        # its rank is bounded by the symmetric-pair dimension.
        cholesky = double_factorize(g_aaaa, norb, 0.0, DoubleFactorizationMethod.CHOLESKY)
        assert all(f.sign == 1 for f in cholesky)
        assert len(cholesky) <= norb * (norb + 1) // 2

    def test_hamiltonian_one_norm_accepts_method(self, water_hamiltonian):
        norm_cholesky = hamiltonian_one_norm(water_hamiltonian, 0.0, DoubleFactorizationMethod.CHOLESKY)
        norm_eigen = hamiltonian_one_norm(water_hamiltonian, 0.0, DoubleFactorizationMethod.EIGEN)
        norm_default = hamiltonian_one_norm(water_hamiltonian, 0.0)

        # The one-body term is independent of the two-body factorization.
        assert np.isclose(norm_cholesky.one_body, norm_eigen.one_body)
        assert np.isclose(norm_default.two_body, norm_cholesky.two_body)
        assert norm_eigen.two_body > 0.0


class TestFermionicLowRankDoubleFactorizationMethod:
    """The BLISS shift is derived from the fragments, so the method is a setting."""

    def test_df_method_setting_accepts_both_values(self, water_hamiltonian):
        for df_method in ("cholesky", "eigen"):
            shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
            shifter.settings().set("df_method", df_method)
            shifted = shifter.run(water_hamiltonian, 5, 5)
            assert shifted is not None

            norm_before = hamiltonian_one_norm(water_hamiltonian, 0.0)
            norm_after = hamiltonian_one_norm(shifted, 0.0)
            assert norm_after.total <= norm_before.total + 1e-10

    def test_df_method_defaults_to_cholesky(self):
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        assert shifter.settings().get("df_method") == "cholesky"

    def test_df_method_rejects_unknown_value(self):
        shifter = algorithms.create("symmetry_shifter", "fermionic_low_rank")
        with pytest.raises(ValueError, match="out of allowed options"):
            shifter.settings().set("df_method", "not_a_method")
