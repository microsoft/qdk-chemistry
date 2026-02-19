"""Integration tests for model Hamiltonian Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import Hamiltonian, LatticeGraph
from qdk_chemistry.utils.model_hamiltonians import (
    create_hubbard_hamiltonian,
    create_huckel_hamiltonian,
    create_ppp_hamiltonian,
    mataga_nishimoto_potential,
    ohno_potential,
)

from .reference_tolerances import float_comparison_absolute_tolerance


class TestModelHamiltonians:
    """Integration test for model Hamiltonians via Python bindings."""

    def test_huckel_chain(self):
        n = 4
        lattice = LatticeGraph.chain(n)
        hamiltonian = create_huckel_hamiltonian(lattice, epsilon=-0.5, t=1.0)

        assert isinstance(hamiltonian, Hamiltonian)
        assert hamiltonian.has_one_body_integrals()
        assert not hamiltonian.has_two_body_integrals()

        h1_alpha, _ = hamiltonian.get_one_body_integrals()

        assert h1_alpha.shape == (n, n)

        h1_expected = np.zeros((n, n))
        np.fill_diagonal(h1_expected, -0.5)
        h1_expected[0, 1] = h1_expected[1, 0] = -1.0
        h1_expected[1, 2] = h1_expected[2, 1] = -1.0
        h1_expected[2, 3] = h1_expected[3, 2] = -1.0

        assert np.allclose(h1_alpha, h1_expected, atol=float_comparison_absolute_tolerance)

    def test_hubbard_chain(self):
        n = 4
        lattice = LatticeGraph.chain(n)
        hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-0.5, t=1.0, U=0.3)

        assert isinstance(hamiltonian, Hamiltonian)
        assert hamiltonian.has_one_body_integrals()
        assert hamiltonian.has_two_body_integrals()

        h1_alpha, _ = hamiltonian.get_one_body_integrals()
        h1_expected = np.zeros((n, n))
        np.fill_diagonal(h1_expected, -0.5)
        h1_expected[0, 1] = h1_expected[1, 0] = -1.0
        h1_expected[1, 2] = h1_expected[2, 1] = -1.0
        h1_expected[2, 3] = h1_expected[3, 2] = -1.0

        assert np.allclose(h1_alpha, h1_expected, atol=float_comparison_absolute_tolerance)
        for i in range(n):
            assert hamiltonian.get_two_body_element(i, i, i, i) == pytest.approx(
                0.3, abs=float_comparison_absolute_tolerance
            )
        assert hamiltonian.get_two_body_element(0, 0, 1, 1) == pytest.approx(
            0.0, abs=float_comparison_absolute_tolerance
        )
        assert hamiltonian.get_core_energy() == pytest.approx(0.0, abs=float_comparison_absolute_tolerance)

    def test_ppp_with_ohno_potential(self):
        n = 4
        epsilon_r = 0.9
        U_val = 0.4
        R_val = 1.2
        lattice = LatticeGraph.chain(n)
        V = ohno_potential(lattice, U=U_val, R=R_val, epsilon_r=epsilon_r)

        assert V.shape == (n, n)

        epsilon_vec = np.zeros(n)
        t_mat = np.ones((n, n))
        U_vec = np.full(n, U_val)
        z_vec = np.ones(n)
        hamiltonian = create_ppp_hamiltonian(lattice, epsilon=epsilon_vec, t=t_mat, U=U_vec, V=V, z=z_vec)

        assert isinstance(hamiltonian, Hamiltonian)
        assert hamiltonian.has_one_body_integrals()
        assert hamiltonian.has_two_body_integrals()

    def test_ppp_with_mataga_nishimoto_potential(self):
        n = 4
        epsilon_r = 0.9
        U_val = 0.4
        R_val = 1.2
        lattice = LatticeGraph.chain(n)
        V = mataga_nishimoto_potential(lattice, U=U_val, R=R_val, epsilon_r=epsilon_r)

        assert V.shape == (n, n)

        epsilon_vec = np.zeros(n)
        t_mat = np.ones((n, n))
        U_vec = np.full(n, U_val)
        z_vec = np.ones(n)
        hamiltonian = create_ppp_hamiltonian(lattice, epsilon=epsilon_vec, t=t_mat, U=U_vec, V=V, z=z_vec)

        assert isinstance(hamiltonian, Hamiltonian)
        assert hamiltonian.has_one_body_integrals()
        assert hamiltonian.has_two_body_integrals()
