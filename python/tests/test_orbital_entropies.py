"""Integration tests for orbital entropy Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry import algorithms
from qdk_chemistry.data import Structure
from qdk_chemistry.utils.orbital_entropies import (
    build_mutual_information,
    build_single_orbital_entropies,
    build_two_orbital_entropies,
    max_entropy,
    min_entropy,
    renyi_entropy,
    von_neumann_entropy,
)

from .reference_tolerances import ci_energy_tolerance


def create_water_structure():
    """Create a water molecule structure (Crawford geometry)."""
    symbols = ["O", "H", "H"]
    coords = np.array(
        [
            [0.000000000000, -0.143225816552, 0.000000000000],
            [1.638036840407, 1.136548822547, 0.000000000000],
            [-1.638036840407, 1.136548822547, 0.000000000000],
        ]
    )
    return Structure(symbols, coords)


class TestOrbitalEntropies:
    """Integration tests for orbital entropy utilities via Python bindings."""

    def test_entropy_factories(self):
        """All entropy factory functions return callables."""
        vn = von_neumann_entropy()
        r2 = renyi_entropy(2.0)
        mi = min_entropy()
        mx = max_entropy()

        eigs = [0.25, 0.25, 0.25, 0.25]
        assert isinstance(vn(eigs), float)
        assert isinstance(r2(eigs), float)
        assert isinstance(mi(eigs), float)
        assert isinstance(mx(eigs), float)

    def test_single_orbital_entropies_from_matrix(self):
        """build_single_orbital_entropies works with a raw eigenvalue matrix."""
        one_ordm = np.array(
            [
                [0.9, 0.05, 0.04, 0.01],
                [0.5, 0.3, 0.15, 0.05],
            ]
        )
        s1 = build_single_orbital_entropies(one_ordm, von_neumann_entropy())

        assert s1.shape == (2,)
        assert np.all(s1 >= 0.0)

    def test_mutual_information_from_vectors(self):
        """build_mutual_information works with raw s1 and s2 arrays."""
        s1 = np.array([0.5, 0.3, 0.8])
        s2 = np.array(
            [
                [0.0, 0.7, 1.0],
                [0.7, 0.0, 0.9],
                [1.0, 0.9, 0.0],
            ]
        )
        mi = build_mutual_information(s1, s2)

        assert mi.shape == (3, 3)
        np.testing.assert_allclose(mi, mi.T)

    def test_wavefunction_api(self):
        """All wavefunction-based functions are callable with MACIS CAS."""
        h2o = create_water_structure()

        scf_solver = algorithms.create("scf_solver")
        mc_calculator = algorithms.create("multi_configuration_calculator", "macis_cas")
        ham_constructor = algorithms.create("hamiltonian_constructor")

        _, wfn_hf = scf_solver.run(h2o, 0, 1, "sto-3g")

        settings = mc_calculator.settings()
        settings.set("ci_residual_tolerance", ci_energy_tolerance)
        settings.set("calculate_one_orbital_rdm", True)
        settings.set("calculate_two_orbital_rdm", True)

        ham = ham_constructor.run(wfn_hf.get_orbitals())
        _, wfn = mc_calculator.run(ham, 5, 5)

        s1 = build_single_orbital_entropies(wfn)
        s2 = build_two_orbital_entropies(wfn)
        mi = build_mutual_information(wfn)

        norb = wfn.get_one_orbital_rdm().shape[0]
        assert s1.shape == (norb,)
        assert s2.shape == (norb, norb)
        assert mi.shape == (norb, norb)

    def test_custom_entropy_from_wavefunction(self):
        """Wavefunction API accepts custom entropy measures."""
        h2o = create_water_structure()

        scf_solver = algorithms.create("scf_solver")
        mc_calculator = algorithms.create("multi_configuration_calculator", "macis_cas")
        ham_constructor = algorithms.create("hamiltonian_constructor")

        _, wfn_hf = scf_solver.run(h2o, 0, 1, "sto-3g")

        settings = mc_calculator.settings()
        settings.set("ci_residual_tolerance", ci_energy_tolerance)
        settings.set("calculate_one_orbital_rdm", True)
        settings.set("calculate_two_orbital_rdm", True)

        ham = ham_constructor.run(wfn_hf.get_orbitals())
        _, wfn = mc_calculator.run(ham, 5, 5)

        s1_renyi = build_single_orbital_entropies(wfn, renyi_entropy(2.0))
        s1_min = build_single_orbital_entropies(wfn, min_entropy())
        s1_max = build_single_orbital_entropies(wfn, max_entropy())

        norb = wfn.get_one_orbital_rdm().shape[0]
        assert s1_renyi.shape == (norb,)
        assert s1_min.shape == (norb,)
        assert s1_max.shape == (norb,)
