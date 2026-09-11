"""Verify that Hamiltonian constructors include ECP one-electron terms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    AuxiliaryBasis,
    AuxiliaryBasisCollection,
    AuxiliaryBasisRole,
    OrbitalType,
    Shell,
    Structure,
)

from .reference_tolerances import scf_energy_tolerance

# Mo ROHF/def2-svp reference: trace of h1e (alpha = beta for ROHF).
# Trace is unitarily invariant, so it does not depend on the MO rotation
# within degenerate subspaces (d-shell degeneracy causes different rotations
# across thread counts / platforms).
# Without ECP (bare Z=42), the trace is ~ -1800.  With ECP (Z_eff=14), ~ -134.
_MO_ROHF_H1E_TRACE = -133.9900221637


@pytest.fixture(scope="module")
def mo_rohf_orbitals():
    """ROHF orbitals for Mo atom with def2-svp (shared across both parameterized tests)."""
    structure = Structure(np.array([[0.0, 0.0, 0.0]]), ["Mo"])
    scf = create("scf_solver", "qdk")
    scf.settings()["method"] = "hf"
    scf.settings()["scf_type"] = "restricted"
    scf.settings()["enable_gdm"] = False
    _, wfn = scf.run(structure, 0, 7, "def2-svp")
    return wfn.get_orbitals()


@pytest.mark.parametrize(
    "constructor_name",
    ["qdk", "qdk_cholesky", "qdk_density_fitted_hamiltonian"],
)
def test_ecp_included_in_hamiltonian_h1e(mo_rohf_orbitals, constructor_name):
    """h1e from HamiltonianConstructor must include ECP terms for Mo/def2-svp."""
    auxiliary_bases = None
    if constructor_name == "qdk_density_fitted_hamiltonian":
        structure = mo_rohf_orbitals.get_basis_set().get_structure()
        rifit = AuxiliaryBasis(
            "ecp-test-rifit",
            [Shell(0, OrbitalType.S, [1.0], [1.0])],
            structure,
        )
        auxiliary_bases = AuxiliaryBasisCollection({AuxiliaryBasisRole.RIFIT: rifit})
    ham = create("hamiltonian_constructor", constructor_name).run(mo_rohf_orbitals, auxiliary_bases)
    h1e_a, h1e_b = ham.get_one_body_integrals()

    np.testing.assert_allclose(np.trace(h1e_a), _MO_ROHF_H1E_TRACE, atol=scf_energy_tolerance)
    np.testing.assert_allclose(np.trace(h1e_b), _MO_ROHF_H1E_TRACE, atol=scf_energy_tolerance)
