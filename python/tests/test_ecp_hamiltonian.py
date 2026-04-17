"""Verify that ECP contributions are included in Hamiltonian one-electron integrals.

Bug: HamiltonianConstructor.run() computes h_core = T + V_nuc but omits V_ecp.
This causes incorrect one-electron integrals for any atom using an ECP basis
(e.g., Mo with def2-svp).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

from .reference_tolerances import scf_energy_tolerance

# Mo ROHF/def2-svp reference: sum of absolute values of h1e (alpha = beta for ROHF).
# Without ECP (bare Z=42) this blows up to ~10,000+.
_MO_ROHF_H1E_ABS_SUM = 233.2282872909


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


@pytest.mark.parametrize("constructor_name", ["qdk", "qdk_cholesky"])
def test_ecp_included_in_hamiltonian_h1e(mo_rohf_orbitals, constructor_name):
    """h1e from HamiltonianConstructor must include ECP terms for Mo/def2-svp."""
    ham = create("hamiltonian_constructor", constructor_name).run(mo_rohf_orbitals)
    h1e_a, h1e_b = ham.get_one_body_integrals()

    np.testing.assert_allclose(np.sum(np.abs(h1e_a)), _MO_ROHF_H1E_ABS_SUM, atol=scf_energy_tolerance)
    np.testing.assert_allclose(np.sum(np.abs(h1e_b)), _MO_ROHF_H1E_ABS_SUM, atol=scf_energy_tolerance)
