"""Tests for the CT-F12 effective Hamiltonian constructor.

These cover the ``effective_hamiltonian_constructor`` algorithm type:
registration, settings schema, and that ``run()`` dispatches to the backend and
returns a dressed Hamiltonian. Numerical F12-MP2 validation lives in the C++
test ``test_ctf12_effective_hamiltonian``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import (
    EffectiveHamiltonianConstructor,
    QdkCtF12HamiltonianConstructor,
    create,
    registry,
)
from qdk_chemistry.data import Hamiltonian, Structure


@pytest.fixture(scope="module")
def neon_hf():
    """Closed-shell Hartree-Fock reference for the neon atom."""
    mol = Structure(np.zeros((1, 3)), ["Ne"])
    _, hf = create("scf_solver").run(mol, 0, 1, "aug-cc-pvdz")
    return hf


class TestCtF12Registration:
    """Registration of the effective_hamiltonian_constructor type."""

    def test_type_default_is_ct_f12(self):
        defaults = registry.show_default()
        assert defaults["effective_hamiltonian_constructor"] == "qdk_ct_f12"

    def test_create_default(self):
        ctf12 = create("effective_hamiltonian_constructor")
        assert ctf12.name() == "qdk_ct_f12"
        assert ctf12.type_name() == "effective_hamiltonian_constructor"
        assert isinstance(ctf12, EffectiveHamiltonianConstructor)

    def test_create_named(self):
        ctf12 = create("effective_hamiltonian_constructor", "qdk_ct_f12")
        assert isinstance(ctf12, QdkCtF12HamiltonianConstructor)


class TestCtF12Settings:
    """The CT-F12 settings schema and its constraints."""

    def test_default_settings(self):
        s = create("effective_hamiltonian_constructor").settings()
        assert s.get("gamma") == 1.0
        assert s.get("cabs_basis") == ""
        assert s.get("frozen_core") == 0
        assert s.get("eri_method") == "direct"
        assert s.get("slater_factor") == "stg"
        assert s.get("orbital_basis") == "relaxed"
        assert s.get("symmetrize_two_body") is False

    def test_create_with_kwargs(self):
        ctf12 = create(
            "effective_hamiltonian_constructor",
            "qdk_ct_f12",
            gamma=1.5,
            cabs_basis="aug-cc-pvtz-optri",
        )
        assert ctf12.settings().get("gamma") == 1.5
        assert ctf12.settings().get("cabs_basis") == "aug-cc-pvtz-optri"

    def test_gamma_bound_enforced(self):
        s = create("effective_hamiltonian_constructor").settings()
        with pytest.raises(ValueError, match="out of allowed range"):
            s.set("gamma", -1.0)

    def test_slater_factor_choices_enforced(self):
        s = create("effective_hamiltonian_constructor").settings()
        with pytest.raises(ValueError, match="out of allowed options"):
            s.set("slater_factor", "bogus")

    def test_orbital_basis_choices_enforced(self):
        s = create("effective_hamiltonian_constructor").settings()
        with pytest.raises(ValueError, match="out of allowed options"):
            s.set("orbital_basis", "bogus")


class TestCtF12Run:
    """run() reaches the CT-F12 backend and emits a dressed Hamiltonian."""

    def test_run_returns_dressed_hamiltonian(self, neon_hf):
        ctf12 = create(
            "effective_hamiltonian_constructor",
            "qdk_ct_f12",
            gamma=1.5,
            frozen_core=1,
            cabs_basis="aug-cc-pvdz-optri",
        )
        dressed = ctf12.run(neon_hf)
        assert isinstance(dressed, Hamiltonian)

        orbitals = dressed.get_orbitals()
        assert orbitals.has_energies()
        # The frozen core (1 orbital) is inactive; the rest is active.
        n_active = len(orbitals.get_active_space_indices()[0])
        assert n_active == orbitals.get_num_molecular_orbitals() - 1
