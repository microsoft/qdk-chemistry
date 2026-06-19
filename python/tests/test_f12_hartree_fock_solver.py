"""Tests for the CT-F12 Hartree-Fock solver.

These cover the ``f12_hartree_fock_solver`` algorithm type: registration,
settings schema, and that ``run()`` dispatches to the backend and returns a
relaxed F12-HF wavefunction. Numerical F12-MP2 validation lives in the C++ test
``test_ctf12_effective_hamiltonian``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import (
    F12HartreeFockSolver,
    QdkCtF12HartreeFockSolver,
    create,
    registry,
)
from qdk_chemistry.data import Structure, Wavefunction


@pytest.fixture(scope="module")
def neon_hf():
    """Closed-shell Hartree-Fock reference for the neon atom."""
    mol = Structure(np.zeros((1, 3)), ["Ne"])
    _, hf = create("scf_solver").run(mol, 0, 1, "aug-cc-pvdz")
    return hf


class TestF12HartreeFockRegistration:
    """Registration of the f12_hartree_fock_solver type."""

    def test_type_default_is_ct_f12(self):
        defaults = registry.show_default()
        assert defaults["f12_hartree_fock_solver"] == "qdk_ct_f12"

    def test_create_default(self):
        solver = create("f12_hartree_fock_solver")
        assert solver.name() == "qdk_ct_f12"
        assert solver.type_name() == "f12_hartree_fock_solver"
        assert isinstance(solver, F12HartreeFockSolver)

    def test_create_named(self):
        solver = create("f12_hartree_fock_solver", "qdk_ct_f12")
        assert isinstance(solver, QdkCtF12HartreeFockSolver)


class TestF12HartreeFockSettings:
    """The CT-F12 Hartree-Fock settings schema and its constraints."""

    def test_default_settings(self):
        s = create("f12_hartree_fock_solver").settings()
        assert s.get("gamma") == 1.0
        assert s.get("cabs_basis") == ""
        assert s.get("frozen_core") == 0

    def test_create_with_kwargs(self):
        solver = create(
            "f12_hartree_fock_solver",
            "qdk_ct_f12",
            gamma=1.5,
            cabs_basis="aug-cc-pvtz-optri",
        )
        assert solver.settings().get("gamma") == 1.5
        assert solver.settings().get("cabs_basis") == "aug-cc-pvtz-optri"

    def test_gamma_bound_enforced(self):
        s = create("f12_hartree_fock_solver").settings()
        with pytest.raises(ValueError, match="out of allowed range"):
            s.set("gamma", -1.0)


class TestF12HartreeFockRun:
    """run() reaches the CT-F12 backend and emits a relaxed wavefunction."""

    def test_run_returns_relaxed_reference(self, neon_hf):
        solver = create(
            "f12_hartree_fock_solver",
            "qdk_ct_f12",
            gamma=1.5,
            frozen_core=1,
            cabs_basis="aug-cc-pvdz-optri",
        )
        relaxed = solver.run(neon_hf)
        assert isinstance(relaxed, Wavefunction)

        orbitals = relaxed.get_orbitals()
        assert orbitals.has_energies()
        # The frozen core (1 orbital) is inactive; the rest is active.
        n_active = len(orbitals.get_active_space_indices()[0])
        assert n_active == orbitals.get_num_molecular_orbitals() - 1
        # Four valence pairs remain in the active space (frozen-core neon).
        assert relaxed.get_active_num_electrons() == (4, 4)
