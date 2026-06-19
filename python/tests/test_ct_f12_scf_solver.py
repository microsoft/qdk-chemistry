"""Tests for the CT-F12 SCF solver.

These cover the CT-F12 implementation registered under the ``scf_solver``
algorithm type: registration, settings schema (including the configurable
canonical-SCF sub-step), and that ``run()`` produces a relaxed F12-HF
wavefunction. Numerical F12-MP2 validation lives in the C++ test
``test_ctf12_effective_hamiltonian``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import (
    QdkCtF12ScfSolver,
    ScfSolver,
    create,
    registry,
)
from qdk_chemistry.data import Structure, Wavefunction


class TestCtF12ScfRegistration:
    """The CT-F12 SCF solver is registered under the scf_solver type."""

    def test_default_scf_is_canonical(self):
        assert registry.show_default()["scf_solver"] == "qdk"

    def test_ct_f12_is_available(self):
        assert "qdk_ct_f12" in registry.available("scf_solver")

    def test_create_named(self):
        solver = create("scf_solver", "qdk_ct_f12")
        assert solver.name() == "qdk_ct_f12"
        assert solver.type_name() == "scf_solver"
        assert isinstance(solver, ScfSolver)
        assert isinstance(solver, QdkCtF12ScfSolver)


class TestCtF12ScfSettings:
    """The CT-F12 SCF settings schema and its constraints."""

    def test_default_settings(self):
        s = create("scf_solver", "qdk_ct_f12").settings()
        assert s.get("gamma") == 1.0
        assert s.get("cabs_basis") == ""
        assert s.get("frozen_core") == 0

    def test_create_with_kwargs(self):
        solver = create(
            "scf_solver",
            "qdk_ct_f12",
            gamma=1.5,
            cabs_basis="aug-cc-pvtz-optri",
        )
        assert solver.settings().get("gamma") == 1.5
        assert solver.settings().get("cabs_basis") == "aug-cc-pvtz-optri"

    def test_gamma_bound_enforced(self):
        s = create("scf_solver", "qdk_ct_f12").settings()
        with pytest.raises(ValueError, match="out of allowed range"):
            s.set("gamma", -1.0)


class TestCtF12ScfRun:
    """run() produces a relaxed F12-HF wavefunction for the neon atom."""

    def test_run_returns_relaxed_reference(self):
        mol = Structure(np.zeros((1, 3)), ["Ne"])
        solver = create(
            "scf_solver",
            "qdk_ct_f12",
            gamma=1.5,
            frozen_core=1,
            cabs_basis="aug-cc-pvdz-optri",
        )
        energy, relaxed = solver.run(mol, 0, 1, "aug-cc-pvdz")
        assert isinstance(relaxed, Wavefunction)
        assert np.isfinite(energy)

        orbitals = relaxed.get_orbitals()
        assert orbitals.has_energies()
        # The frozen core (1 orbital) is inactive; the rest is active.
        n_active = len(orbitals.get_active_space_indices()[0])
        assert n_active == orbitals.get_num_molecular_orbitals() - 1
        # Four valence pairs remain in the active space (frozen-core neon).
        assert relaxed.get_active_num_electrons() == (4, 4)
