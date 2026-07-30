"""Tests for the EffectiveHamiltonianConstructor bindings and registry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import (
    EffectiveHamiltonianConstructor,
    QdkSchriefferWolffPT2Constructor,
)
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import Orbitals, Structure
from qdk_chemistry.data.symmetry import SymmetryLabel, axes

_TYPE = "effective_hamiltonian_constructor"


class TestEffectiveHamiltonianConstructor:
    """Registry, factory, and settings coverage for the downfolding bindings."""

    def test_factory_registration(self):
        """swpt2 is registered and resolvable via the factory."""
        available = algorithms.available(_TYPE)
        assert isinstance(available, list)
        assert "qdk_swpt2" in available

        # default and explicit creation both yield the swpt2 constructor
        default = algorithms.create(_TYPE)
        assert isinstance(default, EffectiveHamiltonianConstructor)
        assert default.name() == "qdk_swpt2"
        assert default.type_name() == _TYPE

        explicit = algorithms.create(_TYPE, "qdk_swpt2")
        assert explicit.name() == "qdk_swpt2"

        # a nonexistent name raises
        with pytest.raises((KeyError, RuntimeError)):
            algorithms.create(_TYPE, "nonexistent_downfolder")

    @pytest.mark.parametrize("alias", ["swpt2", "sw", "schrieffer_wolff"])
    def test_aliases_resolve(self, alias):
        """Aliases (incl. the legacy `swpt2`) resolve to the qdk_swpt2 implementation."""
        assert algorithms.create(_TYPE, alias).name() == "qdk_swpt2"

    def test_direct_construction(self):
        """The concrete class constructs directly."""
        constructor = QdkSchriefferWolffPT2Constructor()
        assert isinstance(constructor, EffectiveHamiltonianConstructor)
        assert constructor.name() == "qdk_swpt2"

    def test_settings_knobs(self):
        """Denominator/regularization settings expose their defaults and are settable."""
        constructor = algorithms.create(_TYPE, "swpt2")
        settings = constructor.settings()
        # flow regularization is on by default at the constructor layer
        assert settings.get("denom_floor") == pytest.approx(1e-8)
        assert settings.get("denom_shift") == pytest.approx(0.0)
        assert settings.get("denom_flow") == pytest.approx(1.0)
        assert settings.get("intruder_warn_amplitude") == pytest.approx(1.0)

        settings.set("denom_shift", 0.5)
        assert constructor.settings().get("denom_shift") == pytest.approx(0.5)

    def test_accepts_mean_field_hf_reference(self):
        """A mean-field HF reference (no active 1-RDM) is accepted directly."""
        alpha = SymmetryLabel([axes.alpha()])
        water = Structure(
            ["O", "H", "H"],
            np.array(
                [
                    [0.0, -0.0757918436, 0.0],
                    [0.866811829, 0.6014357793, 0.0],
                    [-0.866811829, 0.6014357793, 0.0],
                ]
            )
            * ANGSTROM_TO_BOHR,
        )
        _, wfn_hf = algorithms.create("scf_solver").run(water, 0, 1, "sto-3g")
        orbitals = wfn_hf.get_orbitals()

        selector = algorithms.create("active_space_selector", "qdk_valence")
        selector.settings().set("num_active_electrons", 2)
        selector.settings().set("num_active_orbitals", 2)
        reference = selector.run(wfn_hf)  # single-determinant HF, no 1-RDM

        def with_active_space(orb, active, inactive):
            """Return a copy of `orb` with the given active/inactive index sets."""
            ca, _ = orb.get_coefficients()
            ea, _ = orb.get_energies()
            return Orbitals(ca, ea, None, orb.get_basis_set(), (list(active), list(inactive)))

        # small window = active P + the lowest orbital outside P and the core
        ref_orbs = reference.get_orbitals()
        active = list(ref_orbs.active_indices().indices(alpha))
        core = list(ref_orbs.inactive_indices().indices(alpha))
        norb = orbitals.get_coefficients()[0].shape[1]
        used = set(active) | set(core)
        window = sorted([*active, next(i for i in range(norb) if i not in used)])

        h_window = algorithms.create("hamiltonian_constructor").run(with_active_space(orbitals, window, core))
        downfolder = algorithms.create(_TYPE, "swpt2")
        h_eff = downfolder.run(reference, h_window)  # must not raise
        assert h_eff is not None
