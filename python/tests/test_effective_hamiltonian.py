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
from qdk_chemistry.data import Hamiltonian, Orbitals, Structure
from qdk_chemistry.data.symmetry import SymmetryLabel, axes, spin_index_set

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

    @pytest.mark.parametrize("alias", ["swpt2", "schrieffer_wolff"])
    def test_aliases_resolve(self, alias):
        """Aliases resolve to the qdk_swpt2 implementation."""
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
        assert settings.get("denom_imaginary_shift") == pytest.approx(0.0)
        assert settings.get("denom_flow") == pytest.approx(1.0)
        assert settings.get("semicanonicalize") is True
        assert settings.get("fold_above_two_body") is True
        assert settings.get("max_folded_occupation_deviation") == pytest.approx(0.5)

        settings.set("denom_flow", 0.0)
        settings.set("denom_imaginary_shift", 0.5)
        assert constructor.settings().get("denom_flow") == pytest.approx(0.0)
        assert constructor.settings().get("denom_imaginary_shift") == pytest.approx(0.5)

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
            n = orb.get_num_molecular_orbitals()
            return Orbitals(
                orb.coefficients().block([alpha, alpha]),
                orb.energies().block([alpha]),
                None,
                orb.get_basis_set(),
                active_indices=spin_index_set(n, list(active), list(active)),
                inactive_indices=spin_index_set(n, list(inactive), list(inactive)),
            )

        # small window = active P + the lowest orbital outside P and the core
        ref_orbs = reference.get_orbitals()
        active = list(ref_orbs.active_indices().indices(alpha))
        core = list(ref_orbs.inactive_indices().indices(alpha))
        norb = orbitals.get_num_molecular_orbitals()
        used = set(active) | set(core)
        window = sorted([*active, next(i for i in range(norb) if i not in used)])

        h_window = algorithms.create("hamiltonian_constructor").run(with_active_space(orbitals, window, core))
        downfolder = algorithms.create(_TYPE, "swpt2")
        p_space = spin_index_set(norb, active, active)  # kept space P as an index set
        h_eff = downfolder.run(reference, h_window, p_space)  # must not raise
        assert isinstance(h_eff, Hamiltonian)

    def test_full_window_spanning_reference_core(self):
        """Downfold a full-orbital window (the natural ``ham.run(orbitals)`` path).

        The window spans the reference core, so each core orbital appears both as
        a folded window orbital and in the reference inactive set; the emitted
        inactive index set must be deduplicated (regression for a
        strictly-increasing-index crash).
        """
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

        selector = algorithms.create("active_space_selector", "qdk_valence")
        selector.settings().set("num_active_electrons", 2)
        selector.settings().set("num_active_orbitals", 2)
        reference = selector.run(wfn_hf)

        ref_orbs = reference.get_orbitals()
        active = [int(i) for i in ref_orbs.active_indices().indices(alpha)]
        core = [int(i) for i in ref_orbs.inactive_indices().indices(alpha)]

        # window over the full orbital set (spans the reference core)
        h_full = algorithms.create("hamiltonian_constructor").run(wfn_hf.get_orbitals())
        # P == reference active space: pass its index set straight through
        h_eff = algorithms.create(_TYPE, "swpt2").run(reference, h_full, ref_orbs.active_indices())  # must not raise

        assert isinstance(h_eff, Hamiltonian)
        emitted = h_eff.get_orbitals()
        assert list(emitted.active_indices().indices(alpha)) == active
        assert list(emitted.inactive_indices().indices(alpha)) == core
