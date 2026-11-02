"""Tests for the EffectiveHamiltonianConstructor bindings and registry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from types import SimpleNamespace

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

_WATER = Structure(
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


@pytest.fixture(scope="module")
def water_window():
    """Water/STO-3G with a CAS(2e,2o) reference and a Hamiltonian over ``P`` plus one orbital."""
    alpha = SymmetryLabel([axes.alpha()])
    _, wfn_hf = algorithms.create("scf_solver").run(_WATER, 0, 1, "sto-3g")
    orbitals = wfn_hf.get_orbitals()
    norb = orbitals.get_num_molecular_orbitals()

    selector = algorithms.create("active_space_selector", "qdk_valence")
    selector.settings().set("num_active_electrons", 2)
    selector.settings().set("num_active_orbitals", 2)
    reference = selector.run(wfn_hf)

    ref_orbs = reference.get_orbitals()
    kept = [int(i) for i in ref_orbs.active_indices().indices(alpha)]
    core = [int(i) for i in ref_orbs.inactive_indices().indices(alpha)]
    used = set(kept) | set(core)
    window = sorted([*kept, next(i for i in range(norb) if i not in used)])

    window_orbitals = Orbitals(
        orbitals.coefficients().block([alpha, alpha]),
        orbitals.energies().block([alpha]),
        None,
        orbitals.get_basis_set(),
        active_indices=spin_index_set(norb, window, window),
        inactive_indices=spin_index_set(norb, core, core),
    )
    h_window = algorithms.create("hamiltonian_constructor").run(window_orbitals)
    return SimpleNamespace(
        reference=reference,
        hamiltonian=h_window,
        kept_space=spin_index_set(norb, kept, kept),
        window_space=spin_index_set(norb, window, window),
        kept_in_window=[window.index(i) for i in kept],
    )


class TestEffectiveHamiltonianConstructor:
    """Registry, factory, and settings coverage for the downfolding bindings."""

    def test_factory_registration(self):
        """qdk_swpt2 is registered as an effective-Hamiltonian constructor and is the default."""
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

    def test_direct_construction(self):
        """The concrete class constructs directly."""
        constructor = QdkSchriefferWolffPT2Constructor()
        assert isinstance(constructor, EffectiveHamiltonianConstructor)
        assert constructor.name() == "qdk_swpt2"

    def test_settings_knobs(self):
        """Denominator/regularization settings expose their defaults and are settable."""
        constructor = algorithms.create(_TYPE, "qdk_swpt2")
        settings = constructor.settings()
        # sigma^2 regularization is on by default at the constructor layer
        assert settings.get("regularizer_sigma2") == pytest.approx(1.0)
        assert settings.get("semicanonicalize") is True
        assert settings.get("fold_above_two_body") is True
        assert settings.get("max_folded_occupation_deviation") == pytest.approx(0.5)

        settings.set("regularizer_sigma2", 0.4)
        assert constructor.settings().get("regularizer_sigma2") == pytest.approx(0.4)

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
        downfolder = algorithms.create(_TYPE, "qdk_swpt2")
        p_space = spin_index_set(norb, active, active)  # kept space P as an index set
        h_eff = downfolder.run(reference, h_window, p_space)  # must not raise
        assert isinstance(h_eff, Hamiltonian)

    def test_full_window_spanning_reference_core(self):
        """Downfold a full-orbital window (the natural ``ham.run(orbitals)`` path).

        The core orbitals are reference core but lie inside ``W``, so the
        downfold folds them itself instead of inheriting them from the window
        Hamiltonian, whose own inactive set is empty here. They must still come
        back labelled inactive, exactly once each.
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
        h_eff = algorithms.create(_TYPE, "qdk_swpt2").run(
            reference, h_full, ref_orbs.active_indices()
        )  # must not raise

        assert isinstance(h_eff, Hamiltonian)
        emitted = h_eff.get_orbitals()
        assert list(emitted.active_indices().indices(alpha)) == active
        assert list(emitted.inactive_indices().indices(alpha)) == core

    def test_empty_external_space_is_the_identity(self, water_window):
        """Keeping the whole window leaves nothing to fold, so the integrals come back unchanged."""
        case = water_window
        h_eff = algorithms.create(_TYPE, "qdk_swpt2").run(case.reference, case.hamiltonian, case.window_space)

        assert h_eff.get_core_energy() == pytest.approx(case.hamiltonian.get_core_energy(), abs=1e-10)
        one_body, _ = h_eff.get_one_body_integrals()
        window_one_body, _ = case.hamiltonian.get_one_body_integrals()
        np.testing.assert_allclose(one_body, window_one_body, atol=1e-10)
        np.testing.assert_allclose(
            h_eff.get_two_body_integrals()[0], case.hamiltonian.get_two_body_integrals()[0], atol=1e-10
        )

    def test_downfold_dresses_the_two_body_block_and_keeps_four_fold_symmetry(self, water_window):
        """The dressing is nonzero, hermitian, and -- as documented -- not 8-fold symmetric."""
        case = water_window
        h_eff = algorithms.create(_TYPE, "qdk_swpt2").run(case.reference, case.hamiltonian, case.kept_space)

        kept = case.kept_in_window
        n = len(kept)
        window_one_body, _ = case.hamiltonian.get_one_body_integrals()
        window_norb = window_one_body.shape[0]
        g = np.asarray(h_eff.get_two_body_integrals()[0]).reshape(n, n, n, n)
        window_g = np.asarray(case.hamiltonian.get_two_body_integrals()[0])
        assert window_g.size == window_norb**4
        bare = window_g.reshape((window_norb,) * 4)
        bare = bare[np.ix_(kept, kept, kept, kept)]

        assert np.abs(g - bare).max() > 1e-6, "the downfold must actually dress the kept space"
        np.testing.assert_allclose(g, g.transpose(1, 0, 3, 2), atol=1e-10)  # hermiticity
        np.testing.assert_allclose(g, g.transpose(2, 3, 0, 1), atol=1e-10)  # electron exchange
        # The bra swap of a genuine Coulomb integral does not survive the commutator, so
        # consumers reading only the 8-fold-unique elements get a different operator.
        assert np.abs(g - g.transpose(1, 0, 2, 3)).max() > 1e-6

    def test_rejects_non_finite_regularizer(self, water_window):
        """A non-finite sigma must fail loudly rather than silently selecting the bare path."""
        case = water_window
        downfolder = algorithms.create(_TYPE, "qdk_swpt2")
        downfolder.settings().set("regularizer_sigma2", float("nan"))
        with pytest.raises((ValueError, RuntimeError)):
            downfolder.run(case.reference, case.hamiltonian, case.kept_space)
