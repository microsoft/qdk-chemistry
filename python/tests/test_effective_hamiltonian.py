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
    QdkDsrgPT2Constructor,
)
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Configuration,
    Hamiltonian,
    ModelOrbitals,
    Orbitals,
    StateVectorContainer,
    Structure,
    Wavefunction,
)
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

    def test_direct_construction(self):
        """The concrete class constructs directly."""
        constructor = QdkDsrgPT2Constructor()
        assert isinstance(constructor, EffectiveHamiltonianConstructor)
        assert constructor.name() == "qdk_dsrg_pt2"

    def test_settings_knobs(self):
        """flow_parameter exposes its default and is settable."""
        constructor = algorithms.create(_TYPE, "qdk_dsrg_pt2")
        settings = constructor.settings()
        assert settings.get("flow_parameter") == pytest.approx(0.5)

        settings.set("flow_parameter", 0.2)
        assert constructor.settings().get("flow_parameter") == pytest.approx(0.2)

    def test_full_window_spanning_reference_core(self):
        """Downfold a full-orbital window (the natural ``ham.run(orbitals)`` path).

        The core orbitals are reference core but lie inside ``W``, so the
        downfold folds them itself instead of inheriting them from the window
        Hamiltonian, whose own inactive set is empty here. They must still come
        back labelled inactive, exactly once each.
        """
        alpha = SymmetryLabel([axes.alpha()])
        _, wfn_hf = algorithms.create("scf_solver").run(_WATER, 0, 1, "sto-3g")

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
        h_eff = algorithms.create(_TYPE, "qdk_dsrg_pt2").run(
            reference, h_full, ref_orbs.active_indices()
        )  # must not raise

        assert isinstance(h_eff, Hamiltonian)
        emitted = h_eff.get_orbitals()
        assert list(emitted.active_indices().indices(alpha)) == active
        assert list(emitted.inactive_indices().indices(alpha)) == core

    def test_empty_external_space_is_the_identity(self, water_window):
        """Keeping the whole window leaves nothing to fold, so the integrals come back unchanged."""
        case = water_window
        h_eff = algorithms.create(_TYPE, "qdk_dsrg_pt2").run(case.reference, case.hamiltonian, case.window_space)

        assert h_eff.get_core_energy() == pytest.approx(case.hamiltonian.get_core_energy(), abs=1e-10)
        one_body, _ = h_eff.get_one_body_integrals()
        window_one_body, _ = case.hamiltonian.get_one_body_integrals()
        np.testing.assert_allclose(one_body, window_one_body, atol=1e-10)
        np.testing.assert_allclose(
            h_eff.get_two_body_integrals()[0], case.hamiltonian.get_two_body_integrals()[0], atol=1e-10
        )

    def test_zero_flow_returns_bare_operator_over_p(self, water_window):
        """flow_parameter=0 disables the DSRG transformation and returns the bare operator over P."""
        case = water_window
        downfolder = algorithms.create(_TYPE, "qdk_dsrg_pt2")
        downfolder.settings().set("flow_parameter", 0.0)
        h_eff = downfolder.run(case.reference, case.hamiltonian, case.kept_space)

        kept = case.kept_in_window
        window_one_body, _ = case.hamiltonian.get_one_body_integrals()
        window_norb = window_one_body.shape[0]
        bare_h1 = window_one_body[np.ix_(kept, kept)]
        bare_h2 = np.asarray(case.hamiltonian.get_two_body_integrals()[0]).reshape((window_norb,) * 4)
        bare_h2 = bare_h2[np.ix_(kept, kept, kept, kept)].reshape(-1)

        assert h_eff.get_core_energy() == pytest.approx(case.hamiltonian.get_core_energy(), abs=1e-12)
        np.testing.assert_allclose(h_eff.get_one_body_integrals()[0], bare_h1, atol=1e-12)
        np.testing.assert_allclose(h_eff.get_two_body_integrals()[0], bare_h2, atol=1e-12)

    def test_downfold_dresses_the_two_body_block_and_keeps_four_fold_symmetry(self, water_window):
        """The dressing is nonzero, hermitian, and -- as documented -- not 8-fold symmetric."""
        case = water_window
        h_eff = algorithms.create(_TYPE, "qdk_dsrg_pt2").run(case.reference, case.hamiltonian, case.kept_space)

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

    def test_downfolded_hamiltonian_round_trips_through_json(self, water_window, tmp_path):
        """The folded core is inactive without an inactive Fock matrix, and still loads from JSON."""
        case = water_window
        h_eff = algorithms.create(_TYPE, "qdk_dsrg_pt2").run(case.reference, case.hamiltonian, case.kept_space)
        assert list(h_eff.get_orbitals().inactive_indices().indices(SymmetryLabel([axes.alpha()])))
        filename = str(tmp_path / "downfolded.hamiltonian.json")
        h_eff.to_json_file(filename)
        loaded = Hamiltonian.from_json_file(filename)
        assert loaded.get_core_energy() == pytest.approx(h_eff.get_core_energy(), abs=1e-12)
        np.testing.assert_allclose(loaded.get_one_body_integrals()[0], h_eff.get_one_body_integrals()[0], atol=1e-12)
        np.testing.assert_allclose(loaded.get_two_body_integrals()[0], h_eff.get_two_body_integrals()[0], atol=1e-12)

    def test_rejects_non_finite_flow_parameter(self, water_window):
        """A non-finite flow parameter must fail loudly rather than silently selecting the bare path."""
        case = water_window
        downfolder = algorithms.create(_TYPE, "qdk_dsrg_pt2")
        downfolder.settings().set("flow_parameter", float("nan"))
        with pytest.raises((ValueError, RuntimeError)):
            downfolder.run(case.reference, case.hamiltonian, case.kept_space)

    def test_rejects_non_finite_integrals(self, water_window):
        """Non-finite integrals are rejected at the boundary."""
        case = water_window
        one_body, _ = case.hamiltonian.get_one_body_integrals()
        two_body = np.array(case.hamiltonian.get_two_body_integrals()[0])
        two_body[0] = np.nan
        container = CanonicalFourCenterHamiltonianContainer(
            one_body,
            two_body,
            case.hamiltonian.get_orbitals(),
            case.hamiltonian.get_core_energy(),
            np.empty((0, 0)),
        )
        with pytest.raises(ValueError, match="non-finite"):
            algorithms.create(_TYPE, "qdk_dsrg_pt2").run(case.reference, Hamiltonian(container), case.kept_space)

    def test_rejects_correlated_reference(self, water_window):
        """Multi-determinant references are rejected."""
        case = water_window
        h_p = algorithms.create("hamiltonian_constructor").run(case.reference.get_orbitals())
        cas = algorithms.create("multi_configuration_calculator", "macis_cas")
        cas.settings().set("calculate_one_rdm", True)
        _, cas_wfn = cas.run(h_p, 1, 1)
        with pytest.raises(ValueError, match="single-determinant"):
            algorithms.create(_TYPE, "qdk_dsrg_pt2").run(cas_wfn, case.hamiltonian, case.kept_space)

    def test_matches_independent_values_when_the_reference_density_is_not_semicanonical(self):
        """Excited determinants whose semicanonical 1-RDM is not diagonal exercise the singles' active-space correction.

        The expected values come from an independent evaluation of P(H + [H~, A]/2)P in the full Fock space
        (Li & Evangelista 2015, eqs. 35-39), less its reference-normal-ordered three-body part.
        """
        n, p = 5, [1, 2, 3]
        idx = np.arange(n)
        one_body = np.diag(np.linspace(-2.0, 1.4, n)) + 0.04 * np.cos(np.add.outer(idx, idx))
        factors = 0.25 * np.sin(
            1.0 + np.arange(4)[:, None, None] + 0.7 * np.add.outer(idx, idx) + 0.3 * np.multiply.outer(idx, idx)
        )
        two_body = np.einsum("lpq,lrs->pqrs", factors, factors).ravel()
        empty = np.empty((0, 0))
        closed_shell = [
            [-1.351189221109, -0.337193172810, -0.172398121086],
            [-0.337193172810, -0.709598435596, 0.059747549607],
            [-0.172398121086, 0.059747549607, 0.577129719002],
        ]
        cases = [
            ("20200", False, -3.579592577784, closed_shell, closed_shell),
            (
                "2u0d0",
                True,
                -3.547663065324,
                [
                    [-1.390153486186, -0.334179849259, -0.195528033827],
                    [-0.334179849259, -0.742150875760, 0.037815040782],
                    [-0.195528033827, 0.037815040782, 0.678805617545],
                ],
                [
                    [-1.351986305377, -0.349546300520, -0.203746749714],
                    [-0.349546300520, -0.754557102533, 0.040610717282],
                    [-0.203746749714, 0.040610717282, 0.656693872486],
                ],
            ),
        ]
        for determinant, unrestricted, scalar, alpha, beta in cases:
            orbitals = ModelOrbitals(
                spin_index_set(n, list(range(n)), list(range(n)), equivalent=not unrestricted),
                spin_index_set(n, [], [], equivalent=not unrestricted),
            )
            container = (
                CanonicalFourCenterHamiltonianContainer(
                    one_body, one_body, two_body, two_body, two_body, orbitals, 0.3, empty, empty
                )
                if unrestricted
                else CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.3, empty)
            )
            reference = Wavefunction(
                StateVectorContainer(np.array([1.0]), [Configuration.from_spin_half_string(determinant)], orbitals)
            )
            h_eff = algorithms.create(_TYPE, "qdk_dsrg_pt2").run(
                reference, Hamiltonian(container), spin_index_set(n, p, p, equivalent=not unrestricted)
            )
            assert h_eff.get_core_energy() == pytest.approx(scalar, abs=1e-10)
            np.testing.assert_allclose(h_eff.get_one_body_integrals()[0], alpha, atol=1e-10)
            np.testing.assert_allclose(h_eff.get_one_body_integrals()[1], beta, atol=1e-10)
