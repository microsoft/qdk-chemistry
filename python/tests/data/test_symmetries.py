"""Tests for the Symmetries data class — quantitative checks only."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import h5py
import pytest

from qdk_chemistry.data import Symmetries

from ..test_helpers import create_test_ansatz, create_test_wavefunction


class TestSymmetriesValidation:
    """Input validation and error handling."""

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="n_alpha"):
            Symmetries(n_alpha=-1, n_beta=0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError, match="n_beta"):
            Symmetries(n_alpha=0, n_beta=-1)

    def test_float_coercion(self):
        sym = Symmetries(n_alpha=2.0, n_beta=1.0)
        assert isinstance(sym.n_alpha, int)
        assert sym.n_alpha == 2


class TestSymmetriesComputation:
    """Quantitative tests for derived spin properties."""

    @pytest.mark.parametrize(
        ("n_alpha", "n_beta", "expected_particles", "expected_sz", "expected_mult"),
        [
            (2, 2, 4, 0.0, 1),    # singlet
            (2, 1, 3, 0.5, 2),    # doublet
            (3, 1, 4, 1.0, 3),    # triplet
            (1, 3, 4, -1.0, 3),   # beta-excess triplet
            (0, 0, 0, 0.0, 1),    # vacuum
        ],
    )
    def test_spin_properties(self, n_alpha, n_beta, expected_particles, expected_sz, expected_mult):
        sym = Symmetries(n_alpha=n_alpha, n_beta=n_beta)
        assert sym.n_particles == expected_particles
        assert sym.sz == expected_sz
        assert sym.spin_multiplicity == expected_mult


class TestSymmetriesFactoryMethods:
    """Tests for Symmetries.from_wavefunction and from_ansatz."""

    def test_from_wavefunction(self):
        wfn = create_test_wavefunction(num_orbitals=2)
        sym = Symmetries.from_wavefunction(wfn)
        n_alpha, n_beta = wfn.get_active_num_electrons()
        assert sym.n_alpha == n_alpha
        assert sym.n_beta == n_beta

    def test_from_ansatz_matches_from_wavefunction(self):
        ansatz = create_test_ansatz(num_orbitals=3)
        sym_ansatz = Symmetries.from_ansatz(ansatz)
        sym_wfn = Symmetries.from_wavefunction(ansatz.get_wavefunction())
        assert sym_ansatz == sym_wfn


class TestSymmetriesEquality:
    """Equality, hashing, and immutability."""

    def test_equality(self):
        assert Symmetries(n_alpha=1, n_beta=1) == Symmetries(n_alpha=1, n_beta=1)

    def test_inequality(self):
        assert Symmetries(n_alpha=1, n_beta=1) != Symmetries(n_alpha=2, n_beta=1)

    def test_hash_in_set(self):
        s = {Symmetries(n_alpha=1, n_beta=1), Symmetries(n_alpha=1, n_beta=1)}
        assert len(s) == 1

    def test_immutable(self):
        sym = Symmetries(n_alpha=1, n_beta=1)
        with pytest.raises(AttributeError):
            sym.n_alpha = 2  # type: ignore[misc]


class TestSymmetriesSerialization:
    """One canonical roundtrip per format."""

    def test_json_roundtrip(self):
        original = Symmetries(n_alpha=4, n_beta=2)
        restored = Symmetries.from_json(original.to_json())
        assert restored == original

    def test_hdf5_roundtrip(self, tmp_path):
        original = Symmetries(n_alpha=5, n_beta=3)
        filename = str(tmp_path / "test.symmetries.h5")
        with h5py.File(filename, "w") as f:
            original.to_hdf5(f)
        with h5py.File(filename, "r") as f:
            restored = Symmetries.from_hdf5(f)
        assert restored == original
