"""Tests for the Symmetries data class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import pytest

from qdk_chemistry.data import Symmetries

from .test_helpers import create_test_ansatz, create_test_wavefunction


class TestSymmetriesConstruction:
    """Tests for constructing Symmetries instances."""

    def test_basic_construction(self):
        """Symmetries can be constructed with explicit alpha and beta counts."""
        sym = Symmetries(n_alpha=2, n_beta=3)
        assert sym.n_alpha == 2
        assert sym.n_beta == 3

    def test_zero_electrons(self):
        """Symmetries can represent a vacuum state with zero electrons."""
        sym = Symmetries(n_alpha=0, n_beta=0)
        assert sym.n_alpha == 0
        assert sym.n_beta == 0
        assert sym.n_particles == 0

    def test_negative_alpha_raises(self):
        """Negative n_alpha raises ValueError."""
        with pytest.raises(ValueError, match="n_alpha"):
            Symmetries(n_alpha=-1, n_beta=0)

    def test_negative_beta_raises(self):
        """Negative n_beta raises ValueError."""
        with pytest.raises(ValueError, match="n_beta"):
            Symmetries(n_alpha=0, n_beta=-1)

    def test_float_coercion(self):
        """Float values are coerced to int."""
        sym = Symmetries(n_alpha=2.0, n_beta=1.0)
        assert sym.n_alpha == 2
        assert sym.n_beta == 1
        assert isinstance(sym.n_alpha, int)
        assert isinstance(sym.n_beta, int)


class TestSymmetriesDerivedProperties:
    """Tests for derived properties of Symmetries."""

    def test_n_particles(self):
        """n_particles is the sum of alpha and beta electrons."""
        sym = Symmetries(n_alpha=3, n_beta=2)
        assert sym.n_particles == 5

    def test_sz_singlet(self):
        """Sz is zero for equal alpha and beta counts."""
        sym = Symmetries(n_alpha=2, n_beta=2)
        assert sym.sz == 0.0

    def test_sz_doublet(self):
        """Sz is 0.5 for one unpaired alpha electron."""
        sym = Symmetries(n_alpha=2, n_beta=1)
        assert sym.sz == 0.5

    def test_sz_triplet(self):
        """Sz is 1.0 for two unpaired alpha electrons."""
        sym = Symmetries(n_alpha=3, n_beta=1)
        assert sym.sz == 1.0

    def test_spin_multiplicity_singlet(self):
        """Spin multiplicity is 1 for a singlet."""
        sym = Symmetries(n_alpha=2, n_beta=2)
        assert sym.spin_multiplicity == 1

    def test_spin_multiplicity_doublet(self):
        """Spin multiplicity is 2 for a doublet."""
        sym = Symmetries(n_alpha=2, n_beta=1)
        assert sym.spin_multiplicity == 2

    def test_spin_multiplicity_triplet(self):
        """Spin multiplicity is 3 for a triplet."""
        sym = Symmetries(n_alpha=3, n_beta=1)
        assert sym.spin_multiplicity == 3


class TestSymmetriesFactoryMethods:
    """Tests for Symmetries factory methods."""

    def test_from_wavefunction(self):
        """Symmetries.from_wavefunction reads active electron counts correctly."""
        wfn = create_test_wavefunction(num_orbitals=2)
        sym = Symmetries.from_wavefunction(wfn)

        # The test wavefunction has config "20" → 1 alpha + 1 beta in active space
        n_alpha, n_beta = wfn.get_active_num_electrons()
        assert sym.n_alpha == n_alpha
        assert sym.n_beta == n_beta

    def test_from_ansatz(self):
        """Symmetries.from_ansatz delegates to from_wavefunction correctly."""
        ansatz = create_test_ansatz(num_orbitals=2)
        sym = Symmetries.from_ansatz(ansatz)

        wfn = ansatz.get_wavefunction()
        n_alpha, n_beta = wfn.get_active_num_electrons()
        assert sym.n_alpha == n_alpha
        assert sym.n_beta == n_beta

    def test_from_wavefunction_matches_from_ansatz(self):
        """Factory methods produce equal Symmetries for the same underlying wavefunction."""
        ansatz = create_test_ansatz(num_orbitals=3)
        sym_from_ansatz = Symmetries.from_ansatz(ansatz)
        sym_from_wfn = Symmetries.from_wavefunction(ansatz.get_wavefunction())
        assert sym_from_ansatz == sym_from_wfn


class TestSymmetriesDunderMethods:
    """Tests for Symmetries dunder methods."""

    def test_repr(self):
        """Repr contains alpha and beta counts."""
        sym = Symmetries(n_alpha=2, n_beta=3)
        assert repr(sym) == "Symmetries(n_alpha=2, n_beta=3)"

    def test_equality(self):
        """Equal Symmetries compare as equal."""
        assert Symmetries(n_alpha=1, n_beta=1) == Symmetries(n_alpha=1, n_beta=1)

    def test_inequality(self):
        """Unequal Symmetries compare as not equal."""
        assert Symmetries(n_alpha=1, n_beta=1) != Symmetries(n_alpha=2, n_beta=1)
        assert Symmetries(n_alpha=1, n_beta=1) != Symmetries(n_alpha=1, n_beta=2)

    def test_not_equal_to_other_type(self):
        """Symmetries is not equal to non-Symmetries objects."""
        assert Symmetries(n_alpha=1, n_beta=1) != "not a symmetries"
        assert Symmetries(n_alpha=1, n_beta=1) != 42

    def test_hash_equal(self):
        """Equal Symmetries produce equal hashes."""
        a = Symmetries(n_alpha=2, n_beta=3)
        b = Symmetries(n_alpha=2, n_beta=3)
        assert hash(a) == hash(b)

    def test_hash_usable_in_set(self):
        """Symmetries can be used in sets and as dict keys."""
        s = {Symmetries(n_alpha=1, n_beta=1), Symmetries(n_alpha=1, n_beta=1)}
        assert len(s) == 1

    def test_immutable(self):
        """Symmetries attributes cannot be reassigned."""
        sym = Symmetries(n_alpha=1, n_beta=1)
        with pytest.raises(AttributeError):
            sym.n_alpha = 2  # type: ignore[misc]
        with pytest.raises(AttributeError):
            sym.n_beta = 2  # type: ignore[misc]
