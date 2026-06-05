"""Tests for the SBT-native Orbitals pybind surface."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.data import (
    Orbitals,
)
from qdk_chemistry.data import symmetry as sym

from .test_helpers import create_test_basis_set


@pytest.fixture
def restricted_spin():
    """A restricted (equivalent) spin vocabulary with alpha/beta labels."""
    syms = sym.SymmetryProduct([sym.axes.spin(1, True)])
    alpha = sym.SymmetryLabel([sym.axes.alpha()])
    beta = sym.SymmetryLabel([sym.axes.beta()])
    return syms, alpha, beta


def test_v1_restricted_orbitals_expose_sbt_accessors():
    """An RHF Orbitals built from dense data exposes SBT-native containers."""
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 1.0]])
    basis_set = create_test_basis_set(3, "test-sbt-restricted")
    orb = Orbitals(coeffs, None, None, basis_set)

    assert orb.num_modes() == orb.get_num_molecular_orbitals()

    coeff_sbt = orb.coefficients()
    assert isinstance(coeff_sbt, sym.SymmetryBlockedTensorRank2)

    alpha = sym.SymmetryLabel([sym.axes.alpha()])
    beta = sym.SymmetryLabel([sym.axes.beta()])
    assert np.allclose(coeff_sbt.block((alpha, alpha)), coeffs)
    assert np.allclose(coeff_sbt.block((beta, beta)), coeffs)


def test_v1_unrestricted_orbitals_are_not_restricted():
    """A UHF Orbitals exposes distinct alpha/beta coefficient blocks."""
    coeffs_alpha = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 1.0]])
    coeffs_beta = np.array([[0.8, 0.2], [0.2, -0.8], [0.1, 0.9]])
    basis_set = create_test_basis_set(3, "test-sbt-unrestricted")
    orb = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)
    coeff_sbt = orb.coefficients()

    alpha = sym.SymmetryLabel([sym.axes.alpha()])
    beta = sym.SymmetryLabel([sym.axes.beta()])
    assert np.allclose(coeff_sbt.block((alpha, alpha)), coeffs_alpha)
    assert np.allclose(coeff_sbt.block((beta, beta)), coeffs_beta)


def test_sbt_native_constructor_round_trips(restricted_spin):
    """Constructing Orbitals from containers preserves coefficients/energies."""
    syms, alpha, beta = restricted_spin
    nao, nmo = 3, 2
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 1.0]])
    energies = np.array([-0.5, 0.3])

    coeff_sbt = sym.SymmetryBlockedTensorRank2(
        [syms, syms],
        [{alpha: nao, beta: nao}, {alpha: nmo, beta: nmo}],
        [((alpha, alpha), coeffs)],
    )
    energy_sbt = sym.SymmetryBlockedTensorRank1(
        [syms],
        [{alpha: nmo, beta: nmo}],
        [((alpha,), energies)],
    )
    orb = Orbitals(coeff_sbt, energy_sbt, None, None)

    assert orb.get_num_molecular_orbitals() == nmo
    assert orb.get_num_atomic_orbitals() == nao
    assert orb.num_modes() == nmo
    assert np.allclose(orb.coefficients().block((alpha, alpha)), coeffs)
    assert np.allclose(orb.energies().block((alpha,)), energies)
