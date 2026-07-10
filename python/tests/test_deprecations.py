"""Tests for the v1 -> v2 API deprecation surface.

These tests assert that:

- Retained v1 :class:`~qdk_chemistry.data.Orbitals` accessors emit a
  ``DeprecationWarning`` and return data equal to their v2 (symmetry-blocked)
  replacements.
- Removed/renamed ``qdk_chemistry.data`` names resolve to their v2 replacement
  via the module ``__getattr__`` shim, emitting a ``DeprecationWarning``.
- The removed ``EncodingMismatchError`` / ``validate_encoding_compatibility``
  helpers still resolve (with a ``DeprecationWarning``).
- A name whose v2 replacement is unavailable (``ControlledTimeEvolutionUnitary``)
  raises ``AttributeError`` rather than crashing the shim.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

import qdk_chemistry.data as qcd
from qdk_chemistry.data import (
    AmplitudeContainer,
    AmplitudeType,
    Configuration,
    Orbitals,
    StateVectorContainer,
    UnitaryContainer,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.symmetry import (
    spin_channel_indices,
    spin_channel_matrix,
    spin_channel_vector,
    spin_index_set,
)

from .test_helpers import create_test_basis_set


def _restricted_orbitals_with_active_space() -> Orbitals:
    """Build restricted Orbitals with energies and a [1, 2] active space."""
    coeffs = np.eye(4)
    energies = np.array([-1.0, -0.5, 0.5, 1.0])
    basis_set = create_test_basis_set(4, "test-deprecation")
    active = spin_index_set(4, [1, 2], [1, 2])
    return Orbitals(coeffs, energies, None, basis_set, active)


def test_get_coefficients_warns_and_matches_v2():
    """Orbitals.get_coefficients() warns and equals the v2 coefficients() tensor."""
    orb = _restricted_orbitals_with_active_space()
    with pytest.warns(DeprecationWarning, match="get_coefficients"):
        alpha, beta = orb.get_coefficients()
    coeffs = orb.coefficients()
    np.testing.assert_allclose(alpha, spin_channel_matrix(coeffs))
    np.testing.assert_allclose(beta, spin_channel_matrix(coeffs, beta=True))


def test_get_coefficients_alpha_beta_warn_and_match_v2():
    """The alpha/beta coefficient accessors warn and match the v2 spin channels."""
    orb = _restricted_orbitals_with_active_space()
    with pytest.warns(DeprecationWarning, match="get_coefficients_alpha"):
        alpha = orb.get_coefficients_alpha()
    with pytest.warns(DeprecationWarning, match="get_coefficients_beta"):
        beta = orb.get_coefficients_beta()
    coeffs = orb.coefficients()
    np.testing.assert_allclose(alpha, spin_channel_matrix(coeffs))
    np.testing.assert_allclose(beta, spin_channel_matrix(coeffs, beta=True))


def test_get_energies_warns_and_matches_v2():
    """Orbitals.get_energies() warns and equals the v2 energies() tensor."""
    orb = _restricted_orbitals_with_active_space()
    with pytest.warns(DeprecationWarning, match="get_energies"):
        alpha, beta = orb.get_energies()
    energies = orb.energies()
    np.testing.assert_allclose(alpha, spin_channel_vector(energies))
    np.testing.assert_allclose(beta, spin_channel_vector(energies, beta=True))


def test_get_active_space_indices_warns_and_matches_v2():
    """get_active_space_indices() warns and matches active_indices()/num_active_orbitals()."""
    orb = _restricted_orbitals_with_active_space()
    with pytest.warns(DeprecationWarning, match="get_active_space_indices"):
        alpha, beta = orb.get_active_space_indices()
    active = orb.active_indices()
    assert list(alpha) == spin_channel_indices(active, beta=False)
    assert list(beta) == spin_channel_indices(active, beta=True)
    assert len(alpha) == orb.num_active_orbitals()


def test_get_inactive_space_indices_warns_and_matches_v2():
    """get_inactive_space_indices() warns and matches inactive_indices()/num_inactive_orbitals()."""
    orb = _restricted_orbitals_with_active_space()
    with pytest.warns(DeprecationWarning, match="get_inactive_space_indices"):
        alpha, beta = orb.get_inactive_space_indices()
    inactive = orb.inactive_indices()
    assert list(alpha) == spin_channel_indices(inactive, beta=False)
    assert list(beta) == spin_channel_indices(inactive, beta=True)
    assert len(alpha) == orb.num_inactive_orbitals()


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("SlaterDeterminantContainer", StateVectorContainer),
        ("CasWavefunctionContainer", StateVectorContainer),
        ("SciWavefunctionContainer", StateVectorContainer),
        ("MP2Container", AmplitudeContainer),
        ("CoupledClusterContainer", AmplitudeContainer),
        ("TimeEvolutionUnitary", UnitaryRepresentation),
        ("TimeEvolutionUnitaryContainer", UnitaryContainer),
    ],
)
def test_deprecated_data_aliases_resolve_to_v2(name, replacement):
    """Deprecated container/unitary names warn and resolve to their v2 class."""
    with pytest.warns(DeprecationWarning, match=name):
        obj = getattr(qcd, name)
    assert obj is replacement


def test_deprecated_encoding_helpers_resolve():
    """EncodingMismatchError and validate_encoding_compatibility warn and resolve."""
    with pytest.warns(DeprecationWarning, match="EncodingMismatchError"):
        err = qcd.EncodingMismatchError
    assert issubclass(err, ValueError)
    with pytest.warns(DeprecationWarning, match="validate_encoding_compatibility"):
        fn = qcd.validate_encoding_compatibility
    assert callable(fn)


def test_controlled_time_evolution_unitary_alias_unavailable():
    """ControlledTimeEvolutionUnitary has no v2 target on this branch, so it raises."""
    with pytest.raises(AttributeError):
        _ = qcd.ControlledTimeEvolutionUnitary


def test_unknown_data_attribute_raises():
    """An unknown data attribute raises a normal AttributeError."""
    with pytest.raises(AttributeError):
        _ = qcd.ThisNameDoesNotExist


def _amplitude_orbitals() -> Orbitals:
    """Three restricted orbitals (two occupied, one virtual) with energies."""
    return Orbitals(np.eye(3), np.array([-1.0, -0.5, 0.5]), None, create_test_basis_set(3, "alias-construct"))


def test_statevector_container_aliases_construct():
    """The StateVectorContainer aliases construct as the v2 class, not just resolve by identity."""
    orbitals = _amplitude_orbitals()
    reference = Configuration.from_spin_half_string("220")
    for name in ("SlaterDeterminantContainer", "CasWavefunctionContainer", "SciWavefunctionContainer"):
        with pytest.warns(DeprecationWarning, match=name):
            cls = getattr(qcd, name)
        container = cls(reference, orbitals, "electrons")
        assert isinstance(container, StateVectorContainer)


def test_amplitude_container_aliases_construct():
    """The AmplitudeContainer aliases construct as the v2 class with the expected amplitude type."""
    orbitals = _amplitude_orbitals()
    reference = Wavefunction(StateVectorContainer(Configuration.from_spin_half_string("220"), orbitals, "electrons"))
    t1 = np.zeros(2)
    t2 = np.array([0.001, 0.002, 0.003, 0.004])
    for name, amplitude_type in (
        ("MP2Container", AmplitudeType.MollerPlesset),
        ("CoupledClusterContainer", AmplitudeType.CoupledCluster),
    ):
        with pytest.warns(DeprecationWarning, match=name):
            cls = getattr(qcd, name)
        container = cls(orbitals, reference, amplitude_type, t1, t2)
        assert isinstance(container, AmplitudeContainer)
        assert container.get_amplitude_type() == amplitude_type
