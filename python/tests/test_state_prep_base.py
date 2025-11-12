"""Test for state preparation base class in QDK/Chemistry.

Test Categories:
    1. Factory Method Tests: Test the StatePrep.create factory method for creating
       state preparation instances of different types.

    2. Basic Functionality Tests: Test core functionality of state preparation
       classes including circuit creation and parameter validation.

    3. Error Handling Tests: Test proper error handling for edge cases such as
       insufficient determinants or invalid parameters.

Test Data:
    The tests use the 4e4o ethylene test case from the conftest.py fixtures,
    which provides a realistic electronic structure problem for validation.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import pytest

from qdk.chemistry.data import Configuration
from qdk.chemistry.state_preparation.base import (
    StatePrep,
    StatePrepAlgorithm,
    prepare_single_reference_state,
)
from qdk.chemistry.state_preparation.regular_isometry import RegularIsometryStatePrep
from qdk.chemistry.state_preparation.sparse_isometry import SparseIsometryGF2XStatePrep


def test_state_prep_factory(wavefunction_4e4o):
    """Test StatePrep factory algorithm instance creation.

    Test that StatePrep factory creates correct algorithm instances with parameters.
    """
    # Test creating a RegularIsometryStatePrep instance
    prep = StatePrep.from_algorithm(
        StatePrepAlgorithm.REGULAR_ISOMETRY,
        wavefunction_4e4o,
        max_dets=2,
        amplitude_threshold=0.01,
    )

    # Verify it's the right type
    assert isinstance(prep, RegularIsometryStatePrep)
    assert prep.algorithm == StatePrepAlgorithm.REGULAR_ISOMETRY
    assert prep.max_dets == 2
    assert prep.amplitude_threshold == 0.01

    with pytest.raises(
        ValueError,
        match="cannot be greater than the number of determinants",
    ):
        StatePrep.from_algorithm(
            StatePrepAlgorithm.REGULAR_ISOMETRY,
            wavefunction_4e4o,
            max_dets=6,
            amplitude_threshold=0.01,
        )


def test_asymmetric_active_space_error():
    """Test error for asymmetric active space in StatePrep."""

    class MockOrbitals:
        """Mock orbitals with asymmetric active space indices."""

        def get_active_space_indices(self):
            """Return asymmetric active space indices."""
            return ([0, 1, 2], [0, 1, 2, 3])

    class MockWavefunction:
        """Mock wavefunction for testing asymmetric active space."""

        def get_orbitals(self):
            """Return mock orbitals."""
            return MockOrbitals()

        def get_active_determinants(self):
            """Return mock determinants."""
            return [Configuration("2020000"), Configuration("2200000")]

        def get_coefficient(self, _):
            """Return mock coefficient."""
            return 1.0

    mock_wfn = MockWavefunction()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Active space contains 3 alpha orbitals and 4 beta orbitals. Asymmetric active spaces for alpha and beta "
            "orbitals are not supported for state preparation."
        ),
    ):
        StatePrep.from_algorithm(
            StatePrepAlgorithm.SPARSE_ISOMETRY_GF2X,
            mock_wfn,
            max_dets=2,
            amplitude_threshold=0.01,
        )


def test_sparse_isometry_gf2x_factory_wavefunction_4e4o(wavefunction_4e4o):
    """Test creating SparseIsometryGF2XStatePrep via factory pattern."""
    # Use factory pattern with correct enum value
    prep = StatePrep.from_algorithm(
        StatePrepAlgorithm.SPARSE_ISOMETRY_GF2X,
        wavefunction_4e4o,
        amplitude_threshold=0.001,
        save_outputs=False,
    )

    assert isinstance(prep, SparseIsometryGF2XStatePrep)
    assert prep.algorithm == StatePrepAlgorithm.SPARSE_ISOMETRY_GF2X
    circuit_qasm = prep.create_circuit_qasm()
    assert isinstance(circuit_qasm, str)


def test_state_prep_factory_invalid_algorithm(wavefunction_4e4o):
    """Test StatePrep factory with invalid algorithm raises ValueError."""
    with pytest.raises(ValueError, match="State preparation algorithm invalid_algorithm not implemented"):
        StatePrep.from_algorithm("invalid_algorithm", wavefunction_4e4o)


def test_single_reference_state_error_cases():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError, match="Bitstring cannot be empty"):
        prepare_single_reference_state("")

    with pytest.raises(ValueError, match="Bitstring must contain only '0' and '1' characters"):
        prepare_single_reference_state("1012")
