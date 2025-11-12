"""Test for regular isometry state preparation algorithms in QDK/Chemistry.

This module provides comprehensive tests for regular isometry methods for
preparing quantum states from electronic structure wavefunctions.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import pytest

from qdk.chemistry.state_preparation.regular_isometry import RegularIsometryStatePrep


def test_regular_isometry_state_prep(wavefunction_4e4o):
    """Test that RegularIsometryStatePrep creates valid quantum circuits."""
    # Create a state preparation instance
    prep = RegularIsometryStatePrep(wavefunction_4e4o, max_dets=2, amplitude_threshold=0.01)

    # Create a circuit
    circuit = prep.create_circuit_qasm()

    # Check that the circuit is valid
    assert isinstance(circuit, str)
    # Count number of qubits from "qubit[x] q;" to ensure 8 qubits (2 * 4 orbitals)
    qubit_pattern = re.search(r"qubit\[(\d+)\] q;", circuit)
    assert qubit_pattern is not None
    assert int(qubit_pattern.group(1)) == 2 * 4


def test_regular_isometry_state_prep_max_dets(wavefunction_10e6o):
    """Test that RegularIsometryStatePrep filter number of determinants."""
    # Create a state preparation instance
    prep = RegularIsometryStatePrep(wavefunction_10e6o, max_dets=2, amplitude_threshold=0.01)

    # Create a circuit
    circuit = prep.create_circuit_qasm()

    # Check that the circuit is valid
    assert isinstance(circuit, str)
    assert len(prep._filtered_bitstrings) == 2
    assert len(prep._filtered_coeffs) == 2


def test_regular_isometry_max_dets_validation(wavefunction_4e4o):
    """Test RegularIsometryStatePrep max_dets validation."""
    # Test with max_dets greater than available determinants
    with pytest.raises(ValueError, match=r"max_dets .* cannot be greater than"):
        RegularIsometryStatePrep(wavefunction_4e4o, max_dets=12)


def test_regular_isometry_invalid_max_dets(wavefunction_4e4o):
    """Test RegularIsometryStatePrep with invalid max_dets type."""
    # Create a simple wavefunction

    prep = RegularIsometryStatePrep(wavefunction_4e4o, max_dets=-1)

    # Should raise ValueError during filtering
    with pytest.raises(ValueError, match="max_dets must be a positive integer"):
        prep.create_circuit_qasm()


def test_regular_isometry_no_determinants_after_filtering(wavefunction_4e4o):
    """Test RegularIsometryStatePrep when no determinants remain after filtering."""
    # Set a high threshold that filters out all determinants
    prep = RegularIsometryStatePrep(wavefunction_4e4o, amplitude_threshold=0.99)

    with pytest.raises(ValueError, match="No determinants remain after filtering"):
        prep.create_circuit_qasm()
