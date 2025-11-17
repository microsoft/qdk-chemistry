"""Tests for QDK/Chemistry interop with Qiskit noise models."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings

from qiskit_aer.noise import NoiseModel

from qdk_chemistry.noise_models import QuantumErrorProfile, SupportedGate
from qdk_chemistry.plugins.qiskit._interop.noise_model import get_noise_model_from_profile


def test_get_qiskit_noise_model(simple_error_profile):
    """Test generation of a noise model from a quantum error profile."""
    noise_model = get_noise_model_from_profile(simple_error_profile)
    assert isinstance(noise_model, NoiseModel)
    assert set(noise_model.basis_gates) == set(simple_error_profile.basis_gates)
    assert set(noise_model.noise_instructions) == {"h", "cx"}


def test_get_noise_model_except(simple_error_profile):
    """Test generation of a noise model with excluded gates from QuantumErrorProfile."""
    exclude_gates = ["cx"]
    noise_model = get_noise_model_from_profile(simple_error_profile, exclude_gates)
    assert isinstance(noise_model, NoiseModel)
    for gate in exclude_gates:
        assert gate in noise_model.basis_gates
        assert gate not in noise_model.noise_instructions
    for gate in simple_error_profile.basis_gates:
        if gate not in exclude_gates:
            assert gate in noise_model.basis_gates
            assert gate in noise_model.noise_instructions


def test_get_noise_model_with_unsupported_error_type():
    """Test get_noise_model_from_profile with unsupported error type (should trigger warning)."""
    # Create a profile with a mock unsupported error type
    profile = QuantumErrorProfile(name="test", description="test profile", errors={})

    # Manually add an error with unsupported type to trigger the warning
    profile.errors[SupportedGate.H] = {"type": "unsupported_error_type", "rate": 0.01, "num_qubits": 1}

    # Reset the qubit gates since we manually modified errors
    profile.one_qubit_gates = ["h"]
    profile.two_qubit_gates = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        noise_model = get_noise_model_from_profile(profile)

        # Should generate a warning about unsupported error type
        assert len(w) == 1
        assert "Unsupported error type" in str(w[0].message)
        assert noise_model is not None
