"""Tests for QDK/Chemistry interop with Qiskit noise models."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT_AER

if QDK_CHEMISTRY_HAS_QISKIT_AER:
    from qiskit_aer.noise import NoiseModel

    from qdk_chemistry.data.noise_models import QuantumErrorProfile
    from qdk_chemistry.plugins.qiskit._interop.noise_model import get_noise_model_from_profile


pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT_AER, reason="Qiskit Aer not available")


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


def test_get_noise_model_with_multiple_gates():
    """Test noise model generation with multiple gates."""
    profile = QuantumErrorProfile(
        name="multi_gate",
        description="test with multiple gates",
        errors={
            "h": {"depolarizing_error": 0.01},
            "x": {"depolarizing_error": 0.01},
            "cx": {"depolarizing_error": 0.02},
            "cz": {"depolarizing_error": 0.02},
        },
    )

    noise_model = get_noise_model_from_profile(profile)

    assert isinstance(noise_model, NoiseModel)
    assert "h" in noise_model.noise_instructions
    assert "x" in noise_model.noise_instructions
    assert "cx" in noise_model.noise_instructions
    assert "cz" in noise_model.noise_instructions


def test_get_noise_model_excludes_multiple_gates():
    """Test noise model generation with multiple excluded gates."""
    profile = QuantumErrorProfile(
        name="exclude_multiple",
        description="test excluding multiple gates",
        errors={
            "h": {"depolarizing_error": 0.01},
            "x": {"depolarizing_error": 0.01},
            "cx": {"depolarizing_error": 0.02},
            "cz": {"depolarizing_error": 0.02},
        },
    )

    exclude_gates = ["x", "cz"]
    noise_model = get_noise_model_from_profile(profile, exclude_gates)

    assert isinstance(noise_model, NoiseModel)
    assert "h" in noise_model.noise_instructions
    assert "cx" in noise_model.noise_instructions
    assert "x" not in noise_model.noise_instructions
    assert "cz" not in noise_model.noise_instructions

    # Excluded gates should still be in basis gates
    for gate in exclude_gates:
        assert gate in noise_model.basis_gates


def test_get_noise_model_empty_profile():
    """Test noise model generation with empty error profile."""
    profile = QuantumErrorProfile()

    noise_model = get_noise_model_from_profile(profile)

    assert isinstance(noise_model, NoiseModel)
    assert len(noise_model.noise_instructions) == 0


def test_get_noise_model_qubit_loss_warns(simple_error_profile_with_qubit_loss):
    """Test that qubit_loss error type emits a warning and is skipped in the Qiskit noise model."""
    with pytest.warns(UserWarning, match="Unsupported error type.*qubit_loss"):
        noise_model = get_noise_model_from_profile(simple_error_profile_with_qubit_loss)

    assert isinstance(noise_model, NoiseModel)
    # Depolarizing errors should still be applied
    assert "h" in noise_model.noise_instructions
    assert "cx" in noise_model.noise_instructions


def test_get_noise_model_mixed_errors_per_gate():
    """Test noise model with a gate having both depolarizing and qubit_loss errors."""
    profile = QuantumErrorProfile(
        name="mixed",
        description="mixed error types per gate",
        errors={
            "cx": {
                "depolarizing_error": 0.02,
                "qubit_loss": 0.005,
            },
        },
    )

    with pytest.warns(UserWarning, match="Unsupported error type.*qubit_loss"):
        noise_model = get_noise_model_from_profile(profile)

    assert isinstance(noise_model, NoiseModel)
    # Depolarizing should still be applied
    assert "cx" in noise_model.noise_instructions


def test_get_noise_model_basis_gates_match_profile():
    """Test that noise model basis gates match the profile's basis gates."""
    profile = QuantumErrorProfile(
        name="basis_test",
        description="test basis gates",
        errors={
            "h": {"depolarizing_error": 0.01},
            "s": {"depolarizing_error": 0.01},
            "t": {"depolarizing_error": 0.01},
            "cx": {"depolarizing_error": 0.02},
        },
    )

    noise_model = get_noise_model_from_profile(profile)

    # All gates from profile should be in noise model basis gates
    for gate in profile.basis_gates:
        assert gate in noise_model.basis_gates
