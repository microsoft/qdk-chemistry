"""Tests for QDK/Chemistry interop with Qiskit noise models."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

try:
    import qiskit  # noqa: F401
    import qiskit_aer  # noqa: F401
    import qiskit_nature  # noqa: F401

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


if QISKIT_AVAILABLE:
    from qiskit_aer.noise import NoiseModel

    from qdk_chemistry.data.noise_models import QuantumErrorProfile
    from qdk_chemistry.plugins.qiskit._interop.noise_model import get_noise_model_from_profile

import pytest

pytestmark = pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit dependencies not available")


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
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "x": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
            "cz": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
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
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "x": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
            "cz": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
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


def test_get_noise_model_basis_gates_match_profile():
    """Test that noise model basis gates match the profile's basis gates."""
    profile = QuantumErrorProfile(
        name="basis_test",
        description="test basis gates",
        errors={
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "s": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "t": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
        },
    )

    noise_model = get_noise_model_from_profile(profile)

    # All gates from profile should be in noise model basis gates
    for gate in profile.basis_gates:
        assert gate in noise_model.basis_gates
