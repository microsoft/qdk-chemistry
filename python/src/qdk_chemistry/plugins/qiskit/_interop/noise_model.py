"""QDK/Chemistry interoperability for Qiskit noise models."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import warnings

from qiskit_aer.noise import NoiseModel, depolarizing_error

from qdk_chemistry.data.noise_models import QuantumErrorProfile, SupportedErrorTypes
from qdk_chemistry.utils import Logger

__all__ = ["get_noise_model_from_profile"]


def get_noise_model_from_profile(
    quantum_error_profile: QuantumErrorProfile, exclude_gates: list | None = None
) -> NoiseModel:
    """Convert profile to noise model.

    Args:
        quantum_error_profile: Quantum error profile
        exclude_gates: Optional basis gates to use for noise model

    Returns:
        Configured noise model based on profile

    """
    Logger.trace_entering()
    noise_model = NoiseModel(basis_gates=quantum_error_profile.basis_gates)

    for gate, error_dict in quantum_error_profile.errors.items():
        if exclude_gates is not None and str(gate) in exclude_gates:
            continue
        if error_dict["type"] == SupportedErrorTypes.DEPOLARIZING_ERROR:
            noise_model.add_all_qubit_quantum_error(
                depolarizing_error(error_dict["rate"], error_dict["num_qubits"]),
                [str(gate)],  # Convert gate to string for Qiskit
            )
        else:
            warnings.warn(
                f"Unsupported error type: {error_dict['type']} for gate {gate}. "
                "The error contribution will be ignored in this error model.",
                category=UserWarning,
                stacklevel=2,
            )
    return noise_model
