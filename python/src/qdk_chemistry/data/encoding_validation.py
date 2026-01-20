"""Utilities for validating fermion-to-qubit encoding compatibility.

This module provides functions to validate that Circuit and QubitHamiltonian
instances use compatible fermion-to-qubit encodings.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdk_chemistry.data.circuit import Circuit
    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

__all__ = ["EncodingMismatchError", "validate_encoding_compatibility"]


class EncodingMismatchError(ValueError):
    """Exception raised when Circuit and QubitHamiltonian have incompatible encodings."""


def validate_encoding_compatibility(circuit: Circuit, hamiltonian: QubitHamiltonian) -> None:
    """Validate that a Circuit and QubitHamiltonian use compatible encodings.

    This function checks that both the circuit and Hamiltonian either have matching
    encodings or at least one has no encoding specified (encoding=None). If both
    have encodings specified and they don't match, an EncodingMismatchError is raised.

    Args:
        circuit (Circuit): The quantum circuit with potential encoding metadata.
        hamiltonian (QubitHamiltonian): The qubit Hamiltonian with potential encoding metadata.

    Raises:
        EncodingMismatchError: If the circuit and Hamiltonian have incompatible encodings.

    Examples:
        >>> circuit = Circuit(qasm="...", encoding="jordan-wigner")
        >>> hamiltonian = QubitHamiltonian(..., encoding="jordan-wigner")
        >>> validate_encoding_compatibility(circuit, hamiltonian)  # OK
        >>> hamiltonian_bk = QubitHamiltonian(..., encoding="bravyi-kitaev")
        >>> validate_encoding_compatibility(circuit, hamiltonian_bk)  # Raises EncodingMismatchError

    """
    circuit_encoding = circuit.encoding
    hamiltonian_encoding = hamiltonian.encoding

    # If either encoding is None, we can't verify but we don't raise an error
    # This allows backward compatibility with existing code
    if circuit_encoding is None or hamiltonian_encoding is None:
        return

    # Both encodings are specified - they must match
    if circuit_encoding != hamiltonian_encoding:
        raise EncodingMismatchError(
            f"Encoding mismatch detected: Circuit uses '{circuit_encoding}' encoding, "
            f"but QubitHamiltonian uses '{hamiltonian_encoding}' encoding. "
            f"These encodings are incompatible and will lead to incorrect results. "
            f"Please ensure both the circuit and Hamiltonian use the same fermion-to-qubit encoding."
        )
