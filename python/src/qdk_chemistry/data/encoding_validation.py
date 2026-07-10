"""Deprecated utilities for validating fermion-to-qubit encoding compatibility.

.. deprecated:: 2.0
    Encoding compatibility is superseded by the explicit :class:`~qdk_chemistry.data.MajoranaMapping`
    fermion-to-qubit workflow. ``EncodingMismatchError`` and
    ``validate_encoding_compatibility`` remain as deprecated facades for backward
    compatibility and will be removed in a future release.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdk_chemistry.data.circuit import Circuit
    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

__all__ = ["EncodingMismatchError", "validate_encoding_compatibility"]


class EncodingMismatchError(ValueError):
    """Exception raised when Circuit and QubitHamiltonian have incompatible encodings.

    .. deprecated:: 2.0
        Use the :class:`~qdk_chemistry.data.MajoranaMapping` fermion-to-qubit workflow instead.
    """


def validate_encoding_compatibility(circuit: Circuit, hamiltonian: QubitHamiltonian) -> None:
    """Validate that a Circuit and QubitHamiltonian use compatible encodings.

    This function checks that both the circuit and Hamiltonian have matching encodings.
    Both must have their encoding specified (not None), and the encodings must match.

    .. deprecated:: 2.0
        Use the :class:`~qdk_chemistry.data.MajoranaMapping` fermion-to-qubit workflow instead.

    Args:
        circuit: The quantum circuit with encoding metadata.
        hamiltonian: The qubit Hamiltonian with encoding metadata.

    Raises:
        EncodingMismatchError: If the circuit or Hamiltonian encoding is None, or if the encodings don't match.

    Examples:
        >>> circuit = Circuit(qasm="...", encoding="jordan-wigner")
        >>> hamiltonian = QubitHamiltonian(..., encoding="jordan-wigner")
        >>> validate_encoding_compatibility(circuit, hamiltonian)  # OK

    """
    warnings.warn(
        "validate_encoding_compatibility is deprecated and will be removed in a future release; "
        "use the MajoranaMapping fermion-to-qubit workflow instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    circuit_encoding = circuit.encoding
    hamiltonian_encoding = hamiltonian.encoding

    # Require that both encodings are specified
    if circuit_encoding is None:
        raise EncodingMismatchError(
            "Circuit encoding is not specified. All circuits must have an encoding metadata "
            "to ensure compatibility with qubit Hamiltonians."
        )

    if hamiltonian_encoding is None:
        raise EncodingMismatchError(
            "QubitHamiltonian encoding is not specified. All qubit Hamiltonians must have an "
            "encoding metadata to ensure compatibility with circuits."
        )

    # Both encodings are specified - they must match
    if circuit_encoding != hamiltonian_encoding:
        raise EncodingMismatchError(
            f"Encoding mismatch detected: Circuit uses '{circuit_encoding}' encoding, "
            f"but QubitHamiltonian uses '{hamiltonian_encoding}' encoding. "
            f"These encodings are incompatible and will lead to incorrect results. "
            f"Please ensure both the circuit and Hamiltonian use the same fermion-to-qubit encoding."
        )
