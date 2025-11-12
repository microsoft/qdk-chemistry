"""Base class for state preparation in qdk.chemistry.

This module provides classes that wrap different state preparation algorithms
with a common interface. Each class can be initialized with a problem-specific
wavefunction and other parameters, then used to create quantum circuits.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging
from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np
from qiskit import QuantumCircuit

from qdk.chemistry.data import Wavefunction
from qdk.chemistry.utils.bitstring import separate_alpha_beta_to_binary_string

_LOGGER = logging.getLogger(__name__)


class StatePrepAlgorithm(StrEnum):
    """Enumeration for state preparation algorithm types."""

    REGULAR_ISOMETRY = "regular_isometry"
    SPARSE_ISOMETRY_GF2X = "sparse_isometry_gf2x"


class StatePrep(ABC):
    """Base class for state preparation algorithms.

    This abstract class defines the common interface for all state preparation implementations.
    Subclasses must implement the create_circuit_qasm method.
    """

    algorithm: StatePrepAlgorithm | None = None  # To be overridden by subclasses

    def __init__(self, wavefunction: Wavefunction, max_dets: int | None = None, amplitude_threshold: float = 0.0):
        """Initialize the state preparation class with a wavefunction instance.

        Args:
            wavefunction: Wavefunction to prepare state from
            max_dets: Maximum number of determinants to include in the isometry (default: include all)
            amplitude_threshold: Amplitude threshold on absolute value of coefficients for including
            determinants (default: 0.0)

        """
        self.wavefunction = wavefunction
        orbitals = wavefunction.get_orbitals()
        alpha_active_orbital_indices, beta_active_orbital_indices = orbitals.get_active_space_indices()
        if len(alpha_active_orbital_indices) != len(beta_active_orbital_indices):
            raise ValueError(
                f"Active space contains {len(alpha_active_orbital_indices)} alpha orbitals and "
                f"{len(beta_active_orbital_indices)} beta orbitals. Asymmetric active spaces for alpha and beta "
                "orbitals are not supported for state preparation."
            )
        self.num_orbitals = len(alpha_active_orbital_indices)
        bitstrings = []
        coeffs = []
        for det in wavefunction.get_active_determinants():
            coeffs.append(wavefunction.get_coefficient(det))
            alpha_str, beta_str = separate_alpha_beta_to_binary_string(det.to_string()[: self.num_orbitals])
            bitstring = beta_str[::-1] + alpha_str[::-1]  # Convert to little-endian format to match qiskit convention
            bitstrings.append(bitstring)
        self._bitstrings = bitstrings
        self._coefficients = coeffs

        if max_dets is not None and max_dets > len(wavefunction.get_active_determinants()):
            raise ValueError(
                f"max_dets ({max_dets}) cannot be greater than the "
                f"number of determinants ({len(wavefunction.get_active_determinants())})"
            )
        self.max_dets = max_dets or len(wavefunction.get_active_determinants())
        self.amplitude_threshold = amplitude_threshold
        # Cached filtered terms
        self._is_filtered = False
        self._filtered_coeffs: np.ndarray | None = None
        self._filtered_bitstrings: list[str] | None = None

    @classmethod
    def from_algorithm(
        cls,
        state_prep_algorithm: StatePrepAlgorithm,
        wavefunction: Wavefunction,
        max_dets: int | None = None,
        amplitude_threshold: float = 0.0,
        **kwargs,
    ) -> "StatePrep":
        """Factory method to create a StatePrep object from the specified algorithm.

        Args:
            state_prep_algorithm: Algorithm used for state preparation
            wavefunction: Wavefunction to prepare state for
            max_dets: Maximum number of determinants to include (default: include all)
            amplitude_threshold: Amplitude threshold for including determinants
                (default: 0.0)
            **kwargs: Additional parameters to pass to the state preparation constructor.

        Additional kwargs supported by specific algorithms:

        * ``SparseIsometryGF2XStatePrep`` : accepts optional arguments controlling output saving for
            GF2+X elimination summaries and intermediate matrices:

            * ``save_outputs``: whether to save output files (default: False),
            * ``summary_filename``: filename for GF2+X elimination summary
                                    (default: "sparse_isometry_gf2x_summary.txt"),
            * ``output_dir``: directory for output files (default: "sparse_isometry_gf2x_output"),

        * ``RegularIsometryStatePrep``: does not accept any additional arguments.

        Returns:
            An instance of a ``StatePrep`` subclass.

        Raises:
            ValueError: If the prep_algorithm is not implemented

        """
        for subclass in cls.__subclasses__():
            if subclass.algorithm == state_prep_algorithm:
                return subclass(
                    wavefunction=wavefunction,
                    max_dets=max_dets,
                    amplitude_threshold=amplitude_threshold,
                    **kwargs,
                )

        raise ValueError(f"State preparation algorithm {state_prep_algorithm} not implemented")

    def _filter_terms(self, renormalize: bool = True) -> tuple[np.ndarray, list[str]]:
        """Filter coefficients and bitstrings based on threshold and max_dets.

        This is a common filtering method used by all state preparation algorithms
        to reduce the number of determinants based on amplitude threshold and/or
        maximum determinant count.

        Args:
            renormalize: Whether to renormalize the filtered coefficients (default: True)

        Returns:
            Tuple of ``(filtered_coeffs, filtered_bitstrings)``

        Raises:
            ValueError: If no determinants remain after filtering or max_dets is invalid

        """
        # Create a list of (abs_coeff, coeff, bitstring) tuples for sorting by absolute value in descending order
        dets = [
            (abs(coeff), coeff, bitstring)
            for coeff, bitstring in zip(self._coefficients, self._bitstrings, strict=False)
        ]
        dets.sort(key=lambda x: x[0], reverse=True)

        # Filter by amplitude threshold
        dets = [
            (abs_coeff, coeff, bitstring)
            for abs_coeff, coeff, bitstring in dets
            if abs_coeff >= self.amplitude_threshold
        ]

        # Apply max_dets limit
        if self.max_dets is not None:
            if not isinstance(self.max_dets, int) or self.max_dets <= 0:
                raise ValueError("max_dets must be a positive integer or None")
            original_count = len(dets)
            if len(dets) > self.max_dets:
                dets = dets[: self.max_dets]
                _LOGGER.info(f"Limiting determinants from {original_count} to {self.max_dets} (max_dets applied)")

        # Check if any determinants remain
        if not dets:
            raise ValueError("No determinants remain after filtering")

        # Extract coefficients and bitstrings
        _, filtered_coeffs, filtered_bitstrings = zip(*dets, strict=False)

        # Convert to numpy array for coefficients
        filtered_coeffs = np.array(filtered_coeffs, dtype=complex)

        # Renormalize if requested
        if renormalize:
            norm = np.linalg.norm(filtered_coeffs)
            if norm > 0:
                filtered_coeffs /= norm

        self._filtered_coeffs = filtered_coeffs
        self._filtered_bitstrings = list(filtered_bitstrings)
        self._is_filtered = True
        return self._filtered_coeffs, self._filtered_bitstrings

    @abstractmethod
    def create_circuit_qasm(self) -> str:
        """Create a quantum circuit that implements the state preparation.

        Returns:
            A quantum circuit in QASM string format.

        """


def prepare_single_reference_state(bitstring: str) -> QuantumCircuit:
    r"""Prepare a single reference state on a quantum circuit based on a bitstring.

    Args:
        bitstring: Binary string representing the occupation of qubits.
                  '1' means apply X gate, '0' means leave in |0⟩ state.

    Returns:
        ``QuantumCircuit`` with the prepared single reference state

    Example:
        bitstring = "1010" creates a circuit with X gates on qubits 1 and 3:

        * :math:`\left| 0 \right\rangle \rightarrow I \rightarrow \left| 0 \right\rangle`
          (qubit 0, corresponds to rightmost bit '0')
        * :math:`\left| 0 \right\rangle \rightarrow X \rightarrow \left| 1 \right\rangle`
          (qubit 1, corresponds to bit '1')
        * :math:`\left| 0 \right\rangle \rightarrow I \rightarrow \left| 0 \right\rangle`
          (qubit 2, corresponds to bit '0')
        * :math:`\left| 0 \right\rangle \rightarrow X \rightarrow \left| 1 \right\rangle`
          (qubit 3, corresponds to leftmost bit '1')

    """
    # Input validation
    if not bitstring:
        raise ValueError("Bitstring cannot be empty")

    if not all(bit in "01" for bit in bitstring):
        raise ValueError("Bitstring must contain only '0' and '1' characters")

    num_qubits = len(bitstring)
    circuit = QuantumCircuit(num_qubits, name=f"SingleRef_{bitstring}")

    # Apply X gates for positions with '1'
    # Note: bitstring is in little-endian format (rightmost bit = qubit 0)
    for i, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            circuit.x(i)

    return circuit
