"""Conversion utilities for QDK Chemistry to Qiskit interoperability.

This module provides functions to convert QDK Chemistry objects into Qiskit-compatible
representations, particularly for quantum circuit simulation and state preparation.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

    from qdk_chemistry import data
    from qdk_chemistry.algorithms.state_preparation._binary_encoding_utils import MatrixCompressionOp

__all__ = ["apply_matrix_compression_ops", "create_statevector_from_wavefunction"]


def create_statevector_from_wavefunction(wavefunction: data.Wavefunction, normalize: bool = True) -> np.ndarray:
    """Create a Qiskit-compatible statevector from a QDK Chemistry wavefunction.

    This function converts a QDK Chemistry wavefunction into a dense statevector
    representation suitable for use with Qiskit quantum circuit simulators.

    For standard chemistry wavefunctions (2 bits per mode), the encoding uses a
    little-endian qubit ordering convention where each spatial orbital is mapped
    to two qubits (one for alpha spin, one for beta spin).

    For generic wavefunctions (1 bit per mode), each mode maps directly to one
    qubit using little-endian ordering.

    Args:
        wavefunction: The wavefunction to convert to statevector representation.
        normalize: Whether to normalize the resulting statevector to unit norm.
            Default is True.

    Returns:
        numpy.ndarray: Dense complex statevector of size 2^num_qubits.
            The dtype is always complex128, even if the wavefunction has real
            coefficients.

    Examples:
        >>> from qiskit.quantum_info import Statevector
        >>> # Assuming we have a wavefunction already
        >>> sv_array = create_statevector_from_wavefunction(wavefunction)
        >>> qiskit_sv = Statevector(sv_array)
        >>> print(f"Statevector dimension: {len(sv_array)}")

    """
    determinants = wavefunction.get_active_determinants()
    config_set = wavefunction.get_configuration_set()
    bits_per_mode = determinants[0].bits_per_mode()

    if bits_per_mode == 1:
        num_qubits = config_set.num_modes()
    else:
        orbitals = wavefunction.get_orbitals()
        indices, _ = orbitals.get_active_space_indices()
        num_qubits = len(indices) * 2

    dim = 1 << num_qubits

    # Initialize statevector as complex array
    statevector = np.zeros(dim, dtype=np.complex128)

    # Get coefficients
    coefficients = wavefunction.get_coefficients()
    coeffs_array = np.array(coefficients)

    # Fill statevector
    for i, det in enumerate(determinants):
        index = _configuration_to_statevector_index(det, num_qubits)
        statevector[index] += coeffs_array[i]

    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(statevector)
        if norm > 1e-15:
            statevector /= norm

    return statevector


def _configuration_to_statevector_index(configuration: data.Configuration, n_bits: int) -> int:
    """Convert a Configuration to its corresponding integer index in the statevector array.

    This function maps an electronic configuration (orbital occupation pattern) to
    its position in a dense statevector representation. The encoding uses little-endian
    qubit ordering where alpha electrons occupy lower-indexed qubits and beta electrons
    occupy higher-indexed qubits.

    The qubit layout for n spatial orbitals is:
        Qubits: [2n-1, 2n-2, ..., n+1, n] [n-1, n-2, ..., 1, 0]
                      beta orbitals              alpha orbitals

    Example:
        Configuration "2ud0" with 4 orbitals maps to:
        - Orbital 0: doubly occupied
        - Orbital 1: alpha
        - Orbital 2: beta
        - Orbital 3: empty

        Qubit layout:
        Qubits: 7 6 5 4 | 3 2 1 0
                beta    | alpha
                3 2 1 0 | 3 2 1 0
                0 1 0 1 | 0 0 1 1

        As binary (little-endian): 01010011 = 64 + 16 + 2 + 1 = 83

    Args:
        configuration (Configuration): The electronic configuration to convert. This object
            encodes the occupation of each orbital (unoccupied, alpha, beta,
            or doubly occupied).
        n_bits (int): Number of bits to read from the configuration
            (num_modes * bits_per_mode).

    Returns:
        int: The statevector index corresponding to this configuration in the
            computational basis.

    """
    # Get bit vector: [alpha_0,...,alpha_{N-1}, beta_0,...,beta_{N-1}]
    bits = configuration.to_bits(n_bits)

    index = 0

    # Little-endian: bit i corresponds to qubit i
    for i, bit in enumerate(bits):
        if bit:
            index |= 1 << i

    return index


def apply_matrix_compression_ops(
    circuit: QuantumCircuit,
    ops: list[MatrixCompressionOp],
) -> None:
    """Apply matrix compression operations to a Qiskit QuantumCircuit.

    Supports all :class:`~qdk_chemistry.algorithms.state_preparation._binary_encoding_utils.MatrixCompressionType`
    operations: X, CX, SWAP, CCX, MCX, SELECT, and SELECT_AND.

    Args:
        circuit: Qiskit QuantumCircuit to append gates to (modified in place).
        ops: List of MatrixCompressionOp to apply.

    """
    from qiskit.circuit.library import XGate  # noqa: PLC0415

    for op in ops:
        name = op.name.upper()
        if name == "X":
            circuit.x(op.qubits[0])
        elif name == "CX":
            circuit.cx(op.qubits[0], op.qubits[1])
        elif name == "SWAP":
            circuit.swap(op.qubits[0], op.qubits[1])
        elif name == "CCX":
            circuit.ccx(op.qubits[0], op.qubits[1], op.qubits[2])
        elif name == "MCX":
            num_controls = len(op.qubits) - 1
            target = op.qubits[num_controls]
            controls = op.qubits[:num_controls]
            gate = XGate().control(num_controls, ctrl_state=op.control_state)
            circuit.append(gate, [*controls, target])
        elif name in ("SELECT", "SELECT_AND"):
            _apply_select_to_circuit(circuit, op)
        else:
            raise ValueError(f"Unsupported MatrixCompressionOp: {op.name}")


def _apply_select_to_circuit(
    circuit: QuantumCircuit,
    op: MatrixCompressionOp,
) -> None:
    """Decompose a SELECT/SELECT_AND operation into multi-controlled X gates.

    Args:
        circuit: Qiskit QuantumCircuit to append gates to.
        op: A SELECT or SELECT_AND MatrixCompressionOp.

    """
    from qiskit.circuit.library import XGate  # noqa: PLC0415

    num_addr = op.control_state
    addr_qubits = op.qubits[:num_addr]
    target_qubits = op.qubits[num_addr:]

    for row_idx, row in enumerate(op.lookup_data):
        if not any(row):
            continue
        for bit_idx, bit in enumerate(row):
            if bit:
                gate = XGate().control(num_addr, ctrl_state=row_idx)
                circuit.append(gate, [*addr_qubits, target_qubits[bit_idx]])
