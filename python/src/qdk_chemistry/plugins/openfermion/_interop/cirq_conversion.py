"""Cirq circuit conversion utilities for QDK/Chemistry-OpenFermion interoperability.

This module provides functions to convert between Cirq circuits and QDK/Chemistry
``Circuit`` objects (which wrap OpenQASM 3 strings). Conversion is performed by
generating OpenQASM 2.0 from Cirq and upgrading the syntax to OpenQASM 3.

- ``cirq_circuit_to_qasm3``: Convert a Cirq ``Circuit`` to an OpenQASM 3 string.
- ``cirq_circuit_to_qdk_circuit``: Convert a Cirq ``Circuit`` to a QDK ``Circuit``.
- ``qdk_circuit_to_cirq_circuit``: Convert a QDK ``Circuit`` to a Cirq ``Circuit`` (via QASM 2 round-trip).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import re

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm

from qdk_chemistry.data import Circuit
from qdk_chemistry.utils import Logger

__all__ = [
    "cirq_circuit_to_qasm3",
    "cirq_circuit_to_qdk_circuit",
    "qdk_circuit_to_cirq_circuit",
]


def _qasm2_to_qasm3(qasm2: str) -> str:
    """Convert an OpenQASM 2.0 string to OpenQASM 3 syntax.

    Performs the following syntactic transformations:

    1. Replace the header (``OPENQASM 2.0; include "qelib1.inc";``) with
       ``OPENQASM 3.0; include "stdgates.inc";``.
    2. Convert ``qreg name[n];`` → ``qubit[n] name;``.
    3. Convert ``creg name[n];`` → ``bit[n] name;``.
    4. Convert ``measure q[i] -> c[j];`` → ``c[j] = measure q[i];``.
    5. Remove ``barrier`` statements (not part of OpenQASM 3 core).

    Args:
        qasm2: The OpenQASM 2.0 string.

    Returns:
        str: The equivalent OpenQASM 3 string.

    """
    lines = qasm2.strip().split("\n")
    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            output_lines.append("")
            continue

        # Replace OPENQASM version header
        if stripped.startswith("OPENQASM 2"):
            output_lines.append("OPENQASM 3.0;")
            continue

        # Replace include
        if stripped == 'include "qelib1.inc";':
            output_lines.append('include "stdgates.inc";')
            continue

        # Convert qreg → qubit
        m = re.match(r"qreg\s+(\w+)\[(\d+)\]\s*;", stripped)
        if m:
            name, size = m.group(1), m.group(2)
            output_lines.append(f"qubit[{size}] {name};")
            continue

        # Convert creg → bit
        m = re.match(r"creg\s+(\w+)\[(\d+)\]\s*;", stripped)
        if m:
            name, size = m.group(1), m.group(2)
            output_lines.append(f"bit[{size}] {name};")
            continue

        # Convert measure q[i] -> c[j]; → c[j] = measure q[i];
        m = re.match(r"measure\s+(.+?)\s*->\s*(.+?)\s*;", stripped)
        if m:
            qubit_ref, bit_ref = m.group(1), m.group(2)
            output_lines.append(f"{bit_ref} = measure {qubit_ref};")
            continue

        # Remove barrier statements
        if stripped.startswith("barrier"):
            continue

        # Keep everything else unchanged (gate applications, etc.)
        output_lines.append(line)

    return "\n".join(output_lines)


def cirq_circuit_to_qasm3(
    circuit: cirq.Circuit,
) -> str:
    """Convert a Cirq ``Circuit`` to an OpenQASM 3 string.

    Generates OpenQASM 2.0 from Cirq using ``cirq.qasm(circuit)`` and then
    upgrades the syntax to OpenQASM 3.

    Args:
        circuit: The Cirq circuit to convert.

    Returns:
        str: The circuit as an OpenQASM 3 string.

    Raises:
        ValueError: If the Cirq circuit cannot be exported to QASM.

    Examples:
        >>> import cirq
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1)])
        >>> qasm3_str = cirq_circuit_to_qasm3(circuit)

    """
    Logger.trace_entering()

    try:
        qasm2_str = cirq.qasm(circuit)
    except ValueError as e:
        raise ValueError(f"Cannot convert Cirq circuit to QASM: {e}") from e

    return _qasm2_to_qasm3(qasm2_str)


def cirq_circuit_to_qdk_circuit(
    circuit: cirq.Circuit,
    encoding: str | None = None,
) -> Circuit:
    """Convert a Cirq circuit to a QDK/Chemistry ``Circuit`` object.

    Args:
        circuit: The Cirq circuit to convert.
        encoding: Optional encoding label (e.g., ``"jordan-wigner"``) to attach to the resulting ``Circuit``.

    Returns:
        Circuit: A QDK/Chemistry ``Circuit`` wrapping the OpenQASM 3 string.

    Examples:
        >>> import cirq
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> cirq_circ = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])
        >>> qdk_circuit = cirq_circuit_to_qdk_circuit(cirq_circ, encoding="jordan-wigner")

    """
    Logger.trace_entering()
    qasm3_str = cirq_circuit_to_qasm3(circuit)
    return Circuit(qasm=qasm3_str, encoding=encoding)


def qdk_circuit_to_cirq_circuit(
    circuit: Circuit,
) -> cirq.Circuit:
    """Convert a QDK/Chemistry ``Circuit`` to a Cirq ``Circuit``.

    This performs a round-trip through OpenQASM 2 using Cirq's QASM import.
    The QDK ``Circuit`` stores OpenQASM 3 which is first downgraded to
    OpenQASM 2 syntax before being parsed by Cirq.

    Note:
        This conversion may lose some gate-level fidelity if the circuit uses
        OpenQASM 3 features not representable in OpenQASM 2 / Cirq.

    Args:
        circuit: The QDK/Chemistry ``Circuit`` to convert.

    Returns:
        cirq.Circuit: The equivalent Cirq circuit.

    Raises:
        ValueError: If the QASM cannot be parsed by Cirq.

    Examples:
        >>> from qdk_chemistry.data import Circuit
        >>> qdk_circ = Circuit(qasm='OPENQASM 3.0; ...')
        >>> cirq_circ = qdk_circuit_to_cirq_circuit(qdk_circ)

    """
    Logger.trace_entering()
    qasm3_str = circuit.get_qasm()

    # Downgrade QASM 3 → QASM 2 for Cirq import
    qasm2_str = _qasm3_to_qasm2(qasm3_str)

    try:
        return circuit_from_qasm(qasm2_str)
    except ImportError as e:
        raise ImportError("Cirq QASM import requires the 'ply' package. Install it with: pip install ply") from e
    except ValueError as e:
        raise ValueError(f"Cannot parse QASM into a Cirq circuit: {e}") from e


def _qasm3_to_qasm2(qasm3: str) -> str:
    """Downgrade OpenQASM 3 syntax to OpenQASM 2.0 for Cirq import.

    Performs the inverse of ``_qasm2_to_qasm3``:

    1. Replace ``OPENQASM 3.0;`` → ``OPENQASM 2.0;``.
    2. Replace ``include "stdgates.inc";`` → ``include "qelib1.inc";``.
    3. Convert ``qubit[n] name;`` → ``qreg name[n];``.
    4. Convert ``bit[n] name;`` → ``creg name[n];``.
    5. Convert ``c[j] = measure q[i];`` → ``measure q[i] -> c[j];``.

    Args:
        qasm3: The OpenQASM 3 string.

    Returns:
        str: The equivalent OpenQASM 2.0 string.

    """
    lines = qasm3.strip().split("\n")
    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            output_lines.append("")
            continue

        if stripped.startswith("OPENQASM 3"):
            output_lines.append("OPENQASM 2.0;")
            continue

        if stripped == 'include "stdgates.inc";':
            output_lines.append('include "qelib1.inc";')
            continue

        # Convert qubit[n] name; → qreg name[n];
        m = re.match(r"qubit\[(\d+)\]\s+(\w+)\s*;", stripped)
        if m:
            size, name = m.group(1), m.group(2)
            output_lines.append(f"qreg {name}[{size}];")
            continue

        # Convert bit[n] name; → creg name[n];
        m = re.match(r"bit\[(\d+)\]\s+(\w+)\s*;", stripped)
        if m:
            size, name = m.group(1), m.group(2)
            output_lines.append(f"creg {name}[{size}];")
            continue

        # Convert c[j] = measure q[i]; → measure q[i] -> c[j];
        m = re.match(r"(.+?)\s*=\s*measure\s+(.+?)\s*;", stripped)
        if m:
            bit_ref, qubit_ref = m.group(1), m.group(2)
            output_lines.append(f"measure {qubit_ref} -> {bit_ref};")
            continue

        output_lines.append(line)

    return "\n".join(output_lines)
