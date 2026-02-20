"""QDK/Chemistry Quantum Circuits module.

The Circuit class represents a quantum circuit in various formats
(QASM, QIR, Q#) and provides conversion methods between them.

Supported formats and conversions:
- QASM to QIR or Qiskit QuantumCircuit (with Qiskit installed)
- QIR to Qiskit QuantumCircuit (with Qiskit installed)
- Q# circuit object for visualization via QDK widgets
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from collections.abc import Callable
from typing import Any

import h5py
import qsharp._native
import qsharp.openqasm

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger

__all__: list[str] = []


class Circuit(DataClass):
    """Data class for a quantum circuit."""

    # Class attribute for filename validation
    _data_type_name = "circuit"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    # Use keyword arguments to be future-proof
    def __init__(
        self,
        qasm: str | None = None,
        qir: qsharp._qsharp.QirInputData | str | None = None,
        qsharp: qsharp._native.Circuit | None = None,
        qsharp_op: Callable[..., Any] | None = None,
        encoding: str | None = None,
    ) -> None:
        """Initialize a Circuit.

        Args:
            qasm: The quantum circuit in QASM format. Defaults to None.
            qir: The QIR representation of the quantum circuit. Defaults to None.
            qsharp: The quantum circuit as a Q# program. Defaults to None.
            qsharp_op: The Q# operation associated with the circuit. Defaults to None.
            encoding: The fermion-to-qubit encoding assumed by this circuit.
                Valid values include "jordan-wigner", "bravyi-kitaev", "parity", or None.
                Defaults to None.

        """
        Logger.trace_entering()
        self.qasm = qasm
        self.qir = qir
        self.qsharp = qsharp
        self._qsharp_op = qsharp_op
        self.encoding = encoding

        # Check that a representation of the quantum circuit is given by the keyword arguments
        if not any([self.qasm, self.qsharp, self.qir]):
            raise RuntimeError("No representation of the quantum circuit is set.")

        # Make instance immutable after construction (handled by base class)
        super().__init__()

    def get_qasm(self) -> str:
        """Get the quantum circuit in QASM format.

        Returns:
            str: The quantum circuit in QASM format.

        """
        if self.qasm is None:
            raise RuntimeError("The quantum circuit in QASM format is not set.")

        return self.qasm

    def get_qir(self) -> qsharp._qsharp.QirInputData | str:
        """Get QIR representation of the quantum circuit.

        Returns:
            The QIR representation of the quantum circuit.

        """
        if self.qir:
            return self.qir
        if self.qasm:
            return qsharp.openqasm.compile(self.qasm)

        raise RuntimeError("The QIR representation of the quantum circuit is not set.")

    def get_qsharp_circuit(self) -> qsharp._native.Circuit:
        """Parse a Circuit object into a qsharp Circuit object.

        Returns:
            qsharp._native.Circuit: A qsharp Circuit object representing the trimmed circuit.

        """
        Logger.trace_entering()
        if self.qsharp:
            return self.qsharp
        if self.qasm:
            return qsharp.openqasm.circuit(self.qasm)

        raise RuntimeError("The quantum circuit is not set in a qsharp format.")

    def get_qiskit_circuit(self):
        """Convert the Circuit to a Qiskit QuantumCircuit.

        Raises:
            RuntimeError: If Qiskit is not available or if the circuit cannot be converted.

        """
        Logger.trace_entering()
        try:
            from qiskit import qasm3  # noqa: PLC0415

            from qdk_chemistry.plugins.qiskit._interop.qir import qir_ir_to_qiskit  # noqa: PLC0415
        except ImportError as err:
            raise RuntimeError("Qiskit is not available. Cannot convert circuit to Qiskit format.") from err

        if self.qir:
            return qir_ir_to_qiskit(str(self.qir))
        if self.qasm:
            return qasm3.loads(self.qasm)

        raise RuntimeError("The quantum circuit cannot be converted to Qiskit format.")

    # DataClass interface implementation
    def get_summary(self) -> str:
        """Get a human-readable summary of the Circuit.

        Returns:
            str: Summary string describing the quantum circuit.

        """
        lines = ["Circuit"]
        if self.qasm is not None:
            lines.append(f"  QASM string: {self.qasm}")
        if self.encoding is not None:
            lines.append(f"  Encoding: {self.encoding}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Convert the Circuit to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the quantum circuit.

        """
        data: dict[str, Any] = {}
        if self.qasm is not None:
            data["qasm"] = self.qasm
        if self.qir is not None:
            data["qir"] = str(self.qir)
        if self.encoding is not None:
            data["encoding"] = self.encoding
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the Circuit to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write the quantum circuit to.

        """
        self._add_hdf5_version(group)
        if self.qasm is not None:
            group.attrs["qasm"] = self.qasm
        if self.qir is not None:
            group.attrs["qir"] = str(self.qir)
        if self.encoding is not None:
            group.attrs["encoding"] = self.encoding

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "Circuit":
        """Create a Circuit from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized data.

        Returns:
            Circuit: New instance of the Circuit.

        Raises:
            RuntimeError: If version field is missing or incompatible.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls(
            qasm=json_data.get("qasm"),
            qir=json_data.get("qir"),
            encoding=json_data.get("encoding"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "Circuit":
        """Load a Circuit from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to read data from.

        Returns:
            Circuit: New instance of the Circuit.

        Raises:
            RuntimeError: If version attribute is missing or incompatible.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        encoding = group.attrs.get("encoding")
        # Decode encoding if it's stored as bytes (HDF5 behavior can vary)
        if encoding is not None and isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        return cls(
            qasm=group.attrs.get("qasm"),
            qir=group.attrs.get("qir"),
            encoding=encoding,
        )
