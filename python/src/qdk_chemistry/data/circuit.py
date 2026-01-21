"""QDK/Chemistry Quantum Circuits module.

Includes utilities for visualizing circuits with QDK widgets.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py
import qsharp._native
import qsharp.openqasm
from qiskit import QuantumCircuit, qasm3

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger

__all__: list[str] = []


class Circuit(DataClass):
    """Data class for a quantum circuit.

    Attributes:
        qasm (str): The quantum circuit in QASM format.
        encoding (str | None): The fermion-to-qubit encoding assumed by this circuit (e.g., "jordan-wigner").
            If None, no specific encoding is assumed.

    """

    # Class attribute for filename validation
    _data_type_name = "circuit"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    # Use keyword arguments to be future-proof
    def __init__(
        self,
        qasm: str | None = None,
        encoding: str | None = None,
    ) -> None:
        """Initialize a Circuit.

        Args:
            qasm (str | None): The quantum circuit in QASM format. Defaults to None.
            encoding (str | None): The fermion-to-qubit encoding assumed by this circuit.
                Valid values include "jordan-wigner", "bravyi-kitaev", "parity", or None.
                Defaults to None.

        """
        Logger.trace_entering()
        self.qasm = qasm
        self.encoding = encoding

        # Check that a representation of the quantum circuit is given by the keyword arguments
        if self.qasm is None:
            raise RuntimeError("The quantum circuit in QASM format is not set.")

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

    # Utilities for visualizing circuits with QDK widgets.
    def get_qsharp(
        self, remove_idle_qubits: bool = True, remove_classical_qubits: bool = True
    ) -> qsharp._native.Circuit:
        """Parse a Circuit object into a qsharp Circuit object with trimming options.

        Args:
            remove_idle_qubits (bool): If True, remove qubits that are idle (no gates applied).
            remove_classical_qubits (bool): If True, remove qubits with gates but deterministic bitstring outputs (0|1).

        Returns:
            qsharp._native.Circuit: A qsharp Circuit object representing the trimmed circuit.

        """
        Logger.trace_entering()
        circuit_to_visualize = self._trim_circuit(remove_idle_qubits, remove_classical_qubits)

        return qsharp.openqasm.circuit(circuit_to_visualize)

    def _trim_circuit(self, remove_idle_qubits: bool = True, remove_classical_qubits: bool = True) -> str:
        """Trim the quantum circuit by removing idle and classical qubits.

        Args:
            remove_idle_qubits (bool): If True, remove qubits that are idle (no gates applied).
            remove_classical_qubits (bool): If True, remove qubits with gates but deterministic bitstring outputs (0|1).

        Returns:
            str: A trimmed circuit in QASM format.

        """
        Logger.trace_entering()
        from qdk_chemistry.plugins.qiskit._interop.circuit import analyze_qubit_status  # noqa: PLC0415

        if self.qasm is None:
            raise NotImplementedError("Quantum circuit trimming is only implemented for QASM circuits.")
        try:
            qc = qasm3.loads(self.qasm)
        except Exception as e:
            raise ValueError("Invalid QASM3 syntax provided.") from e

        status = analyze_qubit_status(qc)
        remove_status = []
        if remove_idle_qubits:
            remove_status.append("idle")
        if remove_classical_qubits:
            remove_status.append("classical")
            Logger.info(
                "Removing classical qubits will also remove any control operations sourced from them "
                "and measurements involving them."
            )

        kept_qubit_indices = [q for q, role in status.items() if role not in remove_status]
        if not kept_qubit_indices:
            raise ValueError("No qubits remain after filtering. Try relaxing filters.")

        # Check measurement operations
        kept_measurements: list[tuple[int, int]] = []
        for inst in qc.data:
            if inst.operation.name == "measure":
                qidx = qc.find_bit(inst.qubits[0]).index
                cidx = qc.find_bit(inst.clbits[0]).index
                if qidx in kept_qubit_indices:
                    kept_measurements.append((qidx, cidx))

        if remove_classical_qubits:
            kept_clbit_indices = sorted({cidx for _, cidx in kept_measurements})
        else:
            kept_clbit_indices = list(range(len(qc.clbits)))

        if not kept_clbit_indices and len(qc.clbits) > 0:
            Logger.warn("All measurements are dropped, no classical bits remain.")

        new_qc = QuantumCircuit(len(kept_qubit_indices), len(kept_clbit_indices))
        qubit_map = {qc.qubits[i]: new_qc.qubits[new_i] for new_i, i in enumerate(kept_qubit_indices)}
        clbit_map = {qc.clbits[i]: new_qc.clbits[new_i] for new_i, i in enumerate(kept_clbit_indices)}

        for inst in qc.data:
            qargs = [qubit_map[q] for q in inst.qubits if q in qubit_map]
            cargs = [clbit_map[c] for c in inst.clbits if c in clbit_map]
            if len(qargs) != len(inst.qubits) or len(cargs) != len(inst.clbits):
                continue
            new_qc.append(inst.operation, qargs, cargs)

        return qasm3.dumps(new_qc)

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
            encoding=encoding,
        )
