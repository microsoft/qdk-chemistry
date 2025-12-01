"""QDK/Chemistry data class for quantum circuits.

Includes utilities for visualizing circuits with QDK widgets.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging

import h5py
import qsharp._native
import qsharp.openqasm
from qiskit import QuantumCircuit, qasm3

from qdk_chemistry.data import DataClass

_LOGGER = logging.getLogger(__name__)

__all__: list[str] = []


class Circuit(DataClass):
    """Data class for quantum circuits.

    Attributes:
        circuit_qasm (str): The quantum circuit in QASM format.

    """

    # Class attribute for filename validation
    _data_type_name = "circuit"

    # Use keyword arguments to be future-proof
    def __init__(
        self,
        circuit_qasm: str | None = None,
    ) -> None:
        """Initialize a Circuit.

        Args:
            circuit_qasm (str | None): The quantum circuit in QASM format. Defaults to None.

        """
        self.circuit_qasm = circuit_qasm

        # Check that a representation of the quantum circuit is given by the keyword arguments
        if self.circuit_qasm is None:
            raise RuntimeError("The quantum circuit in QASM format is not set.")

        # Make instance immutable after construction (handled by base class)
        super().__init__()

    def get_circuit_qasm(self) -> str:
        """Get the quantum circuit in QASM format.

        Returns:
            str: The quantum circuit in QASM format.

        """
        if self.circuit_qasm is None:
            raise RuntimeError("The quantum circuit in QASM format is not set.")

        return self.circuit_qasm

    # Utilities for visualizing circuits with QDK widgets.
    def qasm_to_qdk_circuit(
        self, remove_idle_qubits: bool = True, remove_classical_qubits: bool = True
    ) -> qsharp._native.Circuit:
        """Parse a QASM circuit into a QDK Circuit object with trimming options.

        Args:
            remove_idle_qubits: If True, remove qubits that are idle (no gates applied).
            remove_classical_qubits: If True, remove qubits with gates but bitstring outputs are deterministic (0 or 1).

        Returns:
            A QDK Circuit object representing the trimmed circuit.

        """
        circuit_to_visualize = self._trim_circuit(remove_idle_qubits, remove_classical_qubits)

        return qsharp.openqasm.circuit(circuit_to_visualize)

    def _trim_circuit(self, remove_idle_qubits: bool = True, remove_classical_qubits: bool = True) -> str:
        """Trim the quantum circuit by removing idle and classical qubits.

        Args:
            remove_idle_qubits: If True, remove qubits that are idle (no gates applied).
            remove_classical_qubits: If True, remove qubits with gates but bitstring outputs are deterministic (0 or 1).

        Returns:
            A trimmed circuit in QASM format.

        """
        from qdk_chemistry.plugins.qiskit._interop.circuit import analyze_qubit_status  # noqa: PLC0415

        if self.circuit_qasm is None:
            raise NotImplementedError("Quantum circuit trimming is only implemented for QASM circuits.")
        try:
            qc = qasm3.loads(self.circuit_qasm)
        except Exception as e:
            raise ValueError("Invalid QASM3 syntax provided.") from e

        status = analyze_qubit_status(qc)
        remove_status = []
        if remove_idle_qubits:
            remove_status.append("idle")
        if remove_classical_qubits:
            remove_status.append("classical")
            _LOGGER.info(
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
            _LOGGER.warning("All measurements are dropped, no classical bits remain.")

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
        if self.circuit_qasm is not None:
            lines.append(f"  QASM string: {self.circuit_qasm}")
        return "\n".join(lines)

    def to_json(self) -> dict:
        """Convert the Circuit to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the quantum circuit.

        """
        data: dict = {}
        if self.circuit_qasm is not None:
            data["circuit_qasm"] = self.circuit_qasm
        return data

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the Circuit to an HDF5 group.

        Args:
            group: HDF5 group or file to write the quantum circuit to

        """
        if self.circuit_qasm is not None:
            group.attrs["circuit_qasm"] = self.circuit_qasm

    @classmethod
    def from_json(cls, json_data: dict) -> "Circuit":
        """Create a Circuit from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            Circuit: New instance of the Circuit

        """
        return cls(
            circuit_qasm=json_data.get("circuit_qasm"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "Circuit":
        """Load a Circuit from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from

        Returns:
            Circuit: New instance of the Circuit

        """
        return cls(
            circuit_qasm=group.attrs.get("circuit_qasm"),
        )
