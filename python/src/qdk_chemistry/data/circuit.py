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
from dataclasses import dataclass
from typing import Any

import h5py
import qsharp._native
import qsharp.estimator
import qsharp.openqasm
from qsharp.openqasm import OutputSemantics

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QsharpFactoryData"]


@dataclass(frozen=True)
class QsharpFactoryData:
    """Data class for Q# factory data used to create Q# circuit."""

    program: Callable
    """The Q# Callable."""

    parameter: dict[str, Any]
    """The parameter to be passed to the Q# Callable when creating the circuit."""


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
        qsharp_factory: QsharpFactoryData | None = None,
        encoding: str | None = None,
    ) -> None:
        """Initialize a Circuit.

        Args:
            qasm: The quantum circuit in QASM format. Defaults to None.
            qir: The QIR representation of the quantum circuit. Defaults to None.
            qsharp: The Q# circuit object. Defaults to None.
            qsharp_factory: The factory data for creating a Q# program. Defaults to None.
            qsharp_op: The Q# operation associated with the circuit. Defaults to None.
            encoding: The fermion-to-qubit encoding assumed by this circuit.
                Valid values include "jordan-wigner", "bravyi-kitaev", "parity", or None.
                Defaults to None.

        Notes:
            At least one representation (qasm, qir, qsharp, or qsharp_factory) must be provided.
            If multiple representations are available, conversion methods attempt to follow this priority order:
            - get_qasm(): Returns qasm string if available, otherwise converts from qir via Qiskit if possible.
            - get_qir(): Returns qir if available, otherwise converts from qasm
            - get_qsharp_circuit(): Returns Q# circuit if available, otherwise converts from qasm
            - get_qiskit_circuit(): Converts from qir if available, otherwise converts from qasm

        """
        Logger.trace_entering()
        self.qasm = qasm
        self.qir = qir
        self.qsharp = qsharp
        self._qsharp_factory = qsharp_factory
        self._qsharp_op = qsharp_op
        self.encoding = encoding

        # Check that a representation of the quantum circuit is given by the keyword arguments
        if not any([self.qasm, self.qsharp, self.qir, self._qsharp_factory]):
            raise RuntimeError("No representation of the quantum circuit is set.")

        # Make instance immutable after construction (handled by base class)
        super().__init__()

    def _has_qir(self) -> bool:
        """Check if the Circuit has a QIR representation.

        Returns:
            True if either qir or qsharp_factory is set, indicating that a QIR representation is available.

        """
        return self.qir is not None or self._qsharp_factory is not None

    def get_qasm(self) -> str:
        """Get the quantum circuit in QASM format.

        Returns:
            str: The quantum circuit in QASM format.

        Notes:
            If both QASM and QIR representations are available, this method returns the QASM string.
            If only QIR is available, it attempts to convert it to QASM using Qiskit.

        """
        if self.qasm:
            if self._has_qir():
                Logger.warn("Both QASM and QIR representations are available. Return QASM.")
            return self.qasm
        if not self._has_qir():
            raise RuntimeError("The quantum circuit in QASM format is not set.")

        try:
            from qiskit import qasm3  # noqa: PLC0415

            from qdk_chemistry.plugins.qiskit._interop.qir import qir_ir_to_qiskit  # noqa: PLC0415

        except ImportError as err:
            raise RuntimeError("Qiskit is not available. Cannot convert circuit to QASM format.") from err

        qir = self.get_qir()
        return qasm3.dumps(qir_ir_to_qiskit(str(qir)))

    def get_qir(self) -> qsharp._qsharp.QirInputData | str:
        """Get QIR representation of the quantum circuit.

        Returns:
            The QIR representation of the quantum circuit.

        Notes:
            If both QIR and QASM representations are available, this method returns the QIR representation.
            If Q# factory data is available, it compiles it to QIR using the Q# compiler.
            If only QASM is available, it attempts to convert it to QIR using the Q# OpenQASM compiler.

        """
        if self.qir:
            if self.qasm:
                Logger.warn("Both QIR and QASM representations are available. Return QIR.")
            return self.qir
        if self._qsharp_factory and self.qir is None:
            compiled_qir = qsharp.compile(self._qsharp_factory.program, *self._qsharp_factory.parameter.values())
            # Cache the compiled qir if qir is not already set
            object.__setattr__(self, "qir", compiled_qir)
            return compiled_qir
        if self.qasm:
            return qsharp.openqasm.compile(self.qasm, output_semantics=OutputSemantics.OpenQasm)

        raise RuntimeError("The QIR representation of the quantum circuit is not set.")

    def get_qsharp_circuit(self, prune_classical_qubits: bool = False) -> qsharp._native.Circuit:
        """Parse a Circuit object into a Q# circuit object.

        Args:
            prune_classical_qubits: If True, classical qubits are removed from the circuit. Only applicable
                when converting from Q# factory data.

        Returns:
            qsharp._native.Circuit: A Q# Circuit object.

        Raises:
            RuntimeError: If the circuit cannot be converted to Q# format.

        Notes:
            If both Q# and QASM representations are available, this method returns the Q# circuit.
            If Q# factory data is available, it compiles it to a Q# circuit using the Q# compiler.
            If only QASM is available, it attempts to convert it to Q#.

        """
        if self.qsharp:
            if self.qasm:
                Logger.warn("Both Q# and QASM representations are available. Return Q# circuit.")
            return self.qsharp
        if self._qsharp_factory and self.qsharp is None:
            return qsharp.circuit(
                self._qsharp_factory.program,
                *self._qsharp_factory.parameter.values(),
                prune_classical_qubits=prune_classical_qubits,
                generation_method=qsharp.CircuitGenerationMethod.Static,
            )
        if self.qasm:
            return qsharp.openqasm.circuit(self.qasm)

        raise RuntimeError("The quantum circuit is not set in a Q# format.")

    def estimate(
        self,
        params: dict[str, Any] | list[Any] | qsharp.estimator.EstimatorParams | None = None,
    ) -> qsharp.estimator.EstimatorResult:
        """Estimate resources for the quantum circuit.

        Args:
            params: Resource estimation parameters. Accepts a dict, list, or ``qsharp.estimator.EstimatorParams``.

        Returns:
            qsharp.estimator.EstimatorResult: The estimated resources.

        Raises:
            RuntimeError: If no suitable circuit representation is available for estimation.

        """
        if self._qsharp_factory is not None:
            return qsharp.estimate(
                self._qsharp_factory.program,
                params,
                *self._qsharp_factory.parameter.values(),
            )
        if self.qasm is not None:
            return qsharp.openqasm.estimate(self.qasm, params)

        raise RuntimeError("Cannot estimate resources: no Q# factory data or QASM representation is available.")

    def get_qiskit_circuit(self):
        """Convert the Circuit to a Qiskit QuantumCircuit.

        Raises:
            RuntimeError: If Qiskit is not available or if the circuit cannot be converted.

        Notes:
            This method attempts to convert the Circuit to a Qiskit QuantumCircuit object when Qiskit is installed.
            If both QIR and QASM representations are available, this method converts from QIR.
            If only QASM is available, it attempts to convert it to a Qiskit QuantumCircuit using Qiskit's QASM parser.

        """
        Logger.trace_entering()

        cached = self.__dict__.get("_qiskit_circuit", None)
        if cached is not None:
            return cached

        try:
            from qiskit import qasm3  # noqa: PLC0415

            from qdk_chemistry.plugins.qiskit._interop.qir import (  # noqa: PLC0415
                qir_ir_to_qiskit,
            )
        except ImportError as err:
            raise RuntimeError("Qiskit is not available. Cannot convert circuit to Qiskit format.") from err

        if self._has_qir():
            if self.qasm:
                Logger.warn("Both QIR and QASM representations are available. Convert from QIR.")
            result = qir_ir_to_qiskit(str(self.get_qir()))
        elif self.qasm:
            result = qasm3.loads(self.qasm)
        else:
            raise RuntimeError("The quantum circuit cannot be converted to Qiskit format.")

        # Cache via object.__setattr__ to bypass the immutability guard
        object.__setattr__(self, "_qiskit_circuit", result)
        return result

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
            data["qasm"] = self.get_qasm()
        if self._has_qir():
            data["qir"] = str(self.get_qir())
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
            group.attrs["qasm"] = self.get_qasm()
        if self._has_qir():
            group.attrs["qir"] = str(self.get_qir())
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
