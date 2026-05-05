"""QDK/Chemistry block encoding LCU (Linear Combination of Unitaries) container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

from .base import UnitaryContainer

__all__ = ["BlockEncodingContainer", "ControlledOperation", "Prepare", "Reflect", "Select"]


@dataclass(frozen=True)
class ControlledOperation:
    """A single controlled unitary operation in the SELECT oracle.

    Attributes:
        ctrl_qubits: Indices of control qubits (relative to select register).
        ctrl_state: Integer encoding of the control state that activates this operation.
        target_qubits: Indices of target qubits (relative to system register).
        operation: Operation descriptor (e.g., a Pauli string like ``"XZI"``).

    """

    ctrl_qubits: list[int]
    """Indices of control qubits (relative to select register)."""

    ctrl_state: int
    """Integer encoding of the control state that activates this operation."""

    target_qubits: list[int]
    """Indices of target qubits (relative to system register)."""

    operation: str
    """Operation descriptor (e.g., a Pauli string like ``"XZI"``)."""


@dataclass(frozen=True)
class Prepare:
    """Class representing the PREPARE oracle for block encoding.

    Attributes:
        method: Preparation method identifier (e.g., ``"block_encoding"`` for PreparePureStateD).
        statevector: Array of amplitudes to prepare in the ancilla register.
        num_prepare_qubits: Number of qubits in the prepare/ancilla register.

    """

    method: str
    """Preparation method identifier (e.g., ``"block_encoding"``)."""

    statevector: np.ndarray
    """Pre-computed amplitude array to load into the ancilla register."""

    num_prepare_qubits: int
    """Number of qubits in the prepare/ancilla register."""


@dataclass(frozen=True)
class Select:
    """Class representing the SELECT oracle for block encoding.

    Attributes:
        controlled_operations: List of controlled operations, each specifying
            which control state activates which unitary on which target qubits.
        signs: Array of +1/-1 phase corrections (uncontrolled global phase per term).
        method: Method identifier for the SELECT oracle (e.g., ``"block_encoding"``).

    """

    controlled_operations: list[ControlledOperation]
    """List of controlled operations."""

    signs: np.ndarray
    """Array of +1/-1 phase corrections (uncontrolled global phase per term)."""

    method: str = "block_encoding"
    """Method identifier for the SELECT oracle."""

    @property
    def num_ctrl_qubits(self) -> int:
        """Number of control qubits in the select register."""
        return len(self.controlled_operations[0].ctrl_qubits)

    @property
    def num_target_qubits(self) -> int:
        """Number of target (system) qubits."""
        return len(self.controlled_operations[0].target_qubits)


@dataclass(frozen=True)
class Reflect:
    """Class representing the reflection operator for block encoding.

    Attributes:
        qubits: Qubit indices (relative to ancilla register) to reflect upon.

    """

    qubits: list[int]
    """Qubit indices (relative to ancilla register) to reflect upon."""


class BlockEncodingContainer(UnitaryContainer):
    r"""Container for a Linear Combination of Unitaries (LCU) decomposition.

    Stores the pre-computed PREPARE, SELECT, and (optionally) REFLECT sub-objects
    that define a block-encoding circuit. This container is agnostic to the specific
    method used to compute these objects — that logic lives in the builder.

    .. math::

        W = \text{PREPARE}^\dagger \cdot \text{SELECT} \cdot \text{PREPARE}

    """

    # Class attribute for filename validation
    _data_type_name = "block_encoding_container"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        prepare: Prepare,
        select: Select,
        power: int = 1,
        reflect: Reflect | None = None,
    ) -> None:
        r"""Initialize a BlockEncodingContainer.

        Args:
            prepare: The PREPARE oracle sub-object.
            select: The SELECT oracle sub-object.
            power: Number of times to apply the walk operator (for W^power in QPE).
            reflect: The REFLECT oracle sub-object. When provided, the circuit mapper
                wraps the block encoding with a quantum walk operator (use with QPE).
                When None, the plain block encoding is used (use with Hadamard test).

        """
        self.prepare = prepare
        self.select = select
        self.power = power
        self.reflect = reflect

        super().__init__()

    # ── Derived properties ──

    @property
    def num_system_qubits(self) -> int:
        """Number of system qubits, derived from the first SELECT operation."""
        return self.select.num_target_qubits

    @property
    def num_select_qubits(self) -> int:
        """Number of ancilla qubits for the PREPARE register."""
        return self.prepare.num_prepare_qubits

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (system + ancilla)."""
        return self.num_system_qubits + self.num_select_qubits

    @property
    def quantum_walk(self) -> bool:
        """Whether the quantum walk operator is used (reflect is not None)."""
        return self.reflect is not None

    @property
    def type(self) -> str:
        """Get the type of the unitary container."""
        return "block_encoding"

    # ── Serialization ──

    def to_json(self) -> dict[str, Any]:
        """Convert the BlockEncodingContainer to a dictionary for JSON serialization."""
        prepare_data = {
            "method": self.prepare.method,
            "statevector": self.prepare.statevector.tolist(),
            "num_prepare_qubits": self.prepare.num_prepare_qubits,
        }
        select_data = {
            "controlled_operations": [
                {
                    "ctrl_qubits": op.ctrl_qubits,
                    "ctrl_state": op.ctrl_state,
                    "target_qubits": op.target_qubits,
                    "operation": op.operation,
                }
                for op in self.select.controlled_operations
            ],
            "signs": self.select.signs.tolist(),
            "method": self.select.method,
        }
        reflect_data = None
        if self.reflect is not None:
            reflect_data = {"qubits": self.reflect.qubits}

        data: dict[str, Any] = {
            "container_type": self.type,
            "power": self.power,
            "prepare": prepare_data,
            "select": select_data,
            "reflect": reflect_data,
        }

        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the BlockEncodingContainer to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["power"] = self.power

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "BlockEncodingContainer":
        """Create BlockEncodingContainer from a JSON dictionary."""
        cls._validate_json_version(cls._serialization_version, json_data)

        # Reconstruct Prepare
        prep_data = json_data["prepare"]
        prepare = Prepare(
            method=prep_data["method"],
            statevector=np.array(prep_data["statevector"], dtype=float),
            num_prepare_qubits=prep_data["num_prepare_qubits"],
        )

        # Reconstruct Select
        sel_data = json_data["select"]
        controlled_ops = [
            ControlledOperation(
                ctrl_qubits=op["ctrl_qubits"],
                ctrl_state=op["ctrl_state"],
                target_qubits=op["target_qubits"],
                operation=op["operation"],
            )
            for op in sel_data["controlled_operations"]
        ]
        select = Select(
            controlled_operations=controlled_ops,
            signs=np.array(sel_data["signs"], dtype=int),
        )

        # Reconstruct Reflect
        reflect = None
        if json_data.get("reflect") is not None:
            reflect = Reflect(qubits=json_data["reflect"]["qubits"])

        return cls(
            prepare=prepare,
            select=select,
            power=json_data.get("power", 1),
            reflect=reflect,
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "BlockEncodingContainer":
        """Load an instance from an HDF5 group."""
        raise NotImplementedError("HDF5 deserialization not yet implemented for v0.2.0 format.")

    def get_summary(self) -> str:
        """Get summary of the LCU container."""
        return (
            f"LCU Container:\n"
            f"  Number of terms: {len(self.select.controlled_operations)}\n"
            f"  System qubits: {self.num_system_qubits}\n"
            f"  Select (ancilla) qubits: {self.num_select_qubits}\n"
            f"  Prepare method: {self.prepare.method}\n"
            f"  Quantum walk: {self.quantum_walk}"
        )
