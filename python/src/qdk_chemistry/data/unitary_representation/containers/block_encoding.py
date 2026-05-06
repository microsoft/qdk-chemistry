"""QDK/Chemistry block encoding LCU (Linear Combination of Unitaries) container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import asdict, dataclass
from typing import Any

import h5py
import numpy as np

from .base import UnitaryContainer

__all__ = ["BlockEncodingContainer", "ControlledOperation", "Prepare", "Select"]


@dataclass(frozen=True)
class ControlledOperation:
    """A single controlled unitary operation in the SELECT oracle."""

    ctrl_state: int
    """Integer encoding of the control state that activates this operation."""

    operation: str
    """Operation descriptor (e.g., a Pauli string like ``"XZI"``)."""

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the ControlledOperation to an HDF5 group."""
        group.attrs["ctrl_state"] = self.ctrl_state
        group.attrs["operation"] = self.operation

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ControlledOperation":
        """Load a ControlledOperation from an HDF5 group."""
        return cls(
            ctrl_state=int(group.attrs["ctrl_state"]),
            operation=str(group.attrs["operation"]),
        )


@dataclass(frozen=True)
class Prepare:
    """Class representing the PREPARE oracle for block encoding."""

    statevector: np.ndarray
    """Pre-computed amplitude array to load into the ancilla register."""

    num_prepare_qubits: int
    """Number of qubits in the prepare register."""

    prepare_qubits: list[int]
    """List of qubit indices for the prepare register."""

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the Prepare oracle to an HDF5 group."""
        group.create_dataset("statevector", data=self.statevector)
        group.attrs["num_prepare_qubits"] = self.num_prepare_qubits
        group.attrs["prepare_qubits"] = self.prepare_qubits

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "Prepare":
        """Load a Prepare oracle from an HDF5 group."""
        return cls(
            statevector=np.array(group["statevector"]),
            num_prepare_qubits=int(group.attrs["num_prepare_qubits"]),
            prepare_qubits=list(group.attrs["prepare_qubits"]),
        )


@dataclass(frozen=True)
class Select:
    """Class representing the SELECT oracle for block encoding."""

    controlled_operations: list[ControlledOperation]
    """List of controlled operations."""

    phases: np.ndarray
    """Array of +1/-1 phase corrections (uncontrolled global phase per term)."""

    num_prepare_qubits: int
    """Number of control qubits."""

    num_target_qubits: int
    """Number of target (system) qubits."""

    prepare_qubits: list[int]
    """List of qubit indices for the prepare register."""

    target_qubits: list[int]
    """List of qubit indices for the target (system) register."""

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the Select oracle to an HDF5 group."""
        group.create_dataset("phases", data=self.phases)
        group.attrs["num_prepare_qubits"] = self.num_prepare_qubits
        group.attrs["num_target_qubits"] = self.num_target_qubits
        group.attrs["prepare_qubits"] = self.prepare_qubits
        group.attrs["target_qubits"] = self.target_qubits
        ops_group = group.create_group("controlled_operations")
        for i, op in enumerate(self.controlled_operations):
            op.to_hdf5(ops_group.create_group(f"op_{i}"))

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "Select":
        """Load a Select oracle from an HDF5 group."""
        ops_group = group["controlled_operations"]
        controlled_ops = [
            ControlledOperation.from_hdf5(ops_group[key])
            for key in sorted(ops_group.keys(), key=lambda k: int(k.split("_")[1]))
        ]
        return cls(
            controlled_operations=controlled_ops,
            phases=np.array(group["phases"]),
            num_prepare_qubits=int(group.attrs["num_prepare_qubits"]),
            num_target_qubits=int(group.attrs["num_target_qubits"]),
            prepare_qubits=list(group.attrs["prepare_qubits"]),
            target_qubits=list(group.attrs["target_qubits"]),
        )


class BlockEncodingContainer(UnitaryContainer):
    r"""Container for a Linear Combination of Unitaries (LCU) decomposition.

    Stores the pre-computed PREPARE and SELECT sub-objects that define a
    block-encoding circuit. This container is agnostic to the specific method
    used to compute these objects — that logic lives in the builder.

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
        reflect: bool = False,
    ) -> None:
        r"""Initialize a BlockEncodingContainer.

        Args:
            power: Number of times to apply the walk operator (for W^power in QPE).
            prepare: The PREPARE oracle data class instance.
            select: The SELECT oracle data class instance.
            reflect: When True, the circuit mapper wraps the block encoding with a
                quantum walk operator (use with QPE). When False, the plain block
                encoding is used (use with Hadamard test).

        """
        self.power = power
        self.prepare = prepare
        self.select = select
        self.reflect = reflect

        super().__init__()

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (system + ancilla)."""
        return self.select.num_target_qubits + self.prepare.num_prepare_qubits

    @property
    def type(self) -> str:
        """Get the type of the unitary container."""
        return "block_encoding"

    def to_json(self) -> dict[str, Any]:
        """Save the BlockEncodingContainer to a JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "container_type": self.type,
            "power": self.power,
            "prepare": asdict(self.prepare),
            "select": asdict(self.select),
            "reflect": self.reflect,
        }

        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the BlockEncodingContainer to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["power"] = self.power
        group.attrs["reflect"] = self.reflect
        self.prepare.to_hdf5(group.create_group("prepare"))
        self.select.to_hdf5(group.create_group("select"))

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "BlockEncodingContainer":
        """Create BlockEncodingContainer from a JSON dictionary."""
        cls._validate_json_version(cls._serialization_version, json_data)

        prep_data = json_data["prepare"]
        prepare = Prepare(
            statevector=np.array(prep_data["statevector"], dtype=float),
            num_prepare_qubits=prep_data["num_prepare_qubits"],
            prepare_qubits=prep_data["prepare_qubits"],
        )

        sel_data = json_data["select"]
        controlled_ops = [
            ControlledOperation(
                ctrl_state=op["ctrl_state"],
                operation=op["operation"],
            )
            for op in sel_data["controlled_operations"]
        ]
        select = Select(
            controlled_operations=controlled_ops,
            phases=np.array(sel_data["phases"], dtype=int),
            num_prepare_qubits=sel_data["num_prepare_qubits"],
            num_target_qubits=sel_data["num_target_qubits"],
            prepare_qubits=sel_data["prepare_qubits"],
            target_qubits=sel_data["target_qubits"],
        )

        return cls(
            power=json_data.get("power", 1),
            prepare=prepare,
            select=select,
            reflect=bool(json_data.get("reflect", False)),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "BlockEncodingContainer":
        """Load an instance from an HDF5 group."""
        prepare = Prepare.from_hdf5(group["prepare"])
        select = Select.from_hdf5(group["select"])
        reflect = bool(group.attrs.get("reflect", False))
        power = int(group.attrs["power"])
        return cls(
            power=power,
            prepare=prepare,
            select=select,
            reflect=reflect,
        )

    def get_summary(self) -> str:
        """Get summary of the LCU container."""
        return (
            f"LCU Container:\n"
            f"  Power: {self.power}\n"
            f"  Prepare: {self.prepare.num_prepare_qubits} qubits, statevector shape {self.prepare.statevector.shape}\n"
            f"  Select: {self.select.num_prepare_qubits} control qubits, {self.select.num_target_qubits} target qubits,"
            f" {len(self.select.controlled_operations)} controlled operations\n"
            f"  Reflect: {'Yes' if self.reflect else 'No'}"
        )
