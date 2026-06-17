"""QDK/Chemistry block encoding container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import tempfile
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from .base import UnitaryContainer

if TYPE_CHECKING:
    from qdk_chemistry.data import Wavefunction

__all__ = ["BlockEncodingContainer", "ControlledOperation", "LCUContainer", "Select"]


class BlockEncodingContainer(UnitaryContainer):
    """Abstract base class for block encoding containers."""

    # Class attribute for filename validation
    _data_type_name = "block_encoding_container"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    @property
    @abstractmethod
    def power(self) -> int:
        """Number of times to apply the walk operator."""

    @property
    @abstractmethod
    def quantum_walk(self) -> bool:
        """Whether to wrap with a quantum walk operator."""


@dataclass(frozen=True)
class ControlledOperation:
    """A single controlled unitary operation in the SELECT oracle."""

    ctrl_state: int
    """Integer encoding of the control state that activates this operation."""

    operation: str
    """Operation descriptor (e.g., a Pauli string like ``"XZI"``)."""

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the ControlledOperation to an HDF5 group.

        Args:
            group: HDF5 group to write attributes to.

        """
        group.attrs["ctrl_state"] = self.ctrl_state
        group.attrs["operation"] = self.operation

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ControlledOperation":
        """Load a ControlledOperation from an HDF5 group.

        Args:
            group: HDF5 group to read attributes from.

        Returns:
            ControlledOperation: The deserialized instance.

        """
        return cls(
            ctrl_state=int(group.attrs["ctrl_state"]),
            operation=str(group.attrs["operation"]),
        )


@dataclass(frozen=True)
class Select:
    """Class representing the SELECT oracle for block encoding."""

    controlled_operations: list[ControlledOperation]
    """List of controlled operations."""

    phases: np.ndarray
    """Array of +1/-1 phase corrections (uncontrolled global phase per term)."""

    num_prepare_ancillas: int
    """Number of control qubits."""

    num_target_qubits: int
    """Number of target (system) qubits."""

    prepare_qubits: list[int]
    """List of qubit indices for the prepare register."""

    target_qubits: list[int]
    """List of qubit indices for the target (system) register."""

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the Select oracle to an HDF5 group.

        Args:
            group: HDF5 group to write phases, attributes, and controlled operations to.

        """
        group.create_dataset("phases", data=self.phases)
        group.attrs["num_prepare_ancillas"] = self.num_prepare_ancillas
        group.attrs["num_target_qubits"] = self.num_target_qubits
        group.attrs["prepare_qubits"] = self.prepare_qubits
        group.attrs["target_qubits"] = self.target_qubits
        ops_group = group.create_group("controlled_operations")
        for i, op in enumerate(self.controlled_operations):
            op.to_hdf5(ops_group.create_group(f"op_{i}"))

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "Select":
        """Load a Select oracle from an HDF5 group.

        Args:
            group: HDF5 group to read phases, attributes, and controlled operations from.

        Returns:
            Select: The deserialized instance.

        """
        ops_group = group["controlled_operations"]
        controlled_ops = [
            ControlledOperation.from_hdf5(ops_group[key])
            for key in sorted(ops_group.keys(), key=lambda k: int(k.split("_")[1]))
        ]
        return cls(
            controlled_operations=controlled_ops,
            phases=np.array(group["phases"]),
            num_prepare_ancillas=int(group.attrs["num_prepare_ancillas"]),
            num_target_qubits=int(group.attrs["num_target_qubits"]),
            prepare_qubits=list(group.attrs["prepare_qubits"]),
            target_qubits=list(group.attrs["target_qubits"]),
        )


class LCUContainer(BlockEncodingContainer):
    r"""Container for a Linear Combination of Unitaries (LCU) decomposition.

    Stores the pre-computed PREPARE and SELECT sub-objects that define a
    block-encoding circuit. This container is agnostic to the specific method
    used to compute these objects — that logic lives in the builder.

    .. math::

        W = \text{PREPARE}^\dagger \cdot \text{SELECT} \cdot \text{PREPARE}

    """

    # Class attribute for filename validation
    _data_type_name = "lcu_container"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        prepare: "Wavefunction",
        select: Select,
        power: int = 1,
        quantum_walk: bool = False,
    ) -> None:
        r"""Initialize an LCUContainer.

        Args:
            prepare: The prepare wavefunction encoding coefficients for the block encoded Hamiltonian.
            select: The select oracle for controlled operations.
            power: Number of times to apply the walk operator (for W^power in QPE).
            quantum_walk: When True, the circuit mapper wraps the block encoding with a
                quantum walk operator (use with QPE). When False, the plain block
                encoding is used (use with Hadamard test).

        """
        self._power = power
        self.prepare = prepare
        self.select = select
        self._quantum_walk = quantum_walk

        super().__init__()

    @property
    def power(self) -> int:
        """Number of times to apply the walk operator.

        Returns:
            int: The power value.

        """
        return self._power

    @property
    def quantum_walk(self) -> bool:
        """Whether to wrap with a quantum walk operator.

        Returns:
            bool: True if quantum walk is enabled.

        """
        return self._quantum_walk

    @property
    def num_prepare_ancillas(self) -> int:
        """Number of qubits in the prepare ancillary register.

        Derived from the orbital basis size of the prepare wavefunction.

        Returns:
            int: The ancilla qubit count.

        """
        return self.prepare.get_orbitals().num_modes()

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (system + ancilla).

        Returns:
            int: The combined qubit count.

        """
        return self.select.num_target_qubits + self.num_prepare_ancillas

    @property
    def type(self) -> str:
        """Get the type of the unitary container.

        Returns:
            str: The type string ``"lcu"``.

        """
        return "lcu"

    def to_json(self) -> dict[str, Any]:
        """Save the LCUContainer to a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Dictionary representation including container type, power,
                prepare, select, and quantum_walk fields.

        """
        data: dict[str, Any] = {
            "container_type": self.type,
            "power": self.power,
            "prepare": self.prepare.to_json(),
            "select": asdict(self.select) | {"phases": self.select.phases.tolist()},
            "quantum_walk": self.quantum_walk,
        }

        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the LCUContainer to an HDF5 group.

        Args:
            group: HDF5 group to write container data to.

        """
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["power"] = self.power
        group.attrs["quantum_walk"] = self.quantum_walk

        # TODO(hid_t bridging): replace temp-file bridge with direct
        # Wavefunction.to_hdf5(h5py.Group) once exposed in the pybind11 bindings.
        fd, tmp_path = tempfile.mkstemp(suffix=".wavefunction.h5")
        os.close(fd)
        try:
            self.prepare.to_hdf5_file(tmp_path)
            with h5py.File(tmp_path, "r") as src:
                for key in src:
                    src.copy(key, group.require_group("prepare"))
                for attr_name, attr_val in src.attrs.items():
                    group["prepare"].attrs[attr_name] = attr_val
        finally:
            os.unlink(tmp_path)

        self.select.to_hdf5(group.create_group("select"))

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "LCUContainer":
        """Create an LCUContainer from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized LCU data.

        Returns:
            LCUContainer: The deserialized instance.

        """
        from qdk_chemistry.data import Wavefunction  # noqa: PLC0415

        cls._validate_json_version(cls._serialization_version, json_data)

        prepare = Wavefunction.from_json(json_data["prepare"])

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
            num_prepare_ancillas=sel_data["num_prepare_ancillas"],
            num_target_qubits=sel_data["num_target_qubits"],
            prepare_qubits=sel_data["prepare_qubits"],
            target_qubits=sel_data["target_qubits"],
        )

        return cls(
            power=json_data.get("power", 1),
            prepare=prepare,
            select=select,
            quantum_walk=bool(json_data.get("quantum_walk", json_data.get("reflect", False))),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "LCUContainer":
        """Load an LCUContainer from an HDF5 group.

        Args:
            group: HDF5 group to read container data from.

        Returns:
            LCUContainer: The deserialized instance.

        """
        from qdk_chemistry.data import Wavefunction  # noqa: PLC0415

        # TODO(hid_t bridging): replace temp-file bridge with direct
        # Wavefunction.from_hdf5(h5py.Group) once exposed in the pybind11 bindings.
        fd, tmp_path = tempfile.mkstemp(suffix=".wavefunction.h5")
        os.close(fd)
        try:
            with h5py.File(tmp_path, "w") as dst:
                src_group = group["prepare"]
                for key in src_group:
                    src_group.copy(key, dst)
                for attr_name, attr_val in src_group.attrs.items():
                    dst.attrs[attr_name] = attr_val
            prepare = Wavefunction.from_hdf5_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        select = Select.from_hdf5(group["select"])
        quantum_walk = bool(group.attrs.get("quantum_walk", group.attrs.get("reflect", False)))
        power = int(group.attrs["power"])
        return cls(
            power=power,
            prepare=prepare,
            select=select,
            quantum_walk=quantum_walk,
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the LCU container.

        Returns:
            str: Multi-line summary describing power, prepare, select, and quantum_walk settings.

        """
        return (
            f"LCU Container:\n"
            f"  Power: {self.power}\n"
            f"  Prepare: {self.num_prepare_ancillas} qubits, statevector shape {len(self.prepare.get_coefficients())}\n"
            f"  Select: {self.num_prepare_ancillas} control qubits, {self.select.num_target_qubits} target qubits,"
            f" {len(self.select.controlled_operations)} controlled operations\n"
            f"  Quantum Walk: {'Yes' if self.quantum_walk else 'No'}"
        )
