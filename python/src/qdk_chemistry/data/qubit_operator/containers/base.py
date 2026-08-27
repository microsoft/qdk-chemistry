"""Qubit operator container base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder

if TYPE_CHECKING:
    import h5py

__all__ = ["QubitOperatorContainer"]


class QubitOperatorContainer(DataClass):
    """Abstract base class for qubit operator representations."""

    _data_type_name = "qubit_operator_container"
    _serialization_version = "0.1.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for qubit operator containers.

        Returns:
            ``"qubit_operator_container"``.

        """
        return "qubit_operator_container"

    def __init__(
        self,
        encoding: str | None,
        fermion_mode_order: FermionModeOrder | str | None,
    ) -> None:
        """Initialize shared qubit operator metadata."""
        self.encoding = encoding
        self.fermion_mode_order = FermionModeOrder(fermion_mode_order) if fermion_mode_order is not None else None
        super().__init__()

    @property
    @abstractmethod
    def type(self) -> str:
        """Return the container type."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits."""

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary."""

    @abstractmethod
    def to_hdf5(self, group: "h5py.Group") -> None:
        """Write the container to an HDF5 group."""

    @classmethod
    @abstractmethod
    def from_json(cls, json_data: dict[str, Any]) -> "QubitOperatorContainer":
        """Create a container from a JSON dictionary."""

    @classmethod
    @abstractmethod
    def from_hdf5(cls, group: "h5py.Group") -> "QubitOperatorContainer":
        """Create a container from an HDF5 group."""

    @abstractmethod
    def get_summary(self) -> str:
        """Return a human-readable summary."""
