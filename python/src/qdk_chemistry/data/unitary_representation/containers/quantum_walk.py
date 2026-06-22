"""QDK/Chemistry quantum walk operator container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from typing import Any

import h5py

from .base import UnitaryContainer
from .block_encoding import BlockEncodingContainer

__all__: list[str] = ["LCUWalkContainer", "QuantumWalkContainer"]


class QuantumWalkContainer(UnitaryContainer):
    r"""Abstract base class for quantum walk operator containers.

    A quantum walk operator is defined as:

    .. math::

        W = (2|0\rangle\langle 0| - I) \cdot B[H]

    where :math:`B[H]` is a block encoding of the Hamiltonian. The eigenvalues
    of :math:`W` are :math:`e^{\pm i \arccos(E_k / \lambda)}`, enabling
    eigenvalue extraction via quantum phase estimation.

    """

    # Class attribute for filename validation
    _data_type_name = "quantum_walk_container"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    @property
    @abstractmethod
    def power(self) -> int:
        """Number of times to apply the walk operator."""


class LCUWalkContainer(QuantumWalkContainer):
    r"""Quantum walk operator wrapping an LCU block encoding.

    Represents:

    .. math::

        W^k = \left[(2|0\rangle\langle 0| - I) \cdot
        \text{PREPARE}^\dagger \cdot \text{SELECT} \cdot \text{PREPARE}\right]^k

    This container stores a reference to the underlying
    :class:`~qdk_chemistry.data.unitary_representation.containers.block_encoding.LCUContainer`
    and exposes all block-encoding data through it.

    """

    # Class attribute for filename validation
    _data_type_name = "lcu_walk_container"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(self, block_encoding: BlockEncodingContainer, power: int = 1) -> None:
        """Initialize an LCUWalkContainer.

        Args:
            block_encoding: The block encoding container to wrap with a reflection.
            power: Number of times to apply the walk operator (for :math:`W^k` in QPE).

        """
        self._block_encoding = block_encoding
        self._power = power
        super().__init__()

    @property
    def block_encoding(self) -> BlockEncodingContainer:
        """Get the underlying block encoding container.

        Returns:
            The LCU block encoding that this walk operator wraps.

        """
        return self._block_encoding

    @property
    def power(self) -> int:
        """Number of times to apply the walk operator.

        Returns:
            int: The power value.

        """
        return self._power

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (same as the block encoding).

        Returns:
            int: The combined qubit count.

        """
        return self._block_encoding.num_qubits

    @property
    def type(self) -> str:
        """Get the type of the unitary container.

        Returns:
            str: The type string ``"lcu_walk"``.

        """
        return "lcu_walk"

    def to_json(self) -> dict[str, Any]:
        """Save the LCUWalkContainer to a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Dictionary representation including container type, power,
                and nested block encoding.

        """
        data: dict[str, Any] = {
            "container_type": self.type,
            "power": self.power,
            "block_encoding": self._block_encoding.to_json(),
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the LCUWalkContainer to an HDF5 group.

        Args:
            group: HDF5 group to write container data to.

        """
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["power"] = self.power
        be_group = group.create_group("block_encoding")
        self._block_encoding.to_hdf5(be_group)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "LCUWalkContainer":
        """Create an LCUWalkContainer from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data.

        Returns:
            LCUWalkContainer: The deserialized instance.

        """
        from .block_encoding import LCUContainer  # noqa: PLC0415

        cls._validate_json_version(cls._serialization_version, json_data)
        block_encoding = LCUContainer.from_json(json_data["block_encoding"])
        return cls(
            block_encoding=block_encoding,
            power=json_data.get("power", 1),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "LCUWalkContainer":
        """Load an LCUWalkContainer from an HDF5 group.

        Args:
            group: HDF5 group to read container data from.

        Returns:
            LCUWalkContainer: The deserialized instance.

        """
        from .block_encoding import LCUContainer  # noqa: PLC0415

        block_encoding = LCUContainer.from_hdf5(group["block_encoding"])
        power = int(group.attrs["power"])
        return cls(
            block_encoding=block_encoding,
            power=power,
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the walk operator container.

        Returns:
            str: Multi-line summary describing the walk operator and its block encoding.

        """
        be_summary = self._block_encoding.get_summary()
        indented = "\n".join("    " + line for line in be_summary.splitlines())
        return f"LCU Walk Operator Container:\n  Power: {self.power}\n  Block Encoding:\n{indented}"
