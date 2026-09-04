"""QDK/Chemistry quantum walk operator container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from typing import Any

import h5py
import numpy as np

from qdk_chemistry.data._hashing import _hash_float, _hash_int, _hash_str

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

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for quantum-walk containers.

        Returns:
            ``"quantum_walk_container"``.

        """
        return "quantum_walk_container"

    # Serialization version for this class
    _serialization_version = "0.2.0"

    def eigenvalue_from_phase(self, phase_fraction: float) -> tuple[float, ...]:
        r"""Recover every eigenvalue consistent with a quantum-walk phase.

        For a walk operator whose eigenvalues are
        :math:`e^{\pm i \arccos(E_k / \lambda)}`, QPE applied to :math:`W^p`
        measures :math:`\varphi` such that
        :math:`\arccos(E / \lambda) = 2\pi(\varphi + k)/p` for some integer
        :math:`k`, so the candidate eigenvalues are

        .. math::

            E_k = \lambda \cos\!\left(\frac{2\pi(\varphi + k)}{p}\right),
            \qquad k = 0, \ldots, p - 1

        Branches whose walk angles coincide (up to a :math:`10^{-12}` phase
        tolerance) are reported once, so :math:`p = 1` reduces to the single
        eigenvalue :math:`E = \lambda \cos(2\pi\varphi)`.

        Args:
            phase_fraction: Measured phase fraction :math:`\varphi \in [0, 1)`.

        Returns:
            tuple[float, ...]: The candidate eigenvalues, sorted ascending.

        """
        phi = phase_fraction % 1.0
        power = self.power
        if power == 1:
            return (float(self.scale * np.cos(2 * np.pi * phi)),)
        # cos is even, so two branches coincide exactly when their walk angles fold onto
        # the same value in [0, 1/2]. Round only the dedup key, never the returned value.
        folded_angles: dict[float, float] = {}
        for k in range(power):
            angle = (phi + k) / power
            folded = min(angle, 1.0 - angle)
            folded_angles.setdefault(round(folded, 12), folded)
        return tuple(sorted(float(self.scale * np.cos(2 * np.pi * a)) for a in folded_angles.values()))

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

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for LCU-walk containers.

        Returns:
            ``"lcu_walk_container"``.

        """
        return "lcu_walk_container"

    # Serialization version for this class
    _serialization_version = "0.2.0"

    def __init__(self, block_encoding: BlockEncodingContainer, power: int = 1, scale: float = 1.0) -> None:
        """Initialize an LCUWalkContainer.

        Args:
            block_encoding: The block encoding container to wrap with a reflection.
            power: Number of times to apply the walk operator (for :math:`W^k` in QPE).
            scale: The 1-norm used for eigenvalue-phase conversion.

        Raises:
            TypeError: If ``power`` is not an integer.
            ValueError: If ``power`` is not positive.

        """
        # bool is an int subclass, but True as a power is always a mistake.
        if isinstance(power, bool) or not isinstance(power, int | np.integer):
            raise TypeError(f"power must be an integer, got {type(power).__name__}.")
        if power < 1:
            raise ValueError(f"power must be a positive integer, got {power}.")
        self._block_encoding = block_encoding
        self._power = int(power)
        self.scale = scale
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
            "scale": self.scale,
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
        group.attrs["scale"] = self.scale
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
            scale=json_data.get("scale", 1.0),
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
            scale=float(group.attrs.get("scale", 1.0)),
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the walk operator container.

        Returns:
            str: Multi-line summary describing the walk operator and its block encoding.

        """
        be_summary = self._block_encoding.get_summary()
        indented = "\n".join("    " + line for line in be_summary.splitlines())
        return f"LCU Walk Operator Container:\n  Power: {self.power}\n  Block Encoding:\n{indented}"

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "lcu_walk_container")
        _hash_int(h, self._power)
        _hash_float(h, self.scale)
        # Delegate to block encoding's content_hash
        _hash_str(h, self._block_encoding.content_hash(truncate_chars=0))
