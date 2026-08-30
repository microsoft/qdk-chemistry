"""Qubit operator wrapper and representation containers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.data._hashing import _hash_str
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer
from qdk_chemistry.data.qubit_operator.containers.pauli_lcu import PauliLCUContainer
from qdk_chemistry.data.qubit_operator.containers.sossa import SOSContainer

if TYPE_CHECKING:
    import h5py

    from qdk_chemistry._core.data import TaperingSpecification
    from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
    from qdk_chemistry.data.term_partition import TermPartition

__all__ = [
    "QubitHamiltonian",
    "QubitOperator",
]


class QubitOperator(DataClass):
    """Data class wrapping a concrete qubit operator container.

    For backward compatibility, Pauli-LCU operators may also be initialized
    directly from Pauli strings and coefficients.

    Attribute access is forwarded to the wrapped container by ``__getattr__``, so the
    available attributes depend on the representation. The two below are documented here
    because they are part of the long-standing public surface; they resolve only when the
    operator wraps a :class:`~qdk_chemistry.data.qubit_operator.containers.pauli_lcu.PauliLCUContainer`,
    and raise :exc:`AttributeError` otherwise.

    Attributes:
        pauli_strings (list[str]): List of Pauli strings representing the ``QubitOperator``.
        term_partition (~qdk_chemistry.data.term_partition.TermPartition | None): Optional index-based partition of
            :attr:`pauli_strings` into algorithm-relevant groups.

    """

    _data_type_name = "qubit_hamiltonian"
    _serialization_version = "0.1.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for qubit operators."""
        return "qubit_hamiltonian"

    def __init__(
        self,
        container: QubitOperatorContainer | list[str] | None = None,
        coefficients: np.ndarray | None = None,
        encoding: str | None = None,
        fermion_mode_order: FermionModeOrder | str | None = None,
        term_partition: TermPartition | None = None,
        tapering: TaperingSpecification | None = None,
        *,
        pauli_strings: list[str] | None = None,
    ) -> None:
        """Initialize from a container or directly from Pauli strings and coefficients.

        Args:
            container: A qubit operator container, or legacy positional Pauli strings.
            coefficients: Coefficients for legacy Pauli-string construction.
            encoding: Fermion-to-qubit encoding for legacy Pauli-string construction.
            fermion_mode_order: Fermion mode ordering for legacy Pauli-string construction.
            term_partition: Term partition for legacy Pauli-string construction.
            tapering: Tapering metadata for legacy Pauli-string construction.
            pauli_strings: Pauli strings for legacy keyword construction.

        """
        if isinstance(container, QubitOperatorContainer):
            if any(
                value is not None
                for value in (coefficients, encoding, fermion_mode_order, term_partition, tapering, pauli_strings)
            ):
                raise TypeError("QubitOperator container construction does not accept Pauli-LCU arguments")
            resolved_container = container
        else:
            if container is not None and pauli_strings is not None:
                raise TypeError("Specify Pauli strings either positionally or by keyword, not both")
            resolved_pauli_strings = pauli_strings if pauli_strings is not None else container
            if resolved_pauli_strings is None or coefficients is None:
                raise TypeError("QubitOperator requires a QubitOperatorContainer or Pauli strings and coefficients")
            resolved_container = PauliLCUContainer(
                resolved_pauli_strings,
                np.asarray(coefficients),
                encoding,
                fermion_mode_order,
                term_partition,
                tapering,
            )
        self._container = resolved_container
        super().__init__()

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "qubit_operator")
        _hash_str(h, self._container.content_hash())

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped container.

        Args:
            name (str): Attribute name

        Returns:
            Any: The corresponding attribute of the wrapped container

        Raises:
            AttributeError: If the wrapped representation does not provide the attribute

        """
        # Underscored names must not forward: DataClass.__setattr__ probes `_initialized`,
        # which the container already has, and would then reject `self._container = ...`.
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        container = self.__dict__.get("_container")
        if container is None:
            return super().__getattr__(name)
        try:
            return getattr(container, name)
        except AttributeError:
            raise AttributeError(
                f"'{name}' is not available on this qubit operator; "
                f"the {container.type!r} representation does not provide it."
            ) from None

    def equiv(self, other: QubitOperator, atol: float = 1e-12) -> bool:
        """Check mathematical equivalence with another Pauli-LCU operator."""
        if not isinstance(other, QubitOperator):
            return False
        return self._container.equiv(other._container, atol=atol)

    def to_interleaved(self, n_spatial: int) -> QubitOperator:
        """Convert a Pauli-LCU operator from blocked to interleaved ordering."""
        return QubitOperator(self._container.to_interleaved(n_spatial))

    def __add__(self, other: QubitOperator) -> QubitOperator:
        """Add two Pauli-LCU operators."""
        if not isinstance(other, QubitOperator):
            return NotImplemented
        return QubitOperator(self._container + other._container)

    def __mul__(self, scalar: Any) -> QubitOperator:
        """Scale a Pauli-LCU operator."""
        result = self._container * scalar
        if result is NotImplemented:
            return NotImplemented
        return QubitOperator(result)

    def __rmul__(self, scalar: Any) -> QubitOperator:
        """Support scalar multiplication of a Pauli-LCU operator."""
        return self.__mul__(scalar)

    def get_container_type(self) -> str:
        """Return the concrete container type."""
        return self._container.type

    def get_container(self) -> QubitOperatorContainer:
        """Return the concrete representation container."""
        return self._container

    def get_summary(self) -> str:
        """Return the container summary."""
        return self._container.get_summary()

    def to_json(self) -> dict[str, Any]:
        """Convert the wrapped container to a JSON dictionary."""
        return self._container.to_json()

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the wrapped container to an HDF5 group."""
        self._container.to_hdf5(group)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> QubitOperator:
        """Create a qubit operator from a JSON dictionary.

        Documents written before this class delegated to a container carry no
        ``container_type``; they are all Pauli LCU operators, so a missing key reads as
        one rather than failing. ``_serialization_version`` did not change when the key
        was added, so the version guard cannot distinguish them.
        """
        container_type = json_data.get("container_type", "pauli_lcu")
        if container_type == "pauli_lcu":
            container = PauliLCUContainer.from_json(json_data)
        elif container_type == "sossa":
            container = SOSContainer.from_json(json_data)
        else:
            raise ValueError(f"Unsupported qubit operator container type: {container_type}")
        return cls(container)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> QubitOperator:
        """Create a qubit operator from an HDF5 group.

        A group without a ``container_type`` attribute predates container delegation and
        holds a Pauli LCU operator; see :meth:`from_json`.
        """
        container_type = group.attrs.get("container_type", "pauli_lcu")
        if container_type == "pauli_lcu":
            container = PauliLCUContainer.from_hdf5(group)
        elif container_type == "sossa":
            container = SOSContainer.from_hdf5(group)
        else:
            raise ValueError(f"Unsupported qubit operator container type: {container_type}")
        return cls(container)


class _DeprecatedQubitOperatorAliasMeta(type(QubitOperator)):  # type: ignore[misc]
    """Metaclass that makes the deprecated alias behave like :class:`QubitOperator` for type checks."""

    def __instancecheck__(cls, instance: object) -> bool:
        """Report any :class:`QubitOperator` instance as an instance of the alias."""
        return isinstance(instance, QubitOperator)

    def __subclasscheck__(cls, subclass: type) -> bool:
        """Report any :class:`QubitOperator` subclass as a subclass of the alias."""
        return issubclass(subclass, QubitOperator)


class QubitHamiltonian(QubitOperator, metaclass=_DeprecatedQubitOperatorAliasMeta):
    """Deprecated alias for :class:`QubitOperator`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct a :class:`QubitOperator`, warning that ``QubitHamiltonian`` is deprecated."""
        warnings.warn(
            "'QubitHamiltonian' has been renamed to 'QubitOperator' and is deprecated; it will be "
            "removed in a future release. Replace 'QubitHamiltonian' with 'QubitOperator' "
            "(from qdk_chemistry.data import QubitOperator).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
