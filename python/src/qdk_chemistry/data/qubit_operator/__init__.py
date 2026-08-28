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
from qdk_chemistry.data.qubit_operator.containers.sossa import SOSSAContainer

if TYPE_CHECKING:
    import h5py
    import scipy

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

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self._container.num_qubits

    @property
    def encoding(self) -> str | None:
        """Return the fermion-to-qubit encoding."""
        return self._container.encoding

    @property
    def fermion_mode_order(self) -> FermionModeOrder | None:
        """Return the fermion mode ordering."""
        return self._container.fermion_mode_order

    def _get_pauli_lcu_container(self, member: str) -> PauliLCUContainer:
        """Return the Pauli-LCU container required by a compatibility member."""
        if not isinstance(self._container, PauliLCUContainer):
            raise TypeError(
                f"{member} is only available for Pauli-LCU operators; "
                f"this operator uses the {self._container.type!r} representation."
            )
        return self._container

    @property
    def pauli_strings(self) -> list[str]:
        """Return the Pauli strings for a Pauli-LCU operator."""
        return self._get_pauli_lcu_container("pauli_strings").pauli_strings

    @property
    def coefficients(self) -> np.ndarray:
        """Return the coefficients for a Pauli-LCU operator."""
        return self._get_pauli_lcu_container("coefficients").coefficients

    @property
    def term_partition(self) -> TermPartition | None:
        """Return the term partition for a Pauli-LCU operator."""
        return self._get_pauli_lcu_container("term_partition").term_partition

    @property
    def tapering(self) -> TaperingSpecification | None:
        """Return the tapering metadata for a Pauli-LCU operator."""
        return self._get_pauli_lcu_container("tapering").tapering

    @property
    def schatten_norm(self) -> float:
        """Return the coefficient one-norm for a Pauli-LCU operator."""
        return self._get_pauli_lcu_container("schatten_norm").schatten_norm

    def to_matrix(self, sparse: bool = False) -> np.ndarray | scipy.sparse.spmatrix:
        """Convert a Pauli-LCU operator to its matrix representation."""
        return self._get_pauli_lcu_container("to_matrix()").to_matrix(sparse=sparse)

    def equiv(self, other: QubitOperator, atol: float = 1e-12) -> bool:
        """Check mathematical equivalence with another Pauli-LCU operator."""
        if not isinstance(other, QubitOperator):
            return False
        container = self._get_pauli_lcu_container("equiv()")
        other_container = other._get_pauli_lcu_container("equiv()")
        return container.equiv(other_container, atol=atol)

    def is_hermitian(self, tolerance: float = 1e-12) -> bool:
        """Return whether a Pauli-LCU operator is Hermitian."""
        return self._get_pauli_lcu_container("is_hermitian()").is_hermitian(tolerance=tolerance)

    def get_real_coefficients(
        self, tolerance: float = 1e-12, sort_by_magnitude: bool = False
    ) -> list[tuple[str, float]]:
        """Return real Pauli coefficients above the requested tolerance."""
        return self._get_pauli_lcu_container("get_real_coefficients()").get_real_coefficients(
            tolerance=tolerance, sort_by_magnitude=sort_by_magnitude
        )

    def to_interleaved(self, n_spatial: int) -> QubitOperator:
        """Convert a Pauli-LCU operator from blocked to interleaved ordering."""
        container = self._get_pauli_lcu_container("to_interleaved()")
        return QubitOperator(container.to_interleaved(n_spatial))

    def __add__(self, other: QubitOperator) -> QubitOperator:
        """Add two Pauli-LCU operators."""
        if not isinstance(other, QubitOperator):
            return NotImplemented
        container = self._get_pauli_lcu_container("addition")
        other_container = other._get_pauli_lcu_container("addition")
        return QubitOperator(container + other_container)

    def __mul__(self, scalar: Any) -> QubitOperator:
        """Scale a Pauli-LCU operator."""
        container = self._get_pauli_lcu_container("multiplication")
        result = container * scalar
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
            container = SOSSAContainer.from_json(json_data)
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
            container = SOSSAContainer.from_hdf5(group)
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
