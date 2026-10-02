"""Lattice qubit operator container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any, NoReturn

from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer

if TYPE_CHECKING:
    import h5py

    from qdk_chemistry.data import LatticeGeometry

__all__ = ["LatticeContainer"]


class LatticeContainer(QubitOperatorContainer):
    """A qubit operator defined by the lattice geometry it lives on and its model couplings.

    The couplings are named model coefficients, such as ``{"hopping": t, "interaction": U}``
    for the Fermi-Hubbard model. The container attaches no meaning to the names: each consuming
    algorithm documents the names it reads and rejects any it does not, so the same container
    serves other lattice models under their own names. The couplings are serialized and hashed
    with the geometry, so two operators that differ only in a coupling are distinct. Each site
    carries two spin orbitals, so the register is twice the site count.

    The geometry is a :class:`~qdk_chemistry.data.LatticeGeometry`, whose factories
    document what their ``nx`` and ``ny`` count: sites for
    :meth:`~qdk_chemistry.data.LatticeGeometry.square` and
    :meth:`~qdk_chemistry.data.LatticeGeometry.triangular`, and unit cells for
    :meth:`~qdk_chemistry.data.LatticeGeometry.honeycomb` and
    :meth:`~qdk_chemistry.data.LatticeGeometry.kagome`. Because the geometry stores
    site positions rather than an edge list, it carries no site numbering that could
    drift out of step with the shape it reports, and loading one validates its layout.

    Args:
        geometry: Site positions and periodic supercell vectors of the lattice.
        couplings: Model coefficients keyed by name. Defaults to none.
        encoding: Rejected; the container stores geometry and fixes no encoding.
        fermion_mode_order: Rejected; the container stores geometry and fixes no ordering.

    Raises:
        TypeError: If *geometry* is not a :class:`~qdk_chemistry.data.LatticeGeometry`, or a
            coupling name is not a string or its value is not a real number.
        ValueError: If *encoding* or *fermion_mode_order* is supplied, a coupling name is empty,
            or a coupling value is not finite.

    """

    _data_type_name = "lattice_container"
    _serialization_version = "0.1.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the container type name.

        Returns:
            ``"lattice_container"``.

        """
        return "lattice_container"

    def __init__(
        self,
        geometry: LatticeGeometry,
        *,
        couplings: Mapping[str, float] | None = None,
        encoding: str | None = None,
        fermion_mode_order: object | None = None,
    ) -> None:
        """Initialize the container from a lattice geometry and its model couplings."""
        if encoding is not None:
            raise ValueError(
                "LatticeContainer stores lattice geometry and fixes no fermion-to-qubit encoding, "
                f"so 'encoding' would be ignored; drop it (got {encoding!r})."
            )
        if fermion_mode_order is not None:
            raise ValueError(
                "LatticeContainer stores lattice geometry and fixes no fermion mode ordering, "
                f"so 'fermion_mode_order' would be ignored; drop it (got {fermion_mode_order!r})."
            )
        from qdk_chemistry.data import LatticeGeometry  # noqa: PLC0415  (avoids an import cycle)

        if not isinstance(geometry, LatticeGeometry):
            raise TypeError(
                "LatticeContainer stores a LatticeGeometry, which carries the site positions the "
                f"consuming algorithm needs, but got a {type(geometry).__name__}; build one with a "
                "factory such as LatticeGeometry.square(nx, ny)."
            )
        self.geometry = geometry
        self._couplings = self._validated_couplings(couplings)
        super().__init__(None, None)

    @staticmethod
    def _validated_couplings(couplings: Mapping[str, float] | None) -> dict[str, float]:
        """Return the couplings as a name-sorted dictionary of finite floats."""
        if couplings is None:
            return {}
        if not isinstance(couplings, Mapping):
            raise TypeError(f"couplings must map names to numbers, got a {type(couplings).__name__}.")
        validated: dict[str, float] = {}
        for name, value in couplings.items():
            if not isinstance(name, str):
                raise TypeError(f"Coupling names must be strings, got {name!r}.")
            if not name:
                raise ValueError("Coupling names must not be empty.")
            # bool is a Real subclass, but True as a coupling is always a mistake.
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Coupling {name!r} must be a real number, got a {type(value).__name__}.")
            if not math.isfinite(value):
                raise ValueError(f"Coupling {name!r} must be finite, got {value}.")
            validated[name] = float(value)
        return dict(sorted(validated.items()))

    @property
    def couplings(self) -> dict[str, float]:
        """Return a copy of the model couplings, keyed by name."""
        return dict(self._couplings)

    @property
    def type(self) -> str:
        """Return the container type."""
        return "lattice"

    @property
    def num_qubits(self) -> int:
        """Return the register width, two spin orbitals per site."""
        return 2 * int(self.geometry.num_sites)

    def to_matrix(self, sparse: bool = False) -> NoReturn:
        """Reject matrix conversion, which this representation does not implement.

        Args:
            sparse: Accepted so the signature matches the Pauli decomposition container.

        Raises:
            NotImplementedError: Always; the container carries geometry, not Pauli terms.

        """
        raise NotImplementedError(
            "Matrix conversion is not implemented for the 'lattice' representation, which carries "
            "geometry and named couplings rather than Pauli terms; build the operator as a Pauli "
            "decomposition first."
        )

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_arg(h, self.to_json())

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary."""
        return self._add_json_version(
            {
                "container_type": self.type,
                "geometry": json.loads(self.geometry.to_json()),
                "couplings": dict(self._couplings),
            }
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the container to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["payload"] = json.dumps(self.to_json())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> LatticeContainer:
        """Create a lattice container from JSON.

        The geometry is rebuilt through :meth:`~qdk_chemistry.data.LatticeGeometry.from_json`, which
        validates the stored layout, so a document whose site positions and lattice
        dimensions disagree is rejected here rather than in the consuming algorithm.

        Args:
            json_data: The serialized container.

        Returns:
            LatticeContainer: The reconstructed container.

        """
        from qdk_chemistry.data import LatticeGeometry  # noqa: PLC0415  (avoids an import cycle)

        cls._validate_json_version(cls._serialization_version, json_data)
        return cls(
            LatticeGeometry.from_json(json.dumps(json_data["geometry"])),
            couplings=json_data.get("couplings"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> LatticeContainer:
        """Create a lattice container from an HDF5 group.

        Args:
            group: The HDF5 group holding the serialized container.

        Returns:
            LatticeContainer: The reconstructed container.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls.from_json(json.loads(group.attrs["payload"]))

    def get_summary(self) -> str:
        """Return a human-readable summary of the container."""
        periods = self.geometry.periods
        directions = 0 if periods is None else int(periods.shape[0])
        couplings = ", ".join(f"{name}={value:g}" for name, value in self._couplings.items()) or "none"
        return (
            f"Lattice qubit operator ({self.geometry.num_sites} sites, "
            f"{directions} periodic direction(s), {self.num_qubits} qubits, couplings: {couplings})"
        )
