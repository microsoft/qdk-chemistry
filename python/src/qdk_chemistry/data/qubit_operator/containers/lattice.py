"""Lattice qubit operator container.

Represents a lattice model operator by the geometry it lives on rather than by an
enumerated list of Pauli terms. Algorithms that are defined in terms of the lattice --
:class:`~qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.plaquette_trotter.PlaquetteTrotter`
tiles it into plaquettes, for example -- read the connectivity directly and never pay
to materialize the Pauli decomposition, which grows with the lattice.

A :class:`~qdk_chemistry.data.LatticeGraph` built by a factory records the extents it was
generated from, so a consumer reads ``container.lattice.dims`` to learn the lattice's
shape. A graph assembled directly from an adjacency matrix reports no extents, since a
rectangle cannot be recovered from connectivity alone.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NoReturn

import scipy.sparse

from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer

if TYPE_CHECKING:
    import h5py

    from qdk_chemistry.data import LatticeGraph

__all__ = ["LatticeContainer"]


class LatticeContainer(QubitOperatorContainer):
    """A qubit operator defined by the lattice it lives on.

    The container carries the geometry only. Model parameters such as the hopping
    amplitude and on-site interaction belong to the algorithm consuming it, so the same
    lattice can be evolved under different couplings without rebuilding the operator.

    Each site carries two spin orbitals, as in a Fermi-Hubbard model, so the register is
    twice the site count.

    Args:
        lattice: Connectivity and edge weights of the lattice.
        encoding: Rejected; the container stores geometry and fixes no encoding.
        fermion_mode_order: Rejected; the container stores geometry and fixes no ordering.

    Raises:
        ValueError: If *encoding* or *fermion_mode_order* is supplied.

    """

    _data_type_name = "qubit_hamiltonian"
    _serialization_version = "0.1.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for qubit operators."""
        return "qubit_hamiltonian"

    def __init__(
        self,
        lattice: LatticeGraph,
        *,
        encoding: str | None = None,
        fermion_mode_order: object | None = None,
    ) -> None:
        """Initialize the container from a lattice."""
        # The geometry fixes neither an encoding nor a mode ordering, so accepting either
        # would silently drop it rather than honour it.
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
        super().__init__(None, None)
        self.lattice = lattice

    @property
    def type(self) -> str:
        """Return the container type."""
        return "lattice"

    @property
    def num_sites(self) -> int:
        """Return the number of lattice sites."""
        return int(self.lattice.num_sites)

    @property
    def num_qubits(self) -> int:
        """Return the register width, two spin orbitals per site."""
        return 2 * self.num_sites

    def to_matrix(self, sparse: bool = False) -> NoReturn:
        """Reject matrix conversion, which this representation does not implement.

        Args:
            sparse: Accepted so the signature matches the Pauli decomposition container.

        Raises:
            NotImplementedError: Always; the container carries geometry, not Pauli terms.

        """
        raise NotImplementedError(
            "Matrix conversion is not implemented for the 'lattice' representation, which carries "
            "no model parameters; build the operator as a Pauli decomposition first."
        )

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_arg(h, self.to_json())

    def _edges(self) -> list[list[float]]:
        """Return the upper-triangular edges as ``[site_a, site_b, weight]`` triples."""
        upper = scipy.sparse.triu(self.lattice.sparse_adjacency_matrix(), k=0, format="coo")
        return [
            [int(row), int(col), float(value)]
            for row, col, value in zip(upper.row, upper.col, upper.data, strict=True)
            if value != 0.0
        ]

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary."""
        return self._add_json_version(
            {
                "container_type": self.type,
                "num_sites": self.num_sites,
                "edges": self._edges(),
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

        Args:
            json_data: The serialized container.

        Returns:
            LatticeContainer: The reconstructed container.

        """
        from qdk_chemistry.data import LatticeGraph  # noqa: PLC0415  (avoids an import cycle)

        cls._validate_json_version(cls._serialization_version, json_data)
        edge_weights = {(int(a), int(b)): float(w) for a, b, w in json_data["edges"]}
        lattice = LatticeGraph.make_bidirectional(LatticeGraph(edge_weights, int(json_data["num_sites"])))
        return cls(lattice)

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
        dims = tuple(int(d) for d in self.lattice.dims)
        geometry = "x".join(str(d) for d in dims) if dims else f"{self.num_sites} sites"
        return f"Lattice qubit operator ({geometry}, {self.num_qubits} qubits)"
