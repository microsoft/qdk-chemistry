"""Term-partition metadata for :class:`~qdk_chemistry.data.QubitHamiltonian`.

A :class:`TermPartition` records how the Pauli terms of a
:class:`~qdk_chemistry.data.QubitHamiltonian` are organised into algorithm-
relevant subsets.  Concrete subclasses include :class:`FlatPartition`
(single-level groups) and :class:`LayeredPartition` (group → layer
hierarchy).

The partition stores **indices** into
:attr:`~qdk_chemistry.data.QubitHamiltonian.pauli_strings`, not nested
``QubitHamiltonian`` objects, so that it serialises trivially and remains
small.

Lifecycle
---------

* The partition is *optional* metadata.  ``term_partition is None`` means the
  partition has not been computed for this Hamiltonian.
* Transformations that change the term ordering or qubit support
  (for example :meth:`~qdk_chemistry.data.QubitHamiltonian.to_interleaved`)
  must reset the partition to ``None`` on the new Hamiltonian.
* Algorithms that consume a partition should treat its presence as an explicit
  signal to exploit it (for example, by applying schedule-level Suzuki
  recursion or grouping measurements by basis).

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import Any

from qdk_chemistry.data.base import DataClass

__all__ = ["FlatPartition", "LayeredPartition", "TermPartition"]


class TermPartition(DataClass):
    """Base class for index-based partitions of Hamiltonian terms.

    Use :class:`FlatPartition` for single-level partitions or
    :class:`LayeredPartition` for hierarchical (group → layer) partitions.

    The ``strategy`` field is a free-form label identifying how the partition
    was produced (for example ``"geometry_coloring"``, ``"commuting"``,
    ``"qubit_wise_commuting"``).

    """

    _data_type_name = "term_partition"
    _serialization_version = "0.1.0"

    def __init__(self, *, strategy: str) -> None:
        """Initialize the term partition.

        Args:
            strategy: Label identifying how the partition was produced.

        """
        self.strategy = strategy

    @property
    def num_groups(self) -> int:
        """Return the number of top-level groups in the partition."""
        raise NotImplementedError

    def all_indices(self) -> list[int]:
        """Return every term index referenced by the partition, in order."""
        raise NotImplementedError

    def get_summary(self) -> str:
        """Return a summary of the partition."""
        return f"TermPartition(strategy={self.strategy!r}, num_groups={self.num_groups})"

    def to_json(self) -> dict[str, Any]:
        """Convert this partition to a JSON-serialisable dictionary."""
        raise NotImplementedError

    def to_hdf5(self, group) -> None:
        """Save this partition to an HDF5 group."""
        import json as _json  # noqa: PLC0415

        group.attrs["term_partition"] = _json.dumps(self.to_json())

    @staticmethod
    def from_json(data: dict[str, Any]) -> TermPartition:
        """Reconstruct a :class:`TermPartition` from :meth:`to_json` output.

        Args:
            data: Dict produced by :meth:`to_json` of either :class:`FlatPartition` or :class:`LayeredPartition`.

        Returns:
            The reconstructed partition.

        Raises:
            ValueError: If ``data["kind"]`` is not a recognised partition kind.

        """
        kind = data.get("kind")
        if kind == "flat":
            return FlatPartition(strategy=data["strategy"], groups=tuple(tuple(g) for g in data["groups"]))
        if kind == "layered":
            return LayeredPartition(
                strategy=data["strategy"],
                groups=tuple(tuple(tuple(layer) for layer in group) for group in data["groups"]),
            )
        raise ValueError(f"Unknown TermPartition kind: {kind!r}. Expected 'flat' or 'layered'.")

    @classmethod
    def from_hdf5(cls, group) -> TermPartition:
        """Load a :class:`TermPartition` from an HDF5 group."""
        import json as _json  # noqa: PLC0415

        data = _json.loads(group.attrs["term_partition"])
        return cls.from_json(data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TermPartition):
            return NotImplemented
        return type(self) is type(other) and self.strategy == other.strategy

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.strategy))


class FlatPartition(TermPartition):
    """Single-level partition: each group is a list of term indices.

    Suitable for algorithms that only care about which terms belong together
    (for example, qubit-wise commuting groups for measurement basis selection).

    The ``groups`` field is a tuple of groups; each group is a tuple of term
    indices into :attr:`~qdk_chemistry.data.QubitHamiltonian.pauli_strings`.

    Raises:
        TypeError: If ``groups`` is not a sequence of sequences of integers.

    """

    def __init__(self, *, strategy: str, groups: tuple[tuple[int, ...], ...]) -> None:
        """Initialize a flat partition.

        Args:
            strategy: Label identifying how the partition was produced.
            groups: Tuple of groups; each group is a tuple of term indices.

        """
        super().__init__(strategy=strategy)
        self.groups: tuple[tuple[int, ...], ...] = tuple(tuple(int(i) for i in group) for group in groups)
        # Freeze after all attributes are set.
        DataClass.__init__(self)

    @property
    def num_groups(self) -> int:
        """Return the number of groups."""
        return len(self.groups)

    def all_indices(self) -> list[int]:
        """Return every term index referenced by the partition, in order."""
        return [i for group in self.groups for i in group]

    def get_summary(self) -> str:
        """Return a summary of the flat partition."""
        return f"FlatPartition(strategy={self.strategy!r}, num_groups={self.num_groups})"

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this :class:`FlatPartition`."""
        return {
            "kind": "flat",
            "strategy": self.strategy,
            "groups": [list(group) for group in self.groups],
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlatPartition):
            return NotImplemented
        return self.strategy == other.strategy and self.groups == other.groups

    def __hash__(self) -> int:
        return hash(("FlatPartition", self.strategy, self.groups))


class LayeredPartition(TermPartition):
    """Two-level partition: each group is a sequence of parallelisable layers.

    Suitable for Trotter-style decompositions where the outer level controls
    Strang/Suzuki splitting order and the inner level groups operators with
    disjoint qubit supports that can be applied in parallel.

    The ``groups`` field is a nested tuple ``(group, layer, term_index)``:
    outer = groups, middle = layers within a group, inner = term indices into
    :attr:`~qdk_chemistry.data.QubitHamiltonian.pauli_strings`.

    Raises:
        TypeError: If ``groups`` is not the expected nested-sequence shape.

    """

    def __init__(self, *, strategy: str, groups: tuple[tuple[tuple[int, ...], ...], ...]) -> None:
        """Initialize a layered partition.

        Args:
            strategy: Label identifying how the partition was produced.
            groups: Nested tuple ``(group, layer, term_index)``.

        """
        super().__init__(strategy=strategy)
        self.groups: tuple[tuple[tuple[int, ...], ...], ...] = tuple(
            tuple(tuple(int(i) for i in layer) for layer in group) for group in groups
        )
        # Freeze after all attributes are set.
        DataClass.__init__(self)

    @property
    def num_groups(self) -> int:
        """Return the number of top-level groups."""
        return len(self.groups)

    def num_layers(self, group_index: int) -> int:
        """Return the number of parallelisable layers in ``group_index``."""
        return len(self.groups[group_index])

    def all_indices(self) -> list[int]:
        """Return every term index referenced by the partition, in order."""
        return [i for group in self.groups for layer in group for i in layer]

    def get_summary(self) -> str:
        """Return a summary of the layered partition."""
        return f"LayeredPartition(strategy={self.strategy!r}, num_groups={self.num_groups})"

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of this :class:`LayeredPartition`."""
        return {
            "kind": "layered",
            "strategy": self.strategy,
            "groups": [[list(layer) for layer in group] for group in self.groups],
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayeredPartition):
            return NotImplemented
        return self.strategy == other.strategy and self.groups == other.groups

    def __hash__(self) -> int:
        return hash(("LayeredPartition", self.strategy, self.groups))
