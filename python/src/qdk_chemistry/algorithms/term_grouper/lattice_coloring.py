"""Lattice-geometry term grouper for fermionic lattice-model Hamiltonians."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper, TermGrouperSettings
from qdk_chemistry.data import LayeredPartition, QubitOperator
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from qdk_chemistry.data import LatticeGraph

__all__ = ["LatticeColoringTermGrouper"]


class LatticeColoringTermGrouperSettings(TermGrouperSettings):
    """Settings for the lattice-coloring term grouper."""

    def __init__(self):
        """Initialize the lattice-coloring settings."""
        super().__init__()
        self._set_default(
            "verify_commutation",
            "bool",
            False,
            "Check that every emitted layer is a commuting set. Quadratic in the layer size.",
        )


class LatticeColoringTermGrouper(TermGrouper):
    r"""Group a lattice-model Hamiltonian into commuting layers by edge color.

    A product formula is only cheaper than term-by-term Trotterization when its
    groups hold terms that commute, because a commuting group is exponentiated
    exactly by an ordered product of rotations. On a lattice the natural source
    of such groups is an **edge coloring**: two hopping terms commute when their
    bonds share no site, so each color class -- a set of vertex-disjoint bonds --
    is a commuting layer. A square lattice colors with four.

    The grouping produced here is

    * one group per edge color, holding that color's hopping terms, and
    * one final group holding every remaining term (the on-site interaction, any
      chemical-potential terms, and the identity), which all commute because they
      are diagonal.

    The interaction group is emitted **last** because
    :class:`~qdk_chemistry.algorithms.Trotter` applies the final group once at
    full time in the middle of its Strang schedule and the others twice at half
    time. The diagonal group is the cheapest to apply, so placing it there is the
    better of the two orderings Campbell compares.

    Note:
        Terms are classified by their Pauli support, so this expects a
        Jordan-Wigner encoded lattice model whose hopping terms carry exactly two
        non-identity :math:`X`/:math:`Y` positions. Terms that do not match a
        colored bond fall through to the diagonal group, so the partition always
        covers every term.

        This is a *grouping*, not a plaquette decomposition: each layer is
        realized as independent commuting Pauli rotations. Tiles larger than a
        single bond (plaquettes) are not products of Pauli exponentials of the
        original terms and need a fermionic-Givens circuit primitive instead.

    References:
        Campbell, E. T. "Early fault-tolerant simulations of the Hubbard model."
        *Quantum Science & Technology* 7.1 (2022): 015007.

        Bay-Smidt, A. J., et al. "Fault-tolerant quantum simulation of
        generalized Hubbard models." (2025). arXiv:2501.10314.

    """

    def __init__(
        self,
        lattice: LatticeGraph | None = None,
        *,
        coloring: dict[tuple[int, int], int] | None = None,
        verify_commutation: bool = False,
    ):
        """Initialize the grouper.

        The lattice is held directly rather than in ``settings`` because a
        :class:`~qdk_chemistry.data.LatticeGraph` is not one of the scalar types
        ``Settings`` serializes.

        Args:
            lattice: Lattice graph supplying the edge coloring and site count.
            coloring: Explicit edge coloring ``{(i, j): color}``. Defaults to the
                lattice's own :attr:`~qdk_chemistry.data.LatticeGraph.edge_coloring`.
            verify_commutation: Check that every emitted layer commutes.

        """
        super().__init__()
        self._settings = LatticeColoringTermGrouperSettings()
        self._settings.set("verify_commutation", verify_commutation)
        self.lattice = lattice
        self.coloring = coloring

    def name(self) -> str:
        """Return ``lattice_coloring`` as the algorithm name."""
        return "lattice_coloring"

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Return a copy of *qubit_hamiltonian* carrying a geometry-coloring partition.

        Args:
            qubit_hamiltonian: Hamiltonian to partition.

        Returns:
            QubitOperator: New instance with a ``LayeredPartition``.

        Raises:
            ValueError: If no lattice is configured, no edge coloring is
                available, or a layer fails the optional commutation check.

        """
        if self.lattice is None:
            raise ValueError(
                "LatticeColoringTermGrouper requires a lattice. Pass one to the constructor, "
                "for example LatticeColoringTermGrouper(LatticeGraph.square(4, 4))."
            )

        coloring = self.coloring if self.coloring is not None else self.lattice.edge_coloring
        if not coloring:
            raise ValueError(
                "The lattice graph carries no edge coloring. Use a factory method that "
                "provides one (for example LatticeGraph.square), or pass an explicit coloring."
            )

        num_sites = self.lattice.num_sites
        labels = qubit_hamiltonian.pauli_strings
        bond_color = {tuple(sorted(bond)): color for bond, color in coloring.items()}

        color_layers: dict[int, list[int]] = {}
        diagonal: list[int] = []
        for index, label in enumerate(labels):
            ends = [position for position, char in enumerate(reversed(label)) if char in "XY"]
            # A hopping term maps to exactly two X/Y endpoints; the Jordan-Wigner string
            # between them is all Z and carries no bond information.
            color = None
            if len(ends) == 2:
                bond = tuple(sorted(position % num_sites for position in ends))
                color = bond_color.get(bond)
            if color is None:
                diagonal.append(index)
            else:
                color_layers.setdefault(color, []).append(index)

        groups: list[tuple[tuple[int, ...], ...]] = [((tuple(color_layers[color])),) for color in sorted(color_layers)]
        if diagonal:
            groups.append((tuple(diagonal),))

        if self._settings.get("verify_commutation"):
            self._verify(labels, groups)

        Logger.debug(
            f"LatticeColoringTermGrouper: {len(color_layers)} color groups "
            f"({sum(len(v) for v in color_layers.values())} terms) "
            f"plus {len(diagonal)} diagonal terms."
        )

        return QubitOperator(
            pauli_strings=list(labels),
            coefficients=qubit_hamiltonian.coefficients.copy(),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            term_partition=LayeredPartition(strategy="geometry_coloring", groups=tuple(groups)),
        )

    @staticmethod
    def _verify(labels: list[str], groups: list[tuple[tuple[int, ...], ...]]) -> None:
        """Raise when any layer holds a non-commuting pair."""
        from qdk_chemistry.utils.pauli_commutation import do_pauli_labels_commute  # noqa: PLC0415

        for group_index, group in enumerate(groups):
            for layer in group:
                for position, i in enumerate(layer):
                    for j in layer[position + 1 :]:
                        if not do_pauli_labels_commute(labels[i], labels[j]):
                            raise ValueError(
                                f"Group {group_index} is not a commuting layer: terms {i} and {j} "
                                f"({labels[i]!r}, {labels[j]!r}) anticommute."
                            )
