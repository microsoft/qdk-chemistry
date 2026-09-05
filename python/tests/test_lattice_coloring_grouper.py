"""Tests for the lattice-coloring term grouper in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.term_grouper.lattice_coloring import LatticeColoringTermGrouper
from qdk_chemistry.data import LatticeGraph, MajoranaMapping
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian
from qdk_chemistry.utils.pauli_commutation import do_pauli_labels_commute


def _hubbard(side):
    lattice = LatticeGraph.square(side, side, periodic_x=True, periodic_y=True)
    operator = create("qubit_mapper").run(
        create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=8.0),
        mapping=MajoranaMapping.jordan_wigner(2 * side * side),
    )
    return lattice, operator


class TestLatticeColoringTermGrouper:
    """Tests for grouping a lattice model by edge colour."""

    @pytest.mark.parametrize("side", [3, 4])
    def test_partition_covers_every_term_exactly_once(self, side):
        """A partition that drops or duplicates a term would change the Hamiltonian."""
        lattice, operator = _hubbard(side)
        grouped = LatticeColoringTermGrouper(lattice).run(operator)
        covered = sorted(index for group in grouped.term_partition.groups for layer in group for index in layer)
        assert covered == list(range(len(operator.pauli_strings)))

    @pytest.mark.parametrize("side", [3, 4])
    def test_every_emitted_layer_commutes(self, side):
        """A layer is only exact if its terms pairwise commute."""
        lattice, operator = _hubbard(side)
        grouped = LatticeColoringTermGrouper(lattice).run(operator)
        labels = grouped.pauli_strings
        for group in grouped.term_partition.groups:
            for layer in group:
                for position, i in enumerate(layer):
                    for j in layer[position + 1 :]:
                        assert do_pauli_labels_commute(labels[i], labels[j]), (
                            f"terms {labels[i]!r} and {labels[j]!r} share a layer but anticommute"
                        )

    def test_verify_commutation_setting_accepts_a_valid_partition(self):
        """The optional self-check must not reject a partition it should accept."""
        lattice, operator = _hubbard(4)
        LatticeColoringTermGrouper(lattice, verify_commutation=True).run(operator)

    def test_verify_commutation_rejects_a_bad_colouring(self):
        """Colouring two touching bonds alike must be caught rather than silently emitted."""
        lattice, operator = _hubbard(4)
        # Collapse the colouring so every bond shares one colour; touching bonds
        # anticommute, so the layer is no longer a commuting set.
        flattened = dict.fromkeys(lattice.edge_coloring, 0)
        with pytest.raises(ValueError, match="not a commuting layer"):
            LatticeColoringTermGrouper(lattice, coloring=flattened, verify_commutation=True).run(operator)

    def test_diagonal_group_is_emitted_last(self):
        """The Strang schedule applies the final group at full time, so it should be the cheap one."""
        lattice, operator = _hubbard(4)
        grouped = LatticeColoringTermGrouper(lattice).run(operator)
        labels = grouped.pauli_strings
        last = grouped.term_partition.groups[-1]
        for layer in last:
            for index in layer:
                assert all(axis not in "XY" for axis in labels[index]), (
                    "the final group should hold only diagonal terms"
                )

    def test_requires_a_lattice(self):
        """Without a lattice there is no colouring to group by."""
        _, operator = _hubbard(4)
        with pytest.raises(ValueError, match="requires a lattice"):
            LatticeColoringTermGrouper().run(operator)
