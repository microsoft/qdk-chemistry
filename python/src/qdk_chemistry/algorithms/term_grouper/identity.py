"""Identity term grouper: each Pauli term is its own group."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper
from qdk_chemistry.data import FlatPartition, QubitHamiltonian

__all__ = ["IdentityTermGrouper"]


class IdentityTermGrouper(TermGrouper):
    """Trivial grouper — every term is placed in its own single-element group.

    Useful to clear an existing :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition`
    while still passing through the standard ``term_grouper`` interface, or to
    disable downstream group-aware optimisation in a controlled way.

    """

    def name(self) -> str:
        """Return ``identity`` as the algorithm name."""
        return "identity"

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> QubitHamiltonian:
        """Return a copy of ``qubit_hamiltonian`` with one group per term.

        Args:
            qubit_hamiltonian: Hamiltonian to wrap.

        Returns:
            QubitHamiltonian: A new instance with a :class:`FlatPartition`.

        """
        n = len(qubit_hamiltonian.pauli_strings)
        partition = FlatPartition(
            strategy="identity",
            groups=tuple((i,) for i in range(n)),
        )
        return QubitHamiltonian(
            pauli_strings=list(qubit_hamiltonian.pauli_strings),
            coefficients=qubit_hamiltonian.coefficients.copy(),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            term_partition=partition,
        )
