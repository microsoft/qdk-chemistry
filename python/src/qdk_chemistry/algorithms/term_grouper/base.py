"""Abstract base class and factory for term-grouper algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import QubitHamiltonian, Settings

__all__ = ["TermGrouper", "TermGrouperFactory"]


class TermGrouperSettings(Settings):
    """Settings for term-grouper algorithms."""

    def __init__(self):
        """Initialise default term-grouper settings (currently empty)."""
        super().__init__()


class TermGrouper(Algorithm):
    """Abstract base class for algorithms that partition Hamiltonian terms.

    A ``TermGrouper`` consumes a :class:`~qdk_chemistry.data.QubitHamiltonian`
    and returns a *new* ``QubitHamiltonian`` whose
    :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition` is populated
    with the grouping computed by the strategy.

    Subclasses implement ``_run_impl``, which must return a new
    ``QubitHamiltonian`` (the input must not be mutated).

    """

    def __init__(self):
        """Initialise the term grouper with default settings."""
        super().__init__()
        self._settings = TermGrouperSettings()

    def type_name(self) -> str:
        """Return ``term_grouper`` as the algorithm type name."""
        return "term_grouper"

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> QubitHamiltonian:
        """Compute a term partition and return a new ``QubitHamiltonian`` carrying it.

        Args:
            qubit_hamiltonian: Hamiltonian whose Pauli terms should be partitioned.

        Returns:
            QubitHamiltonian: A copy of the input with
            :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition` populated.

        """


class TermGrouperFactory(AlgorithmFactory):
    """Factory for :class:`TermGrouper` instances."""

    def algorithm_type_name(self) -> str:
        """Return ``term_grouper`` as the algorithm type name."""
        return "term_grouper"

    def default_algorithm_name(self) -> str:
        """Return ``commuting`` as the default term-grouper algorithm."""
        return "commuting"
