"""Abstract base class and factory for term-grouper algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import QubitOperator, Settings, TaperingSpecification, TermPartition
from qdk_chemistry.data.qubit_operator.containers.pauli_decomposition import PauliDecompositionContainer
from qdk_chemistry.data.qubit_operator.containers.sparse_pauli_decomposition import SparsePauliDecompositionContainer

__all__ = ["TermGrouper", "TermGrouperFactory", "TermGrouperSettings"]


class TermGrouperSettings(Settings):
    """Settings for term-grouper algorithms."""

    def __init__(self):
        """Initialise default term-grouper settings (currently empty)."""
        super().__init__()


class TermGrouper(Algorithm):
    """Abstract base class for algorithms that partition Hamiltonian terms.

    A ``TermGrouper`` consumes a :class:`~qdk_chemistry.data.QubitOperator`
    and returns a *new* ``QubitOperator`` whose
    :attr:`~qdk_chemistry.data.QubitOperator.term_partition` is populated
    with the grouping computed by the strategy.

    Subclasses implement ``_run_impl``, which must return a new
    ``QubitOperator`` (the input must not be mutated).

    """

    def __init__(self):
        """Initialise the term grouper with default settings."""
        super().__init__()
        self._settings = TermGrouperSettings()

    def type_name(self) -> str:
        """Return ``term_grouper`` as the algorithm type name."""
        return "term_grouper"

    def run(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Reject non-Pauli representations, then run the grouping strategy.

        Args:
            qubit_hamiltonian: Hamiltonian whose Pauli terms should be partitioned.

        Returns:
            QubitOperator: A copy of the input with its term partition populated.

        Raises:
            ValueError: If the operator is not a Pauli decomposition.

        """
        container_type = qubit_hamiltonian.get_container_type()
        if not isinstance(qubit_hamiltonian.get_container(), PauliDecompositionContainer):
            raise ValueError(
                f"Term grouping requires a Pauli decomposition qubit operator; "
                f"got the {container_type!r} representation."
            )
        return super().run(qubit_hamiltonian)

    @staticmethod
    def _with_partition(
        qubit_hamiltonian: QubitOperator, partition: TermPartition, tapering: TaperingSpecification | None = None
    ) -> QubitOperator:
        """Copy the terms, coefficients, encoding and mode order with *partition*, keeping sparse storage sparse."""
        if isinstance(qubit_hamiltonian.get_container(), SparsePauliDecompositionContainer):
            return QubitOperator(
                container=SparsePauliDecompositionContainer(
                    qubit_hamiltonian.pauli_strings,
                    qubit_hamiltonian.coefficients,
                    qubit_hamiltonian.encoding,
                    qubit_hamiltonian.fermion_mode_order,
                    partition,
                    tapering,
                )
            )
        return QubitOperator(
            pauli_strings=list(qubit_hamiltonian.pauli_strings),
            coefficients=qubit_hamiltonian.coefficients.copy(),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            term_partition=partition,
            tapering=tapering,
        )

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Compute a term partition and return a new ``QubitOperator`` carrying it.

        Args:
            qubit_hamiltonian: Hamiltonian whose Pauli terms should be partitioned.

        Returns:
            QubitOperator: A copy of the input with
            :attr:`~qdk_chemistry.data.QubitOperator.term_partition` populated.

        """


class TermGrouperFactory(AlgorithmFactory):
    """Factory for :class:`TermGrouper` instances."""

    def algorithm_type_name(self) -> str:
        """Return ``term_grouper`` as the algorithm type name."""
        return "term_grouper"

    def default_algorithm_name(self) -> str:
        """Return ``commuting`` as the default term-grouper algorithm."""
        return "commuting"
