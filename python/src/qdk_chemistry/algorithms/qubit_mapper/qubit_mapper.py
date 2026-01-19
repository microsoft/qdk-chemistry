"""QDK/Chemistry qubit mapper abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory

if TYPE_CHECKING:  # Only needed for type annotations; avoid importing into module namespace
    from qdk_chemistry.data import Hamiltonian, QubitHamiltonian

__all__: list[str] = []


class QubitMapper(Algorithm):
    """Abstract base class for mapping a Hamiltonian to a QubitHamiltonian."""

    def __init__(self):
        """Initialize the QubitMapper."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``qubit_mapper`` as the algorithm type name."""
        return "qubit_mapper"

    @abstractmethod
    def _run_impl(self, hamiltonian: Hamiltonian) -> QubitHamiltonian:
        """Construct a QubitHamiltonian from a Hamiltonian using the mapping specified.

        Args:
            hamiltonian (Hamiltonian): The fermionic Hamiltonian.

        Returns:
           QubitHamiltonian: An instance of the QubitHamiltonian.

        """


class QubitMapperFactory(AlgorithmFactory):
    """Factory class for creating QubitMapper instances."""

    def algorithm_type_name(self) -> str:
        """Return ``qubit_mapper`` as the algorithm type name."""
        return "qubit_mapper"

    def default_algorithm_name(self) -> str:
        """Return ``qdk`` as the default algorithm name."""
        return "qdk"
