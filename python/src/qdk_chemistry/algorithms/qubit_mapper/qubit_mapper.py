"""QDK/Chemistry qubit mapper abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Settings

if TYPE_CHECKING:  # Only needed for type annotations; avoid importing into module namespace
    from qdk_chemistry.data import Hamiltonian, QubitHamiltonian, Symmetries
    from qdk_chemistry.data.majorana_mapping import MajoranaMapping

__all__: list[str] = []


class QubitMapperSettings(Settings):
    """Base settings for all QubitMapper implementations.

    Settings are variant-specific (thresholds, etc.). The encoding is
    determined by the :class:`~qdk_chemistry.data.MajoranaMapping` passed
    to :meth:`~QubitMapper.run`.

    """

    def __init__(self) -> None:
        """Initialize QubitMapperSettings."""
        super().__init__()


class QubitMapper(Algorithm):
    """Abstract base class for mapping a Hamiltonian to a QubitHamiltonian."""

    def __init__(self):
        """Initialize the QubitMapper."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``qubit_mapper`` as the algorithm type name."""
        return "qubit_mapper"

    @abstractmethod
    def _run_impl(
        self,
        hamiltonian: Hamiltonian,
        mapping: MajoranaMapping,
        symmetries: Symmetries | None = None,
    ) -> QubitHamiltonian:
        """Construct a QubitHamiltonian from a Hamiltonian using the given mapping.

        Args:
            hamiltonian: The fermionic Hamiltonian.
            mapping: The Majorana-to-Pauli encoding to use.
            symmetries: Optional symmetry information. Required by symmetry-exploiting algorithms.

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
