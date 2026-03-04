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

__all__: list[str] = []


class QubitMapperSettings(Settings):
    """Base settings for all QubitMapper implementations.

    Common settings:
        encoding (string, default="jordan-wigner"): Fermion-to-qubit encoding strategy.

    """

    def __init__(self, valid_encodings: list[str] | None = None) -> None:
        """Initialize QubitMapperSettings.

        Args:
            valid_encodings: Allowed encoding values. Default: ``["jordan-wigner"]``.

        """
        super().__init__()
        if valid_encodings is None:
            valid_encodings = ["jordan-wigner"]
        self._set_default(
            "encoding",
            "string",
            "jordan-wigner",
            "Fermion-to-qubit encoding strategy",
            valid_encodings,
        )


class QubitMapper(Algorithm):
    """Abstract base class for mapping a Hamiltonian to a QubitHamiltonian."""

    def __init__(self):
        """Initialize the QubitMapper."""
        super().__init__()

    def type_name(self) -> str:
        """Return ``qubit_mapper`` as the algorithm type name."""
        return "qubit_mapper"

    @abstractmethod
    def _run_impl(self, hamiltonian: Hamiltonian, symmetries: Symmetries | None = None) -> QubitHamiltonian:
        """Construct a QubitHamiltonian from a Hamiltonian using the mapping specified.

        Args:
            hamiltonian: The fermionic Hamiltonian.
            symmetries: Optional conserved quantum numbers. Required by symmetry-exploiting encodings.

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
