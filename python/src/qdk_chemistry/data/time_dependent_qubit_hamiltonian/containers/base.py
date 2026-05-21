"""Time-dependent qubit Hamiltonian container abstract base class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

__all__: list[str] = []


class TimeDependentQubitHamiltonianContainer:
    """Abstract base for time-dependent qubit Hamiltonian storage.

    Concrete containers define how a time-dependent Hamiltonian is stored
    and evaluated.  :class:`~qdk_chemistry.data.TimeDependentQubitHamiltonian`
    delegates to a container instance.

    """

    @abstractmethod
    def evaluate(self, t: float) -> QubitHamiltonian:
        """Return the qubit Hamiltonian at time *t*.

        Args:
            t: The time at which to evaluate the Hamiltonian.

        Returns:
            The qubit Hamiltonian at the given time.

        """

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits (uniform across all times)."""
