"""Time-dependent qubit Hamiltonian abstract base class."""

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


class TimeDependentQubitHamiltonian:
    """Abstract base for time-dependent qubit Hamiltonians.

    Subclasses must implement :meth:`evaluate` and :attr:`schedule` to
    define how the Hamiltonian varies with time.  The base class provides
    :meth:`to_snapshots` for algorithms that need materialized lists and
    default implementations of ``__len__`` and ``__repr__``.

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
    def schedule(self) -> list[float]:
        """Ordered time points at which evolution steps are taken."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits (uniform across all times)."""

    def to_snapshots(self) -> tuple[list[QubitHamiltonian], list[float]]:
        """Materialize the schedule into explicit ``(hamiltonians, times)`` lists.

        Returns:
            A tuple of ``(hamiltonians, times)`` evaluated at each scheduled time point.

        """
        return [self.evaluate(t) for t in self.schedule], list(self.schedule)

    def __len__(self) -> int:
        """Return the number of time steps."""
        return len(self.schedule)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{type(self).__name__}(steps={len(self)}, num_qubits={self.num_qubits})"
