"""Time-dependent qubit Hamiltonian wrapper class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

    from .containers.base import TimeDependentQubitHamiltonianContainer

__all__: list[str] = ["TimeDependentQubitHamiltonian"]


class TimeDependentQubitHamiltonian:
    """Wrapper for time-dependent qubit Hamiltonians.

    Delegates storage and evaluation to a
    :class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.base.TimeDependentQubitHamiltonianContainer`.

    Args:
        container: The container that stores and evaluates the time-dependent Hamiltonian.

    """

    def __init__(self, container: TimeDependentQubitHamiltonianContainer) -> None:
        """Initialize with the given container."""
        self._container = container

    def get_container(self) -> TimeDependentQubitHamiltonianContainer:
        """Return the underlying container.

        Returns:
            The time-dependent qubit Hamiltonian container.

        """
        return self._container

    def evaluate(self, t: float) -> QubitHamiltonian:
        """Return the qubit Hamiltonian at time *t*.

        Args:
            t: The time at which to evaluate the Hamiltonian.

        Returns:
            The qubit Hamiltonian at the given time.

        """
        return self._container.evaluate(t)

    @property
    def num_qubits(self) -> int:
        """Number of qubits (uniform across all times)."""
        return self._container.num_qubits

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{type(self).__name__}(num_qubits={self.num_qubits})"
