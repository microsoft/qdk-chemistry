"""Driven time-dependent qubit Hamiltonian container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TimeDependentQubitHamiltonianContainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

__all__: list[str] = ["DrivenContainer"]


class DrivenContainer(TimeDependentQubitHamiltonianContainer):
    """Container for a driven Hamiltonian H(t) = H0 + f(t) * H1.

    The Hamiltonian is split into a time-independent part *H0* and a
    time-dependent part *H1* whose coefficient is modulated by a scalar
    drive function *f(t)*.

    Args:
        base_hamiltonian: Time-independent qubit Hamiltonian.
        drive_hamiltonian: Driven qubit Hamiltonian (modulated by *drive*).
        drive: Scalar function f(t) that modulates *drive_hamiltonian*.

    Raises:
        ValueError: If *base_hamiltonian* and *drive_hamiltonian* have different qubit counts.

    """

    def __init__(
        self,
        base_hamiltonian: QubitHamiltonian,
        drive_hamiltonian: QubitHamiltonian,
        drive: Callable[[float], float],
    ) -> None:
        """Initialize the driven container."""
        if base_hamiltonian.num_qubits != drive_hamiltonian.num_qubits:
            raise ValueError("base_hamiltonian and drive_hamiltonian must have the same number of qubits.")

        self._base_hamiltonian = base_hamiltonian
        self._drive_hamiltonian = drive_hamiltonian
        self._drive = drive

    def evaluate(self, t: float) -> QubitHamiltonian:
        """Return H0 + f(t) * H1 at time *t*.

        Args:
            t: The time at which to evaluate the Hamiltonian.

        Returns:
            The qubit Hamiltonian at the given time.

        """
        return self._base_hamiltonian + self._drive(t) * self._drive_hamiltonian

    @property
    def base_hamiltonian(self) -> QubitHamiltonian:
        """The time-independent Hamiltonian."""
        return self._base_hamiltonian

    @property
    def drive_hamiltonian(self) -> QubitHamiltonian:
        """The driven Hamiltonian (modulated by the drive)."""
        return self._drive_hamiltonian

    @property
    def drive(self) -> Callable[[float], float]:
        """The scalar drive function f(t)."""
        return self._drive

    @property
    def num_qubits(self) -> int:
        """Number of qubits (uniform across all times)."""
        return self._base_hamiltonian.num_qubits
