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

    from qdk_chemistry.data.qubit_operator import QubitOperator

__all__: list[str] = ["DrivenContainer"]


class DrivenContainer(TimeDependentQubitHamiltonianContainer):
    """Container for a driven Hamiltonian H(t) = H0 + f(t) * H1.

    The Hamiltonian is split into a time-independent part *H0* and a
    time-dependent part *H1* whose coefficient is modulated by a scalar
    drive function *f(t)*. Without a drive, the Hamiltonian is the static *H0*.

    Args:
        base_hamiltonian: Time-independent qubit operator.
        drive_hamiltonian: Driven qubit operator (modulated by *drive*), or None for a static Hamiltonian.
        drive: Scalar function f(t) that modulates *drive_hamiltonian*, or None for a static Hamiltonian.

    Raises:
        ValueError: If only one of *drive_hamiltonian* and *drive* is given, or the operators' qubit counts differ.

    """

    def __init__(
        self,
        base_hamiltonian: QubitOperator,
        drive_hamiltonian: QubitOperator | None = None,
        drive: Callable[[float], float] | None = None,
    ) -> None:
        """Initialize the driven container."""
        if (drive_hamiltonian is None) != (drive is None):
            raise ValueError("drive_hamiltonian and drive must be given together.")
        if drive_hamiltonian is not None and base_hamiltonian.num_qubits != drive_hamiltonian.num_qubits:
            raise ValueError("base_hamiltonian and drive_hamiltonian must have the same number of qubits.")

        self._base_hamiltonian = base_hamiltonian
        self._drive_hamiltonian = drive_hamiltonian
        self._drive = drive

    def evaluate(self, t: float) -> QubitOperator:
        """Return H0 + f(t) * H1 at time *t*.

        Args:
            t: The time at which to evaluate the Hamiltonian.

        Returns:
            The qubit operator at the given time, or H0 itself when there is no drive.

        """
        if self._drive is None or self._drive_hamiltonian is None:
            return self._base_hamiltonian
        return self._base_hamiltonian + self._drive(t) * self._drive_hamiltonian

    @property
    def base_hamiltonian(self) -> QubitOperator:
        """The time-independent Hamiltonian."""
        return self._base_hamiltonian

    @property
    def drive_hamiltonian(self) -> QubitOperator | None:
        """The driven Hamiltonian (modulated by the drive), or None for a static Hamiltonian."""
        return self._drive_hamiltonian

    @property
    def drive(self) -> Callable[[float], float] | None:
        """The scalar drive function f(t), or None for a static Hamiltonian."""
        return self._drive

    @property
    def num_qubits(self) -> int:
        """Number of qubits (uniform across all times)."""
        return self._base_hamiltonian.num_qubits
