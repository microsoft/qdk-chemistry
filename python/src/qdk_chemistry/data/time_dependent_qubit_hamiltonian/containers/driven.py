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
        h0: Time-independent qubit Hamiltonian.
        h1: Time-dependent qubit Hamiltonian (modulated by *drive*).
        drive: Scalar function f(t) that modulates *h1*.

    Raises:
        ValueError: If *h0* and *h1* have different qubit counts.

    """

    def __init__(
        self,
        h0: QubitHamiltonian,
        h1: QubitHamiltonian,
        drive: Callable[[float], float],
    ) -> None:
        """Initialize the driven container."""
        if h0.num_qubits != h1.num_qubits:
            raise ValueError("h0 and h1 must have the same number of qubits.")

        self._h0 = h0
        self._h1 = h1
        self._drive = drive

    def evaluate(self, t: float) -> QubitHamiltonian:
        """Return H0 + f(t) * H1 at time *t*.

        Args:
            t: The time at which to evaluate the Hamiltonian.

        Returns:
            The qubit Hamiltonian at the given time.

        """
        return self._h0 + self._drive(t) * self._h1

    @property
    def h0(self) -> QubitHamiltonian:
        """The time-independent Hamiltonian."""
        return self._h0

    @property
    def h1(self) -> QubitHamiltonian:
        """The time-dependent Hamiltonian (modulated by the drive)."""
        return self._h1

    @property
    def drive(self) -> Callable[[float], float]:
        """The scalar drive function f(t)."""
        return self._drive

    @property
    def num_qubits(self) -> int:
        """Number of qubits (uniform across all times)."""
        return self._h0.num_qubits
