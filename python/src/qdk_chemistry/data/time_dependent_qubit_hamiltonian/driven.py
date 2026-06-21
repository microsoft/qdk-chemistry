"""Driven time-dependent qubit Hamiltonian."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import TimeDependentQubitHamiltonian
from .containers.driven import DrivenContainer

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

__all__: list[str] = []


class DrivenQubitHamiltonian(TimeDependentQubitHamiltonian):
    """A driven Hamiltonian H(t) = H0 + f(t) * H1.

    Convenience subclass of :class:`TimeDependentQubitHamiltonian` that
    creates a :class:`~qdk_chemistry.data.time_dependent_qubit_hamiltonian.containers.driven.DrivenContainer`
    under the hood.

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
        """Initialize a driven time-dependent qubit Hamiltonian.

        Args:
            h0: Time-independent qubit Hamiltonian.
            h1: Time-dependent qubit Hamiltonian (modulated by *drive*).
            drive: Scalar function f(t) that modulates *h1*.

        """
        super().__init__(DrivenContainer(h0, h1, drive))

    @property
    def h0(self) -> QubitHamiltonian:
        """The time-independent Hamiltonian."""
        container: DrivenContainer = self.get_container()  # type: ignore[assignment]
        return container.base_hamiltonian

    @property
    def h1(self) -> QubitHamiltonian:
        """The time-dependent Hamiltonian (modulated by the drive)."""
        container: DrivenContainer = self.get_container()  # type: ignore[assignment]
        return container.drive_hamiltonian

    @property
    def drive(self) -> Callable[[float], float]:
        """The scalar drive function f(t)."""
        container: DrivenContainer = self.get_container()  # type: ignore[assignment]
        return container.drive
