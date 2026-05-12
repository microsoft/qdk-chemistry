"""Piecewise-constant time-dependent qubit Hamiltonian."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from .base import TimeDependentQubitHamiltonian

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

__all__: list[str] = []


class PiecewiseConstantQubitHamiltonian(TimeDependentQubitHamiltonian):
    """An immutable sequence of Hamiltonians at monotonically increasing times.

    Each time interval is associated with a fixed
    :class:`~qdk_chemistry.data.QubitHamiltonian`.  This is the simplest
    concrete implementation of :class:`TimeDependentQubitHamiltonian`.

    Args:
        hamiltonians: Non-empty list of qubit Hamiltonians, one per time step.
        times: Strictly monotonically increasing list of evolution times, same length as ``hamiltonians``.

    Raises:
        ValueError: If inputs are empty, lengths differ, times are not strictly increasing, or qubit counts differ.

    """

    def __init__(self, hamiltonians: list[QubitHamiltonian], times: list[float]) -> None:
        """Initialize a piecewise-constant time-dependent qubit Hamiltonian.

        Args:
            hamiltonians: Non-empty list of qubit Hamiltonians, one per time step.
            times: Strictly monotonically increasing list of evolution times, same length as ``hamiltonians``.

        """
        if not hamiltonians:
            raise ValueError("hamiltonians must not be empty.")
        if not times:
            raise ValueError("times must not be empty.")
        if len(hamiltonians) != len(times):
            raise ValueError("hamiltonians and times must have the same length.")
        if not all(a < b for a, b in itertools.pairwise(times)):
            raise ValueError("times must be strictly monotonically increasing.")

        ref_nq = hamiltonians[0].num_qubits
        for h in hamiltonians[1:]:
            if h.num_qubits != ref_nq:
                raise ValueError("All Hamiltonians must have the same number of qubits.")

        self._hamiltonians = list(hamiltonians)
        self._times = list(times)

    def evaluate(self, t: float) -> QubitHamiltonian:
        """Return the Hamiltonian snapshot at time *t*.

        For a piecewise-constant schedule, *t* must exactly match one of
        the stored time coordinates.

        Args:
            t: The time at which to evaluate the Hamiltonian.

        Returns:
            The qubit Hamiltonian at the given time.

        Raises:
            ValueError: If *t* does not match any stored time coordinate.

        """
        try:
            idx = self._times.index(t)
        except ValueError:
            raise ValueError(f"Time {t} not found in schedule {self._times}.") from None
        return self._hamiltonians[idx]

    @property
    def schedule(self) -> list[float]:
        """Ordered time points at which evolution steps are taken."""
        return list(self._times)

    @property
    def hamiltonians(self) -> list[QubitHamiltonian]:
        """The Hamiltonian snapshots."""
        return self._hamiltonians

    @property
    def times(self) -> list[float]:
        """The evolution time coordinates."""
        return self._times

    @property
    def num_qubits(self) -> int:
        """Number of qubits (uniform across all snapshots)."""
        return self._hamiltonians[0].num_qubits
