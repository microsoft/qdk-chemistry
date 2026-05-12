"""Time-dependent Hamiltonian for evolve-and-measure simulations.

A :class:`TimeDependentQubitHamiltonian` bundles a sequence of
:class:`~qdk_chemistry.data.QubitHamiltonian` snapshots with their
corresponding time coordinates.  Validation (length consistency,
strict monotonicity, qubit-count uniformity) is performed at
construction time.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

__all__ = ["TimeDependentQubitHamiltonian"]


class TimeDependentQubitHamiltonian:
    """An immutable sequence of Hamiltonians at monotonically increasing times.

    Args:
        hamiltonians: Non-empty list of qubit Hamiltonians, one per time step.
        times: Strictly monotonically increasing list of evolution times, same length as ``hamiltonians``.

    Raises:
        ValueError: If inputs are empty, lengths differ, times are not strictly increasing, or qubit counts differ.

    """

    def __init__(self, hamiltonians: list[QubitHamiltonian], times: list[float]) -> None:
        """Initialize a time-dependent qubit Hamiltonian.

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

    def __len__(self) -> int:
        """Return the number of time steps."""
        return len(self._hamiltonians)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"TimeDependentQubitHamiltonian(steps={len(self)}, num_qubits={self.num_qubits})"
