"""Base classes for phase estimation algorithms in QDK/Chemistry.

This module provides standalone abstractions for phase estimation workflows.
Concrete algorithms should inherit from :class:`PhaseEstimation` and expose
creation helpers (for example, ``create_circuit`` or ``create_iterations``) that
assemble the circuits required by the chosen phase estimation variant.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC
from enum import StrEnum
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qiskit import QuantumCircuit

    from qdk_chemistry.data import QubitHamiltonian

AlgorithmT = TypeVar("AlgorithmT", bound="PhaseEstimation")

__all__: list[str] = []


class PhaseEstimationAlgorithm(StrEnum):
    """Enumeration of supported phase estimation routines.

    References:
        * Iterative QPE: Kitaev, A. (1995). "Quantum measurements and the Abelian
            Stabilizer Problem." arXiv:quant-ph/9511026. https://arxiv.org/abs/quant-ph/9511026
        * Traditional QPE: Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation
            and Quantum Information" (10th Anniversary Edition), Ch. 5.2.

    """

    ITERATIVE = "iterative"
    TRADITIONAL = "traditional"


class PhaseEstimation(ABC):  # noqa: B024
    """Abstract interface for phase estimation strategies."""

    algorithm: PhaseEstimationAlgorithm | None = None

    def __init__(self, hamiltonian: QubitHamiltonian, evolution_time: float):
        """Store common data for phase estimation routines.

        Args:
            hamiltonian: Target Hamiltonian whose eigenvalues are estimated.
            evolution_time: Time parameter ``t`` used in the time-evolution unitary ``U = exp(-i H t)``.

        """
        self._hamiltonian = hamiltonian
        self._evolution_time = evolution_time

    @property
    def hamiltonian(self) -> QubitHamiltonian:
        """Return the Hamiltonian used by the algorithm."""
        return self._hamiltonian

    @property
    def evolution_time(self) -> float:
        """Return the evolution time used for ``U = exp(-i H t)``."""
        return self._evolution_time

    @classmethod
    def from_algorithm(
        cls: type[AlgorithmT],
        algorithm: PhaseEstimationAlgorithm | str | None,
        *,
        hamiltonian: QubitHamiltonian,
        evolution_time: float,
        **kwargs,
    ) -> AlgorithmT:
        """Factory method returning the requested phase estimation strategy.

        Args:
            algorithm: Identifier for the desired algorithm.

                ``None`` selects :class:`PhaseEstimationAlgorithm.ITERATIVE`.

            hamiltonian: Target Hamiltonian.
            evolution_time: Time parameter ``t`` for ``U = exp(-i H t)``.
            kwargs: Additional options forwarded to the algorithm constructor.

        Raises:
            ValueError: If the requested algorithm is not registered.

        """
        normalized_algorithm = cls._normalize_algorithm(algorithm)

        for subclass in cls._iter_subclasses():
            if subclass.algorithm == normalized_algorithm:
                return cast("AlgorithmT", subclass(hamiltonian, evolution_time, **kwargs))

        raise ValueError(f"Phase estimation algorithm {normalized_algorithm.value} is not implemented.")

    @staticmethod
    def _normalize_algorithm(algorithm: PhaseEstimationAlgorithm | str | None) -> PhaseEstimationAlgorithm:
        if algorithm is None:
            return PhaseEstimationAlgorithm.ITERATIVE
        if isinstance(algorithm, PhaseEstimationAlgorithm):
            return algorithm
        try:
            return PhaseEstimationAlgorithm(algorithm.lower())
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unrecognized phase estimation algorithm '{algorithm}'.") from exc

    @classmethod
    def _iter_subclasses(cls) -> Iterable[type[PhaseEstimation]]:
        for subclass in cls.__subclasses__():
            yield subclass
            yield from subclass._iter_subclasses()  # noqa: SLF001

    def _validate_state_prep_qubits(self, state_prep: QuantumCircuit) -> None:
        """Ensure ``state_prep`` matches the Hamiltonian system size."""
        if state_prep.num_qubits != self.hamiltonian.num_qubits:
            raise ValueError(
                "state_prep must prepare the same number of system qubits as the Hamiltonian "
                f"(expected {self.hamiltonian.num_qubits}, received {state_prep.num_qubits}).",
            )
