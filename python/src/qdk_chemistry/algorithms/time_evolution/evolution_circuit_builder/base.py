"""Base classes for evolution circuit builders."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    Settings,
    TimeDependentQubitHamiltonian,
)

__all__: list[str] = [
    "EvolutionCircuitBuilder",
    "EvolutionCircuitBuilderFactory",
    "EvolutionCircuitBuilderSettings",
]


class EvolutionCircuitBuilderSettings(Settings):
    """Settings for the evolution circuit builder."""

    def __init__(self):
        """Initialize defaults for evolution circuit builder."""
        super().__init__()
        self._set_default(
            "evolution_builder",
            "algorithm_ref",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
            "Time evolution builder used to construct the unitary.",
        )
        self._set_default(
            "circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("circuit_mapper", "pauli_sequence"),
            "Circuit mapper used to convert the unitary to a circuit.",
        )
        self._set_default(
            "propagator",
            "algorithm_ref",
            AlgorithmRef("propagator", "magnus"),
            "Propagator used to evaluate the effective Hamiltonian over each dt interval.",
        )
        self._set_default(
            "total_time",
            "float",
            1.0,
            "Total evolution time.",
        )
        self._set_default(
            "dt",
            "float",
            0.0,
            "Time step for time-dependent evolution. Each step is passed to the builder.",
        )


class EvolutionCircuitBuilder(Algorithm):
    """Abstract base class for evolution circuit builders.

    An evolution circuit builder constructs a quantum circuit for
    time evolution without executing it.
    """

    def __init__(self):
        """Initialize the evolution circuit builder."""
        super().__init__()
        self._settings = EvolutionCircuitBuilderSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as evolution_circuit_builder."""
        return "evolution_circuit_builder"

    @abstractmethod
    def _run_impl(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        state_prep: Circuit,
    ) -> Circuit:
        """Build the evolution circuit.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            state_prep: Circuit that prepares the initial state.

        Returns:
            The combined state-prep + evolution circuit.

        """


class EvolutionCircuitBuilderFactory(AlgorithmFactory):
    """Factory class for creating evolution circuit builder instances."""

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as evolution_circuit_builder."""
        return "evolution_circuit_builder"

    def default_algorithm_name(self) -> str:
        """Return euler as the default algorithm name."""
        return "euler"
