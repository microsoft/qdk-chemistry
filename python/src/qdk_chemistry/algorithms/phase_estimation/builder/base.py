"""QDK/Chemistry phase estimation builder abstractions."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from functools import cached_property

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    ControlledUnitary,
    QubitHamiltonian,
    Settings,
)

__all__: list[str] = [
    "PhaseEstimationBuilder",
    "PhaseEstimationBuilderFactory",
    "PhaseEstimationBuilderSettings",
]


class PhaseEstimationBuilderSettings(Settings):
    """Settings for the Phase Estimation Builder algorithm."""

    def __init__(self):
        """Initialize the settings for the Phase Estimation Builder.

        Includes nested algorithm references for the evolution builder
        and the circuit mapper used to construct phase estimation circuits.

        """
        super().__init__()
        self._set_default("num_bits", "int", -1, "The number of phase bits to estimate.")
        self._set_default(
            "unitary_builder",
            "algorithm_ref",
            AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
        )
        self._set_default(
            "circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        )


class PhaseEstimationBuilder(Algorithm):
    """Abstract base class for phase estimation circuit builders.

    This algorithm constructs the quantum circuits needed for phase estimation
    without executing them. It can be used independently for resource estimation,
    circuit preview, or composed inside a full PhaseEstimation runner.

    """

    def __init__(self, num_bits: int = -1):
        """Initialize the PhaseEstimationBuilder with default settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.

        """
        super().__init__()
        self._settings = PhaseEstimationBuilderSettings()
        self._settings.set("num_bits", num_bits)

    def type_name(self) -> str:
        """Return the algorithm type name as phase_estimation_builder."""
        return "phase_estimation_builder"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[Circuit]:
        """Build all phase estimation iteration circuits.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build circuits.

        Returns:
            A list of quantum circuits, one per phase bit iteration.

        """

    @cached_property
    def unitary_builder(self):
        """The nested unitary builder algorithm instance."""
        return self._create_nested("unitary_builder")

    def _create_controlled_circuit(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        power: int,
    ) -> Circuit:
        r"""Create the controlled circuit for the given Hamiltonian and power.

        Sets the ``power`` on the unitary builder so it produces :math:`U^{\\text{power}}`
        according to its ``power_strategy``, then maps the result to a controlled circuit.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            power: The power to which the unitary should be raised.

        Returns:
            The controlled circuit implementing controlled-:math:`U^{\\text{power}}`.

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_builder.settings().update("power", power)
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])
        circuit_mapper = self._create_nested("circuit_mapper")
        return circuit_mapper.run(controlled_unitary=controlled_unitary)


class PhaseEstimationBuilderFactory(AlgorithmFactory):
    """Factory class for creating PhaseEstimationBuilder instances."""

    def __init__(self):
        """Initialize the PhaseEstimationBuilderFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as phase_estimation_builder."""
        return "phase_estimation_builder"

    def default_algorithm_name(self) -> str:
        """Return the iterative as default algorithm name."""
        return "iterative"
