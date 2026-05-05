"""QDK/Chemistry phase estimation abstractions and utilities."""

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
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
    Settings,
)

__all__: list[str] = ["PhaseEstimation", "PhaseEstimationFactory", "PhaseEstimationSettings"]


class PhaseEstimationSettings(Settings):
    """Settings for the Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the settings for Phase Estimation.

        Includes nested algorithm references for the evolution builder,
        circuit mapper, and circuit executor.

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
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )


class PhaseEstimation(Algorithm):
    """Abstract base class for phase estimation algorithms."""

    def __init__(self, num_bits: int = -1):
        """Initialize the PhaseEstimation with default settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.

        """
        super().__init__()
        self._settings = PhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)

    def type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        r"""Run the phase estimation algorithm with the given state preparation circuit and qubit Hamiltonian.

        This method implements the quantum phase estimation procedure:
        1. The state preparation circuit initializes the system in the desired quantum state.
        2. The unitary_builder constructs a unitary from the qubit Hamiltonian.
        3. The circuit_mapper transforms the unitary into controlled-U operations,
           where the control qubits are ancilla qubits used for phase readout.
        4. The circuit_executor runs the resulting quantum circuits on the target backend.
        5. Measurement results are processed to extract the eigenvalue phase estimates.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the estimated phases and associated metadata.

        """

    @cached_property
    def unitary_builder(self):
        """The nested unitary builder algorithm instance."""
        return self._create_nested("unitary_builder")

    def _create_controlled_circuit(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        power: int,
    ) -> tuple[Circuit, int]:
        r"""Create the controlled circuit for the given Hamiltonian and power.

        Sets the ``power`` on the unitary builder so it produces :math:`U^{\\text{power}}`
        according to its ``power_strategy``, then maps the result to a controlled circuit.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to evolve under.
            power: The power to which the unitary should be raised.

        Returns:
            A tuple of (circuit, num_qubits) where circuit is the controlled circuit
            implementing controlled-:math:`U^{\\text{power}}` and num_qubits is the total
            number of qubits required by the unitary (including any ancilla).

        """
        unitary_builder = self._create_nested("unitary_builder")
        unitary_builder.settings().update("power", power)
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        num_qubits = unitary_rep.get_container().num_qubits
        controlled_unitary = ControlledUnitary(unitary=unitary_rep, control_indices=[0])
        circuit_mapper = self._create_nested("circuit_mapper")
        return circuit_mapper.run(controlled_unitary=controlled_unitary), num_qubits


class PhaseEstimationFactory(AlgorithmFactory):
    """Factory class for creating PhaseEstimation instances."""

    def __init__(self):
        """Initialize the PhaseEstimationFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as phase_estimation."""
        return "phase_estimation"

    def default_algorithm_name(self) -> str:
        """Return the iterative as default algorithm name."""
        return "iterative"
