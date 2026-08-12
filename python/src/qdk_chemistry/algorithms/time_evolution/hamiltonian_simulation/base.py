"""QDK/Chemistry Hamiltonian simulation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitOperator,
    Settings,
    TimeDependentQubitHamiltonian,
)

__all__: list[str] = ["HamiltonianSimulation", "HamiltonianSimulationFactory", "HamiltonianSimulationSettings"]


class HamiltonianSimulationSettings(Settings):
    """Settings for Hamiltonian simulation."""

    def __init__(self):
        """Initialize defaults for Hamiltonian simulation."""
        super().__init__()
        self._set_default(
            "evolution_circuit_builder",
            "algorithm_ref",
            AlgorithmRef("evolution_circuit_builder", "euler"),
            "Evolution circuit builder used to construct the state-prep + evolution circuit.",
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
            "Circuit executor used to run quantum circuits.",
        )
        self._set_default(
            "observable_estimator",
            "algorithm_ref",
            AlgorithmRef("expectation_estimator", "qdk"),
            "Estimator used to compute observable expectation values.",
        )


class HamiltonianSimulation(Algorithm):
    """Abstract base class for Hamiltonian evolution and observable measurement."""

    def __init__(self):
        """Initialize the Hamiltonian simulation settings."""
        super().__init__()
        self._settings = HamiltonianSimulationSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as hamiltonian_simulation."""
        return "hamiltonian_simulation"

    @abstractmethod
    def _run_impl(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        observables: list[QubitOperator],
        state_prep: Circuit,
        shots: int = 1000,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> list[tuple[EnergyExpectationResult, MeasurementData]]:
        r"""Run Hamiltonian simulation.

        This method implements a quantum ODE integrator for the
        Schrödinger equation :math:`i\\partial_t U = H(t)\\,U`.  The
        time-dependent Hamiltonian is discretized into steps, each
        step is Trotterized into a quantum circuit, and the resulting
        circuit is executed to measure observable expectation values.

        The evolution builder, circuit mapper, circuit executor, and
        observable estimator are resolved from the algorithm's settings
        via ``AlgorithmRef``.

        Args:
            hamiltonian: Time-dependent Hamiltonian specifying the evolution schedule.
            observables: Observable Hamiltonians to measure after evolution; each returns its own expectation value.
            state_prep: Circuit that prepares the initial state before time evolution.
            shots: Number of measurement shots per observable. Defaults to 1000.
            noise: Optional noise profile.

        Returns:
            A list of tuples containing ``EnergyExpectationResult`` and ``MeasurementData`` objects.

        """

    def _build_evolution_circuit(
        self,
        hamiltonian: TimeDependentQubitHamiltonian,
        state_prep: Circuit,
    ) -> Circuit:
        """Construct the evolution circuit using the configured circuit builder.

        Args:
            hamiltonian: Time-dependent Hamiltonian.
            state_prep: Circuit that prepares the initial state.

        Returns:
            The combined state-prep + evolution circuit.

        """
        from qdk_chemistry.algorithms.time_evolution.evolution_circuit_builder.base import (  # noqa: PLC0415
            EvolutionCircuitBuilder,
        )

        circuit_builder = self._create_nested("evolution_circuit_builder")
        if not isinstance(circuit_builder, EvolutionCircuitBuilder):
            raise TypeError(
                f"evolution_circuit_builder must be an EvolutionCircuitBuilder, got {type(circuit_builder).__name__}."
            )
        return circuit_builder.run(hamiltonian, state_prep)

    def _measure_observable(
        self,
        circuit: Circuit,
        observable: QubitOperator,
        shots: int = 1000,
        noise: QuantumErrorProfile | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Measure a qubit observable on the provided circuit state."""
        expectation_estimator = self._create_nested("observable_estimator")
        # Propagate this algorithm's circuit_executor setting to the energy estimator
        expectation_estimator.settings().set("circuit_executor", self._settings.get("circuit_executor"))
        energy_result, measurement_data = expectation_estimator.run(
            circuit,
            observable,
            total_shots=shots,
            noise_model=noise,
        )
        return energy_result, measurement_data


class HamiltonianSimulationFactory(AlgorithmFactory):
    """Factory class for creating Hamiltonian simulation algorithm instances."""

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as hamiltonian_simulation."""
        return "hamiltonian_simulation"

    def default_algorithm_name(self) -> str:
        """Return euler_integrator as the default algorithm name."""
        return "euler_integrator"
