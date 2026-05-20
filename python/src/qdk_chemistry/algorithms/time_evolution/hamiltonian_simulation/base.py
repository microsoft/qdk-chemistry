"""QDK/Chemistry Hamiltonian simulation abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
    Settings,
    TimeDependentQubitHamiltonian,
    UnitaryRepresentation,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["HamiltonianSimulation", "HamiltonianSimulationFactory", "HamiltonianSimulationSettings"]


class HamiltonianSimulationSettings(Settings):
    """Settings for Hamiltonian simulation."""

    def __init__(self):
        """Initialize defaults for Hamiltonian simulation."""
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
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
            "Circuit executor used to run quantum circuits.",
        )
        self._set_default(
            "observable_estimator",
            "algorithm_ref",
            AlgorithmRef("energy_estimator", "qdk"),
            "Estimator used to compute observable expectation values.",
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
        self._set_default(
            "propagator",
            "algorithm_ref",
            AlgorithmRef("propagator", "magnus"),
            "Propagator used to evaluate the effective Hamiltonian over each dt interval.",
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
        observables: list[QubitHamiltonian],
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

    def _create_time_step_evolution(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time: float,
    ) -> UnitaryRepresentation:
        """Create the time-evolution unitary for current settings.

        Raises:
            TypeError: If the evolution builder is not a ``TimeEvolutionBuilder``.

        """
        evolution_builder = self._create_nested("evolution_builder")
        if not isinstance(evolution_builder, TimeEvolutionBuilder):
            raise TypeError(
                f"evolution_builder must be a TimeEvolutionBuilder, got {type(evolution_builder).__name__}."
            )
        evolution_builder.settings().set("time", time)
        return evolution_builder.run(qubit_hamiltonian)

    def _map_time_evolution_to_circuit(
        self,
        evolution: UnitaryRepresentation,
    ) -> Circuit:
        """Map a time-evolution unitary into an executable circuit."""
        circuit_mapper = self._create_nested("circuit_mapper")
        return circuit_mapper.run(evolution)

    def _prepend_state_prep_circuit(self, state_prep: Circuit, circuit: Circuit, num_qubits: int) -> Circuit:
        state_prep_op = state_prep._qsharp_op  # noqa: SLF001
        circuit_op = circuit._qsharp_op  # noqa: SLF001
        if state_prep_op is None or circuit_op is None:
            raise RuntimeError("State-preparation circuit composition requires Q# operations on both circuits.")

        if state_prep.encoding is not None and circuit.encoding is not None and state_prep.encoding != circuit.encoding:
            raise ValueError(
                "State-preparation circuit and evolution circuit use different encodings "
                f"('{state_prep.encoding}' and '{circuit.encoding}')."
            )

        target_indices = list(range(num_qubits))
        combined_encoding = circuit.encoding if circuit.encoding is not None else state_prep.encoding
        sequential_parameters = {
            "first": state_prep_op,
            "second": circuit_op,
            "targets": target_indices,
        }

        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.CircuitComposition.MakeSequentialCircuit,
                parameter=sequential_parameters,
            ),
            qsharp_op=QSHARP_UTILS.CircuitComposition.MakeSequentialOp(state_prep_op, circuit_op),
            encoding=combined_encoding,
        )

    def _measure_observable(
        self,
        circuit: Circuit,
        observable: QubitHamiltonian,
        shots: int = 1000,
        noise: QuantumErrorProfile | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Measure a qubit observable on the provided circuit state."""
        energy_estimator = self._create_nested("observable_estimator")
        # Propagate this algorithm's circuit_executor setting to the energy estimator
        energy_estimator.settings().set("circuit_executor", self._settings.get("circuit_executor"))
        energy_result, measurement_data = energy_estimator.run(
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
