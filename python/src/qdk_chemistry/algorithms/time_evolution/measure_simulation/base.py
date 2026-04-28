"""QDK/Chemistry evolve-and-measure abstractions and utilities."""

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
    QubitHamiltonian,
    Settings,
    TimeEvolutionUnitary,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["MeasureSimulation", "MeasureSimulationFactory", "MeasureSimulationSettings"]


class MeasureSimulationSettings(Settings):
    """Settings for evolve-and-measure simulation."""

    def __init__(self):
        """Initialize defaults for evolve-and-measure simulation."""
        super().__init__()
        self._set_default(
            "evolution_builder",
            "algorithm_ref",
            AlgorithmRef("time_evolution_builder", "trotter"),
        )
        self._set_default(
            "circuit_mapper",
            "algorithm_ref",
            AlgorithmRef("evolution_circuit_mapper", "pauli_sequence"),
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
        )
        self._set_default(
            "energy_estimator",
            "algorithm_ref",
            AlgorithmRef("energy_estimator", "qdk"),
        )


class MeasureSimulation(Algorithm):
    """Abstract base class for Hamiltonian evolution and observable measurement."""

    def __init__(self):
        """Initialize the evolve-and-measure simulation settings."""
        super().__init__()
        self._settings = MeasureSimulationSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as measure_simulation."""
        return "measure_simulation"

    @abstractmethod
    def _run_impl(
        self,
        qubit_hamiltonians: list[QubitHamiltonian],
        times: list[float],
        observables: list[QubitHamiltonian],
        *,
        state_prep: Circuit | None = None,
        shots: int = 1000,
        noise: QuantumErrorProfile | None = None,
        device_backend_name: str | None = None,
        pre_transpilation_passes: list[str] | None = None,
        post_transpilation_passes: list[str] | None = None,
    ) -> list[tuple[EnergyExpectationResult, MeasurementData]]:
        """Run evolve-and-measure simulation.

        The evolution builder, circuit mapper, circuit executor, and energy
        estimator are resolved from the algorithm's settings via
        ``AlgorithmRef``.

        Args:
            qubit_hamiltonians: List of Hamiltonians used to build time evolution.
            times: List of times to evolve under the Hamiltonians.
            observables: List of observable Hamiltonians to measure after evolution.
            state_prep: Optional circuit that prepares the initial state before time evolution.
            shots: Number of shots to use for measurement.
            noise: Optional noise profile.
            device_backend_name: Optional device backend name.
            pre_transpilation_passes: Optional list of passes to apply before transpilation.
            post_transpilation_passes: Optional list of passes to apply after transpilation.

        Returns:
            A list of tuples containing ``EnergyExpectationResult`` and ``MeasurementData`` objects.

        """

    def _create_time_evolution(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time: float,
    ) -> TimeEvolutionUnitary:
        """Create the time-evolution unitary for current settings."""
        evolution_builder = self._create_nested("evolution_builder")
        return evolution_builder.run(qubit_hamiltonian, time)

    def _map_time_evolution_to_circuit(
        self,
        evolution: TimeEvolutionUnitary,
    ) -> Circuit:
        """Map a time-evolution unitary into an executable circuit."""
        circuit_mapper = self._create_nested("circuit_mapper")
        return circuit_mapper.run(evolution)

    def _prepend_state_prep_circuit(self, state_prep: Circuit, circuit: Circuit, num_qubits: int) -> Circuit:
        state_prep_op = getattr(state_prep, "_qsharp_op", None)
        circuit_op = getattr(circuit, "_qsharp_op", None)
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
        device_backend_name: str | None = None,
        pre_transpilation_passes: list[str] | None = None,
        post_transpilation_passes: list[str] | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Measure a qubit observable on the provided circuit state."""
        energy_estimator = self._create_nested("energy_estimator")
        # Propagate this algorithm's circuit_executor setting to the energy estimator
        energy_estimator.settings().set("circuit_executor", self._settings.get("circuit_executor"))
        energy_result, measurement_data = energy_estimator.run(
            circuit,
            observable,
            total_shots=shots,
            noise_model=noise,
            device_backend_name=device_backend_name,
            pre_transpilation_passes=pre_transpilation_passes,
            post_transpilation_passes=post_transpilation_passes,
        )
        return energy_result, measurement_data


class MeasureSimulationFactory(AlgorithmFactory):
    """Factory class for creating evolve-and-measure algorithm instances."""

    def __init__(self):
        """Initialize the MeasureSimulationFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as measure_simulation."""
        return "measure_simulation"

    def default_algorithm_name(self) -> str:
        """Return evolve_and_measure as the default algorithm name."""
        return "evolve_and_measure"
