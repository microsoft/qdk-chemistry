"""QDK/Chemistry evolve-and-measure abstractions and utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re
from abc import abstractmethod

from qdk import qsharp

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.algorithms.energy_estimator.energy_estimator import EnergyEstimator
from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.circuit_mapper.base import EvolutionCircuitMapper
from qdk_chemistry.data import (
    Circuit,
    EnergyExpectationResult,
    MeasurementData,
    QuantumErrorProfile,
    QubitHamiltonian,
    Settings,
    TimeEvolutionUnitary,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["MeasureSimulation", "MeasureSimulationFactory", "MeasureSimulationSettings"]

_QASM_QUBIT_DECLARATION_PATTERN = re.compile(r"\bqubit\s*\[(\d+)\]")
_QIR_REQUIRED_NUM_QUBITS_PATTERN = re.compile(r'"required_num_qubits"="(\d+)"')


class MeasureSimulationSettings(Settings):
    """Settings for evolve-and-measure simulation."""

    def __init__(self):
        """Initialize defaults for evolve-and-measure simulation."""
        super().__init__()


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
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: EvolutionCircuitMapper,
        shots: int = 1000,
        circuit_executor: CircuitExecutor,
        energy_estimator: EnergyEstimator,
        noise: QuantumErrorProfile | None = None,
        basis_gates: list[str] | None = None,
    ) -> list[tuple[EnergyExpectationResult, MeasurementData]]:
        """Run evolve-and-measure simulation.

        Args:
            qubit_hamiltonians: List of Hamiltonians used to build time evolution.
            times: List of times to evolve under the Hamiltonians.
            observables: List of observable Hamiltonians to measure after evolution.
            state_prep: Optional circuit that prepares the initial state before time evolution.
            evolution_builder: Time-evolution builder.
            circuit_mapper: Mapper for time-evolution unitary to circuit.
            shots: Number of shots to use for measurement.
            circuit_executor: Circuit executor backend.
            energy_estimator: Energy estimator algorithm.
            noise: Optional noise profile.
            basis_gates: Optional list of basis gates to transpile the circuit into before execution.

        Returns:
            A list of tuples containing ``EnergyExpectationResult`` and ``MeasurementData`` objects.

        """

    def _create_time_evolution(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time: float,
        evolution_builder: TimeEvolutionBuilder,
    ) -> TimeEvolutionUnitary:
        """Create the time-evolution unitary for current settings."""
        return evolution_builder.run(qubit_hamiltonian, time)

    def _map_time_evolution_to_circuit(
        self,
        evolution: TimeEvolutionUnitary,
        circuit_mapper: EvolutionCircuitMapper,
    ) -> Circuit:
        """Map a time-evolution unitary into an executable circuit."""
        return circuit_mapper.run(evolution)

    def _prepend_state_prep_circuit(self, state_prep: Circuit, circuit: Circuit) -> Circuit:
        state_prep_op = getattr(state_prep, "_qsharp_op", None)
        circuit_op = getattr(circuit, "_qsharp_op", None)
        if state_prep_op is None or circuit_op is None:
            raise RuntimeError("State-preparation circuit composition requires Q# operations on both circuits.")

        if state_prep.encoding is not None and circuit.encoding is not None and state_prep.encoding != circuit.encoding:
            raise ValueError(
                "State-preparation circuit and evolution circuit use different encodings "
                f"('{state_prep.encoding}' and '{circuit.encoding}')."
            )

        num_qubits = None
        for representation in (state_prep.qir, circuit.qir, state_prep.qasm, circuit.qasm):
            if representation is None:
                continue

            match = _QIR_REQUIRED_NUM_QUBITS_PATTERN.search(str(representation))
            if match:
                candidate = int(match.group(1))
            else:
                qasm_match = _QASM_QUBIT_DECLARATION_PATTERN.search(str(representation))
                if not qasm_match:
                    continue
                candidate = int(qasm_match.group(1))

            if num_qubits is None:
                num_qubits = candidate
            elif num_qubits != candidate:
                raise ValueError(
                    "State-preparation circuit and evolution circuit must act on the same number of qubits "
                    f"(received {num_qubits} and {candidate})."
                )

        if num_qubits is None:
            raise RuntimeError("Unable to infer the number of qubits needed to compose the Q# circuits.")

        target_indices = list(range(num_qubits))
        combined_encoding = circuit.encoding if circuit.encoding is not None else state_prep.encoding
        qsharp_circuit = qsharp.circuit(
            QSHARP_UTILS.CircuitComposition.MakeSequentialCircuit,
            state_prep_op,
            circuit_op,
            target_indices,
        )
        qir = qsharp.compile(
            QSHARP_UTILS.CircuitComposition.MakeSequentialCircuit,
            state_prep_op,
            circuit_op,
            target_indices,
        )

        return Circuit(
            qsharp=qsharp_circuit,
            qir=qir,
            qsharp_op=QSHARP_UTILS.CircuitComposition.MakeSequentialOp(state_prep_op, circuit_op),
            encoding=combined_encoding,
        )

    @staticmethod
    def _transpile_to_basis_gates(circuit: Circuit, basis_gates: list[str]) -> Circuit:
        """Transpile a Circuit to a target basis gate set using the qdk-chemistry transpiler.

        Args:
            circuit: The circuit to transpile.
            basis_gates: Target basis gates (e.g. ``["cx", "rz", "h", "x"]``).

        Returns:
            A new ``Circuit`` restricted to the requested basis gates.

        """
        try:
            from qiskit import qasm3, transpile  # noqa: PLC0415
            from qiskit.transpiler import PassManager  # noqa: PLC0415

            from qdk_chemistry.plugins.qiskit._interop.transpiler import (  # noqa: PLC0415
                MergeZBasisRotations,
                RemoveZBasisOnZeroState,
                SubstituteCliffordRz,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Qiskit is required to transpile circuits to the requested basis_gates, "
                "but it is not installed. Please install the 'qiskit' package to use "
                "the basis_gates option."
            ) from exc

        qc = circuit.get_qiskit_circuit()
        qc = transpile(qc, basis_gates=basis_gates, optimization_level=3)

        pm = PassManager(
            [
                MergeZBasisRotations(),
                SubstituteCliffordRz(),
                RemoveZBasisOnZeroState(),
            ]
        )
        qc = pm.run(qc)

        return Circuit(qasm=qasm3.dumps(qc), encoding=circuit.encoding)

    def _measure_observable(
        self,
        circuit: Circuit,
        observable: QubitHamiltonian,
        circuit_executor: CircuitExecutor,
        energy_estimator: EnergyEstimator,
        shots: int = 1000,
        noise: QuantumErrorProfile | None = None,
    ) -> tuple[EnergyExpectationResult, MeasurementData]:
        """Measure a qubit observable on the provided circuit state."""
        energy_result, measurement_data = energy_estimator.run(
            circuit,
            observable,
            circuit_executor,
            total_shots=shots,
            noise_model=noise,
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
        """Return classical sampling as the default algorithm name."""
        return "classical_sampling"
