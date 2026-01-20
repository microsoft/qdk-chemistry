"""Qiskit Aer Simulator circuit executor for QDK/Chemistry.

This module provides a CircuitExecutor implementation that uses Qiskit Aer Simulator
to execute quantum circuits. It accepts QDK/Chemistry Circuit and QuantumErrorProfile
data classes and returns measurement bitstring results via CircuitExecutorData.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qiskit import qasm3, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, CircuitExecutorData, QuantumErrorProfile, Settings
from qdk_chemistry.plugins.qiskit._interop.noise_model import get_noise_model_from_profile
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitAerSimulator", "QiskitAerSimulatorSettings"]


class QiskitAerSimulatorSettings(Settings):
    """Settings for the Qiskit Aer Simulator circuit executor."""

    def __init__(self) -> None:
        """Initialize Qiskit Aer Simulator settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("seed", "int", 42)
        self._set_default("method", "string", "statevector")
        self._set_default("transpile_optimization_level", "int", 0)


class QiskitAerSimulator(CircuitExecutor):
    """Qiskit Aer Simulator circuit executor implementation."""

    def __init__(
        self,
        method: str = "statevector",
        seed: int = 42,
        transpile_optimization_level: int = 0,
    ) -> None:
        """Initialize the Qiskit Aer Simulator circuit executor.

        Args:
            method: The simulation method to use.
            seed: The random seed for simulation reproducibility.
            transpile_optimization_level: The optimization level for transpilation.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = QiskitAerSimulatorSettings()
        self._settings.set("seed", seed)
        self._settings.set("method", method)
        self._settings.set("transpile_optimization_level", transpile_optimization_level)

    def _run_impl(
        self,
        circuit: Circuit,
        shots: int,
        noise: QuantumErrorProfile | None = None,
    ) -> CircuitExecutorData:
        """Execute the given quantum circuit using the Qiskit Aer Simulator.

        Args:
            circuit: The quantum circuit to execute.
            shots: The number of shots to execute the circuit.
            noise: Optional noise profile to apply during execution.

        Returns:
            CircuitExecutorData: Object containing the results of the circuit execution.

        """
        Logger.trace_entering()
        meas_circuit = qasm3.loads(circuit.qasm)
        Logger.debug("QASM circuit loaded into Qiskit QuantumCircuit.")
        noise_model = get_noise_model_from_profile(noise) if noise else None
        backend = AerSimulator(
            method=self._settings.get("method"), seed_simulator=self._settings.get("seed"), noise_model=noise_model
        )
        if noise_model:
            transpiled_circuit = transpile(
                meas_circuit,
                basis_gates=noise_model.basis_gates,
                optimization_level=self._settings.get("transpile_optimization_level"),
            )
        else:
            # Use qiskit_aer NoiseModel() default basis gates if no noise model is provided
            transpiled_circuit = transpile(
                meas_circuit,
                basis_gates=NoiseModel().basis_gates,
                optimization_level=self._settings.get("transpile_optimization_level"),
            )
        raw_results = backend.run(transpiled_circuit, shots=shots).result()
        counts = raw_results.get_counts()
        Logger.debug(f"Measurement results obtained: {counts}")
        return CircuitExecutorData(
            bitstring_counts=counts,
            total_shots=shots,
            executor=self.name(),
            executor_metadata=raw_results,
        )

    def name(self) -> str:
        """Return the algorithm name as qiskit_aer_simulator."""
        return "qiskit_aer_simulator"
