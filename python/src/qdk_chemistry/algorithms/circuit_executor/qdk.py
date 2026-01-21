"""QDK/Chemistry Circuit Executor implementation using QDK.

This module provides a CircuitExecutor implementation that uses the QDK backends
to execute quantum circuits. It accepts QDK/Chemistry Circuit and QuantumErrorProfile
data classes and returns measurement bitstring results via CircuitExecutorData.

Supported QDK backends include:
    * QDK Full State Simulator
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from collections import Counter
from typing import Literal

from qdk.openqasm import compile as compile_qir
from qsharp._simulation import run_qir

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, CircuitExecutorData, QuantumErrorProfile, Settings
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QdkFullStateSimulator", "QdkFullStateSimulatorSettings"]


class QdkFullStateSimulatorSettings(Settings):
    """Settings for the QDK Full State Simulator circuit executor."""

    def __init__(self) -> None:
        """Initialize QDK Full State Simulator settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default(
            "type", "string", "cpu", "Type of simulator to use: 'cpu', 'gpu', or 'clifford'", ["cpu", "gpu", "clifford"]
        )
        self._set_default("seed", "int", 42, "Random seed for simulation reproducibility")


class QdkFullStateSimulator(CircuitExecutor):
    """QDK Full State Simulator circuit executor implementation."""

    def __init__(self, simulator_type: Literal["cpu", "gpu", "clifford"] = "cpu", seed: int = 42) -> None:
        """Initialize the QDK Full State Simulator circuit executor.

        Args:
            simulator_type: The type of simulator to use.
            seed: The random seed for simulation reproducibility.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = QdkFullStateSimulatorSettings()
        self._settings.set("type", simulator_type)
        self._settings.set("seed", seed)

    def _run_impl(
        self,
        circuit: Circuit,
        shots: int,
        noise: QuantumErrorProfile | None = None,
    ) -> CircuitExecutorData:
        """Execute the given quantum circuit using the QDK Full State Simulator.

        Args:
            circuit: The quantum circuit to execute.
            shots: The number of shots to execute the circuit.
            noise: Optional noise profile to apply during execution.

        Returns:
            CircuitExecutorData: Object containing the results of the circuit execution.

        """
        Logger.trace_entering()
        qir = compile_qir(circuit.qasm)
        Logger.debug("QIR compiled")
        noise_config = noise.to_qdk_noise_config() if noise is not None else None
        raw_results = run_qir(
            qir, shots=shots, noise=noise_config, seed=self._settings.get("seed"), type=self._settings.get("type")
        )
        Logger.debug(f"Measurement results obtained: {raw_results}")
        bitstrings = ["".join("0" if str(x) == "Zero" else "1" for x in one_run) for one_run in raw_results]
        counts = dict(Counter(bitstrings))
        return CircuitExecutorData(
            bitstring_counts=counts,
            total_shots=shots,
            executor=self.name(),
            executor_metadata=raw_results,
        )

    def name(self) -> str:
        """Return the algorithm name as qdk_full_state_simulator."""
        return "qdk_full_state_simulator"
