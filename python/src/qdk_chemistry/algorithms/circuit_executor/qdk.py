"""QDK/Chemistry Circuit Executor QDK implementation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
from collections import Counter
from typing import Literal

from qdk.openqasm import compile as compile_qir
from qdk.simulation import NoiseConfig, run_qir

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, CircuitExecutorData, Settings
from qdk_chemistry.utils import Logger

__all__: list[str] = []


class QdkFullStateSimulatorSettings(Settings):
    """Settings for the QDK Full State Simulator circuit executor."""

    def __init__(self) -> None:
        """Initialize QDK Full State Simulator settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("type", "string", "cpu")
        self._set_default("seed", "int", 42)


class QdkFullStateSimulator(CircuitExecutor):
    """QDK Full State Simulator circuit executor implementation."""

    def __init__(self, simulator_type: Literal["cpu", "gpu", "clifford"], seed: int = 42) -> None:
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
        noise: NoiseConfig | None = None,
    ) -> None:
        """Execute the given quantum circuit using the QDK Full State Simulator.

        Args:
            circuit: The quantum circuit to execute.
            shots: The number of shots to execute the circuit.
            noise: The noise configuration to apply during execution.

        Returns:
            CircuitExecutorData: Object containing the results of the circuit execution.

        """
        Logger.trace_entering()
        qir = compile_qir(circuit.qasm)
        raw_results = run_qir(
            qir, shots=shots, noise=noise, seed=self._settings.get("seed"), type=self._settings.get("type")
        )
        Logger.debug(f"Measurement results obtained: {raw_results}")
        reversed_bitstrings = [
            "".join("0" if str(x) == "Zero" else "1" for x in reversed(one_run)) for one_run in raw_results
        ]
        Logger.debug(f"Bitstrings saved in Little-Endian format: {reversed_bitstrings}")
        counts = dict(Counter(reversed_bitstrings))
        return CircuitExecutorData(
            bitstring_counts=counts,
            total_shots=shots,
            executor=self.name(),
            executor_metadata=raw_results,
        )

    def name(self) -> str:
        """Return the algorithm name as qdk_full_state_simulator."""
        return "qdk_full_state_simulator"
