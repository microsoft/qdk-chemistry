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

import qsharp
from qsharp._simulation import run_qir
from qsharp.openqasm import run as sparse_state_run_qasm

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

    def __init__(
        self,
        simulator_type: Literal["cpu", "gpu", "clifford"] = "cpu",
        seed: int = 42,
        skip_qir_order_check: bool = False,
    ) -> None:
        """Initialize the QDK Full State Simulator circuit executor.

        Args:
            simulator_type: The type of simulator to use.
            seed: The random seed for simulation reproducibility.
            skip_qir_order_check: Whether to skip check of QIR to determine bitstring order.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = QdkFullStateSimulatorSettings()
        self._settings.set("type", simulator_type)
        self._settings.set("seed", seed)
        self._settings.set("skip_qir_order_check", skip_qir_order_check)

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
        qir = circuit.get_qir()
        Logger.debug("QIR compiled")
        noise_config = noise.to_qdk_noise_config() if noise is not None else None
        raw_results = run_qir(
            qir, shots=shots, noise=noise_config, seed=self._settings.get("seed"), type=self._settings.get("type")
        )
        Logger.debug(f"Measurement results obtained: {raw_results}")
        # Reorder bits in each measurement result to match Little Endian convention
        bitstrings = ["".join("0" if str(x) == "Zero" else "1" for x in reversed(one_run)) for one_run in raw_results]
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


class QdkSparseStateSimulatorSettings(Settings):
    """Settings for the QDK Sparse State Simulator circuit executor."""

    def __init__(self) -> None:
        """Initialize QDK Sparse State Simulator settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("qubit_loss", "double", 0.0, "Qubit loss rate for sparse state simulation")
        self._set_default("seed", "int", 42, "Random seed for simulation reproducibility")
        self._set_default(
            "noise_type",
            "string",
            "none",
            "Type of noise to apply",
            ["none", "depolarizing", "pauli", "bitflip", "phaseflip"],
        )
        self._set_default(
            "noise_rate",
            "vector<double>",
            [0.0],
            "Noise rate to apply during simulation (ignored if noise_type is 'none')",
        )


_QSHARP_SPARSE_STATE_NOISE_MAPPING = {
    "depolarizing": qsharp.DepolarizingNoise,
    "bitflip": qsharp.BitFlipNoise,
    "pauli": qsharp.PauliNoise,
    "phaseflip": qsharp.PhaseFlipNoise,
}


class QdkSparseStateSimulator(CircuitExecutor):
    """QDK Sparse State Simulator circuit executor implementation."""

    def __init__(self) -> None:
        """Initialize the QDK Sparse State Simulator circuit executor."""
        Logger.trace_entering()
        super().__init__()
        self._settings = QdkSparseStateSimulatorSettings()

    def _run_impl(
        self,
        circuit: Circuit,
        shots: int,
        noise: QuantumErrorProfile | None = None,
    ) -> CircuitExecutorData:
        """Execute the given quantum circuit using the QDK Sparse State Simulator.

        Args:
            circuit: The quantum circuit to execute.
            shots: The number of shots to execute the circuit.
            noise: Optional noise profile to apply during execution.

        Returns:
            CircuitExecutorData: Object containing the results of the circuit execution.

        """
        Logger.trace_entering()
        qasm = circuit.get_qasm()
        if noise is not None:
            raise NotImplementedError(
                "Gate specific noise is not yet supported for the QDK Sparse State Simulator. "
                "Please define noise at the settings level using the 'noise_type' and 'noise_rate' parameters."
            )

        noise_type = self._settings.get("noise_type")
        noise_rate = self._settings.get("noise_rate")
        if noise_type != "none":
            if noise_type in ["depolarizing", "bitflip", "phaseflip"] and len(noise_rate) != 1:
                raise ValueError(f"Noise rate for {noise_type} noise must be a single value")
            if noise_type == "pauli" and len(noise_rate) != 3:
                raise ValueError("Noise rate for pauli noise must be a list of 3 values")
            noise_model = _QSHARP_SPARSE_STATE_NOISE_MAPPING[noise_type](*noise_rate)
        else:
            noise_model = None

        raw_results = sparse_state_run_qasm(
            qasm,
            shots=shots,
            noise=noise_model,
            as_bitstring=True,
            seed=self._settings.get("seed"),
            qubit_loss=self._settings.get("qubit_loss"),
        )
        Logger.debug(f"Measurement results obtained: {raw_results}")
        # Reverse the order of bits in each measurement result to match Little Endian convention
        bitstring_count = {bitstring[::-1]: count for bitstring, count in Counter(raw_results).items()}
        return CircuitExecutorData(
            bitstring_counts=bitstring_count,
            total_shots=shots,
            executor=self.name(),
            executor_metadata=raw_results,
        )

    def name(self) -> str:
        """Return the algorithm name as qdk_sparse_state_simulator."""
        return "qdk_sparse_state_simulator"
