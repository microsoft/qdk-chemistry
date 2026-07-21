"""QDK/Chemistry Circuit Executor for Azure Quantum neutral-atom emulators.

This module provides a CircuitExecutor implementation that submits QIR circuits
to an Azure Quantum emulator target (e.g. the AC1000 emulator) and returns
measurement bitstring results via CircuitExecutorData.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, CircuitExecutorData, QuantumErrorProfile, Settings
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from azure.quantum.target import Target

__all__: list[str] = ["AzureQuantumEmulator", "AzureQuantumEmulatorSettings"]

_DEFAULT_EMULATION_SETTINGS: dict = {
    "simulationType": "cliffordrounding",
    "enableNoise": False,
    "emulateTiming": False,
    "seed": 42,
}


def _process_raw_results(raw_results: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Convert emulator histogram results to integer bitstring counts.

    Uses the ``microsoft.quantum-results.v2`` histogram format returned by
    ``job.get_results_histogram()``, which maps a label to
    ``{'outcome': [...], 'count': n}``. Each ``outcome`` list holds per-qubit
    values of ``0``, ``1``, or ``'-'`` (a lost qubit). Shots with at least one
    lost qubit are separated into a loss dictionary, with ``'-'`` rendered as
    ``'L'`` to match the loss-bitstring convention.

    Args:
        raw_results: Histogram results from ``job.get_results_histogram()``.

    Returns:
        A ``(bitstring_counts, loss_bitstrings)`` tuple of label-to-count dicts; the latter is empty absent qubit loss.

    """
    counts: dict[str, int] = {}
    loss: dict[str, int] = {}
    for entry in raw_results.values():
        outcome = entry["outcome"]
        count = entry["count"]
        if "-" in outcome:
            key = "".join("L" if bit == "-" else str(bit) for bit in outcome)
            loss[key] = loss.get(key, 0) + count
        else:
            key = "".join(str(bit) for bit in outcome)
            counts[key] = counts.get(key, 0) + count
    return counts, loss


class AzureQuantumEmulatorSettings(Settings):
    """Settings for the Azure Quantum Emulator circuit executor."""

    def __init__(self) -> None:
        """Initialize Azure Quantum Emulator settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default(
            "emulation_settings",
            "string",
            json.dumps(_DEFAULT_EMULATION_SETTINGS),
            "Azure Quantum emulationSettings, as a JSON object string",
        )
        self._set_default("job_name", "string", "qdk-chemistry-azure-quantum-emulator", "Name for the submitted job")
        self._set_default("timeout_secs", "int", 3600, "Maximum seconds to wait for job completion")


class AzureQuantumEmulator(CircuitExecutor):
    """Circuit executor that submits QIR to an Azure Quantum emulator target."""

    def __init__(
        self,
        target: Target | None = None,
        emulation_settings: dict | None = None,
        job_name: str = "qdk-chemistry-azure-quantum-emulator",
        timeout_secs: int = 3600,
    ) -> None:
        """Initialize the Azure Quantum Emulator circuit executor.

        Pass an already-resolved ``azure.quantum`` ``Target`` (e.g.
        ``workspace.get_targets("...")``). The executor reuses your existing
        workspace and credential, so no subscription/resource-group/workspace
        connection details are needed here. The target is optional here and can
        instead be provided later via :meth:`set_target`, but it must be set
        before :meth:`run` is called.

        Args:
            target: Pre-resolved Azure Quantum ``Target`` to submit circuits to; may be set later via set_target().
            emulation_settings: Azure Quantum ``emulationSettings`` dict; defaults to a Clifford-rounding config.
            job_name: Name for the submitted Azure Quantum job.
            timeout_secs: Maximum seconds to wait for job completion.

        """
        Logger.trace_entering()
        super().__init__()
        self._target = target
        self._settings = AzureQuantumEmulatorSettings()
        if emulation_settings is not None:
            self._settings.set("emulation_settings", json.dumps(emulation_settings))
        self._settings.set("job_name", job_name)
        self._settings.set("timeout_secs", timeout_secs)

    def set_target(self, target: Target) -> None:
        """Set the Azure Quantum target to submit circuits to.

        Args:
            target: Pre-resolved Azure Quantum ``Target`` to submit circuits to.

        """
        Logger.trace_entering()
        self._target = target

    def _run_impl(
        self,
        circuit: Circuit,
        shots: int,
        noise: QuantumErrorProfile | None = None,
    ) -> CircuitExecutorData:
        """Execute the given quantum circuit on the Azure Quantum emulator.

        Args:
            circuit: The quantum circuit to execute.
            shots: The number of shots to execute the circuit.
            noise: Not used. Noise is controlled via the ``enable_noise`` setting.

        Returns:
            CircuitExecutorData: Object containing the results of the circuit execution.

        """
        Logger.trace_entering()
        if noise is not None:
            raise NotImplementedError(
                "Custom noise profiles are not yet supported by the Azure Quantum emulator executor."
                " Use the 'enable_noise' setting to enable the emulator's default noise model."
            )

        qir_string = str(circuit.get_qir())
        Logger.debug("QIR compiled")

        if self._target is None:
            raise ValueError("No Azure Quantum target set; pass one to the constructor or via set_target().")
        target = self._target

        emulation_settings: dict = json.loads(self._settings.get("emulation_settings"))

        job = target.submit(
            name=self._settings.get("job_name"),
            shots=shots,
            input_data=qir_string,
            input_data_format="qir.v1",
            output_data_format="microsoft.quantum-results.v2",
            input_params={
                "emulationSettings": emulation_settings,
            },
        )
        Logger.debug(f"Job submitted: {job.id}")

        timeout = self._settings.get("timeout_secs")
        raw_results = job.get_results_histogram(timeout_secs=timeout)
        Logger.debug(f"Job completed: {raw_results}")

        bitstring_counts, loss_bitstrings = _process_raw_results(raw_results)
        return CircuitExecutorData(
            bitstring_counts=bitstring_counts,
            total_shots=shots,
            executor=self.name(),
            executor_metadata=raw_results,
            loss_bitstrings=loss_bitstrings or None,
        )

    def name(self) -> str:
        """Return the algorithm name as azure_quantum_emulator."""
        return "azure_quantum_emulator"
