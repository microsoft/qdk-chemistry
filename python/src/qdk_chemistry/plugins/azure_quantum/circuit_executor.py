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

from azure.identity import AzureCliCredential
from azure.quantum import Workspace

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, CircuitExecutorData, QuantumErrorProfile, Settings
from qdk_chemistry.utils import Logger

__all__: list[str] = ["AzureQuantumEmulator", "AzureQuantumEmulatorSettings"]


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
        self._set_default("subscription_id", "string", "", "Azure subscription ID")
        self._set_default("resource_group", "string", "", "Azure resource group name")
        self._set_default("workspace_name", "string", "", "Azure Quantum workspace name")
        self._set_default("location", "string", "", "Azure region (e.g. eastus)")
        self._set_default("target_name", "string", "", "Emulator target ID (e.g. <target-id>)")
        self._set_default(
            "simulation_type",
            "string",
            "cliffordrounding",
            "Emulator simulation type (e.g. cliffordrounding, clifford)",
        )
        self._set_default("seed", "int", 42, "Random seed for simulation reproducibility")
        self._set_default("enable_noise", "bool", False, "Enable noise model on the emulator")
        self._set_default("emulate_timing", "bool", False, "Enable timing emulation")
        self._set_default(
            "m_reset_z_us", "int", -1, "Measurement-reset-Z gate time in microseconds (-1 for device default)"
        )
        self._set_default("cz_us", "int", -1, "CZ gate time in microseconds (-1 for device default)")
        self._set_default("sx_us", "int", -1, "SX gate time in microseconds (-1 for device default)")
        self._set_default("rz_us", "int", -1, "RZ gate time in microseconds (-1 for device default)")
        self._set_default("timeout_secs", "int", 600, "Maximum seconds to wait for job completion")


class AzureQuantumEmulator(CircuitExecutor):
    """Circuit executor that submits QIR to an Azure Quantum emulator target."""

    def __init__(
        self,
        subscription_id: str = "",
        resource_group: str = "",
        workspace_name: str = "",
        location: str = "",
        target_name: str = "",
        simulation_type: str = "cliffordrounding",
        seed: int = 42,
        enable_noise: bool = False,
        emulate_timing: bool = False,
        m_reset_z_us: int = -1,
        cz_us: int = -1,
        sx_us: int = -1,
        rz_us: int = -1,
        timeout_secs: int = 600,
    ) -> None:
        """Initialize the Azure Quantum Emulator circuit executor.

        Args:
            subscription_id: Azure subscription ID.
            resource_group: Azure resource group name.
            workspace_name: Azure Quantum workspace name.
            location: Azure region (e.g. eastus).
            target_name: Emulator target ID (e.g. <target-id>).
            simulation_type: Emulator simulation type (e.g. cliffordrounding, clifford).
            seed: Random seed for simulation reproducibility.
            enable_noise: Enable noise model on the emulator.
            emulate_timing: Enable timing emulation.
            m_reset_z_us: Measurement-reset-Z gate time in microseconds (-1 for device default).
            cz_us: CZ gate time in microseconds (-1 for device default).
            sx_us: SX gate time in microseconds (-1 for device default).
            rz_us: RZ gate time in microseconds (-1 for device default).
            timeout_secs: Maximum seconds to wait for job completion.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = AzureQuantumEmulatorSettings()
        self._settings.set("subscription_id", subscription_id)
        self._settings.set("resource_group", resource_group)
        self._settings.set("workspace_name", workspace_name)
        self._settings.set("location", location)
        self._settings.set("target_name", target_name)
        self._settings.set("simulation_type", simulation_type)
        self._settings.set("seed", seed)
        self._settings.set("enable_noise", enable_noise)
        self._settings.set("emulate_timing", emulate_timing)
        self._settings.set("m_reset_z_us", m_reset_z_us)
        self._settings.set("cz_us", cz_us)
        self._settings.set("sx_us", sx_us)
        self._settings.set("rz_us", rz_us)
        self._settings.set("timeout_secs", timeout_secs)

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

        workspace = Workspace(
            subscription_id=self._settings.get("subscription_id"),
            resource_group=self._settings.get("resource_group"),
            name=self._settings.get("workspace_name"),
            location=self._settings.get("location"),
            credential=AzureCliCredential(),
        )
        target = workspace.get_targets(self._settings.get("target_name"))

        emulation_settings: dict = {
            "simulationType": self._settings.get("simulation_type"),
            "enableNoise": self._settings.get("enable_noise"),
            "emulateTiming": self._settings.get("emulate_timing"),
            "seed": self._settings.get("seed"),
        }

        command_timings = {}
        if self._settings.get("m_reset_z_us") >= 0:
            command_timings["m_reset_z_us"] = self._settings.get("m_reset_z_us")
        if self._settings.get("cz_us") >= 0:
            command_timings["cz_us"] = self._settings.get("cz_us")
        if self._settings.get("sx_us") >= 0:
            command_timings["sx_us"] = self._settings.get("sx_us")
        if self._settings.get("rz_us") >= 0:
            command_timings["rz_us"] = self._settings.get("rz_us")
        if command_timings:
            emulation_settings["commandTimings"] = command_timings

        job = target.submit(
            name=f"qdk-chemistry-{self.name()}",
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
