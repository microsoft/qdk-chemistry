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
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from azure.quantum import Workspace

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, CircuitExecutorData, QuantumErrorProfile, Settings
from qdk_chemistry.plugins._azure_auth import create_credential
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from azure.quantum.job import Job

__all__: list[str] = ["AzureQuantumEmulator", "AzureQuantumEmulatorSettings"]

_DEFAULT_EMULATION_SETTINGS: dict = {
    "simulationType": "cliffordrounding",
    "enableNoise": False,
    "emulateTiming": False,
    "seed": 42,
}

_WORKSPACE_ENV_VARS: dict[str, str] = {
    "subscription_id": "AZURE_QUANTUM_SUBSCRIPTION_ID",
    "resource_group": "AZURE_QUANTUM_RESOURCE_GROUP",
    "workspace_name": "AZURE_QUANTUM_WORKSPACE_NAME",
    "location": "AZURE_QUANTUM_LOCATION",
    "target_name": "AZURE_QUANTUM_TARGET_NAME",
    "auth_mode": "AZURE_QUANTUM_AUTH_MODE",
}


def _env(setting_key: str) -> str:
    """Read the environment default for one workspace setting.

    Args:
        setting_key: Key in :data:`_WORKSPACE_ENV_VARS`.

    Returns:
        str: The environment value, or an empty string when unset.

    """
    return os.environ.get(_WORKSPACE_ENV_VARS[setting_key], "")


def _process_raw_results(raw_results: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Convert emulator histogram results to integer bitstring counts.

    Uses the ``microsoft.quantum-results.v2`` histogram format returned by
    ``job.get_results_histogram()``, which maps a label to an outcome and count.
    An outcome is a scalar for one measurement or a sequence for multiple
    measurements, with values of ``0``, ``1``, or ``'-'`` (a lost qubit). Shots
    with at least one lost qubit are separated into a loss dictionary, with
    ``'-'`` rendered as ``'L'`` to match the loss-bitstring convention.

    The ``outcome`` list is ordered first-recorded-result-first; it is reversed
    here so the emitted bitstrings follow the qubit-0-rightmost convention used
    by the QDK and Qiskit executors.

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
        outcome_bits = outcome if isinstance(outcome, (list | tuple)) else (outcome,)
        if "-" in outcome_bits:
            key = "".join("L" if bit == "-" else str(bit) for bit in reversed(outcome_bits))
            loss[key] = loss.get(key, 0) + count
        else:
            key = "".join(str(bit) for bit in reversed(outcome_bits))
            counts[key] = counts.get(key, 0) + count
    return counts, loss


class AzureQuantumEmulatorSettings(Settings):
    """Settings for the Azure Quantum Emulator circuit executor.

    The connection settings default to the corresponding ``AZURE_QUANTUM_*``
    environment variable, so a configured shell needs no explicit arguments.
    """

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
        self._set_default("input_params", "string", "{}", "Additional input parameters for the job submission")
        self._set_default("subscription_id", "string", _env("subscription_id"), "Azure subscription ID")
        self._set_default("resource_group", "string", _env("resource_group"), "Azure resource group name")
        self._set_default("workspace_name", "string", _env("workspace_name"), "Azure Quantum workspace name")
        self._set_default("location", "string", _env("location"), "Azure Quantum workspace region")
        self._set_default("target_name", "string", _env("target_name"), "Azure Quantum target to submit to")
        self._set_default(
            "auth_mode",
            "string",
            _env("auth_mode") or "azure-cli",
            "Azure credential mode: 'azure-cli' or 'default'",
        )
        self._set_default("output_dir", "string", "", "Local directory to save job attachments into; empty disables")
        self._set_default(
            "attachments",
            "vector<string>",
            [],
            "Attachment names to download from the job container; empty disables saving",
        )


class AzureQuantumEmulator(CircuitExecutor):
    """Circuit executor that submits QIR to an Azure Quantum emulator target."""

    def __init__(
        self,
        subscription_id: str | None = None,
        resource_group: str | None = None,
        workspace_name: str | None = None,
        location: str | None = None,
        target_name: str | None = None,
        auth_mode: str | None = None,
        emulation_settings: dict | None = None,
        job_name: str | None = None,
        timeout_secs: int | None = None,
        output_dir: str | None = None,
        attachments: list[str] | None = None,
    ) -> None:
        """Initialize the Azure Quantum Emulator circuit executor.

        Every connection parameter falls back to the matching ``AZURE_QUANTUM_*``
        environment variable, so a configured shell needs no explicit arguments. The
        ``Workspace`` and ``Target`` are resolved on each run.

        Args:
            subscription_id: Azure subscription ID; defaults to ``$AZURE_QUANTUM_SUBSCRIPTION_ID``.
            resource_group: Azure resource group; defaults to ``$AZURE_QUANTUM_RESOURCE_GROUP``.
            workspace_name: Azure Quantum workspace name; defaults to ``$AZURE_QUANTUM_WORKSPACE_NAME``.
            location: Azure Quantum workspace region; defaults to ``$AZURE_QUANTUM_LOCATION``.
            target_name: Target to submit to; defaults to ``$AZURE_QUANTUM_TARGET_NAME``.
            auth_mode: Credential mode, ``"azure-cli"`` or ``"default"``; defaults to ``$AZURE_QUANTUM_AUTH_MODE``.
            emulation_settings: Azure Quantum ``emulationSettings`` dict; defaults to a Clifford-rounding config.
            job_name: Name for the submitted Azure Quantum job.
            timeout_secs: Maximum seconds to wait for job completion.
            output_dir: Local directory to save job attachments into; empty or None disables saving.
            attachments: Attachment names to download from the job container; empty or None disables saving.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = AzureQuantumEmulatorSettings()
        explicit: dict[str, Any] = {
            "subscription_id": subscription_id,
            "resource_group": resource_group,
            "workspace_name": workspace_name,
            "location": location,
            "target_name": target_name,
            "auth_mode": auth_mode,
            "job_name": job_name,
            "timeout_secs": timeout_secs,
            "output_dir": output_dir,
            "attachments": attachments,
        }
        for key, value in explicit.items():
            if value is not None:
                self._settings.set(key, value)
        if emulation_settings is not None:
            self._settings.set("emulation_settings", json.dumps(emulation_settings))

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

        Raises:
            ValueError: If the connection settings are incomplete.

        """
        Logger.trace_entering()
        if noise is not None:
            raise NotImplementedError(
                "Custom noise profiles are not yet supported by the Azure Quantum emulator executor."
                " Use the 'enable_noise' setting to enable the emulator's default noise model."
            )

        required = ("subscription_id", "resource_group", "workspace_name", "location", "target_name")
        missing = [key for key in required if not self._settings.get(key)]
        if missing:
            raise ValueError(
                "Azure Quantum target cannot be resolved; set "
                + ", ".join(f"{key} (${_WORKSPACE_ENV_VARS[key]})" for key in missing)
            )

        qir_string = str(circuit.get_qir())
        Logger.debug("QIR compiled")

        workspace = Workspace(
            subscription_id=self._settings.get("subscription_id"),
            resource_group=self._settings.get("resource_group"),
            name=self._settings.get("workspace_name"),
            location=self._settings.get("location"),
            credential=create_credential(self._settings.get("auth_mode")),
        )
        target = workspace.get_targets(name=self._settings.get("target_name"))

        emulation_settings: dict = json.loads(self._settings.get("emulation_settings"))

        job = target.submit(
            name=self._settings.get("job_name"),
            shots=shots,
            input_data=qir_string,
            input_data_format="qir.v1",
            output_data_format="microsoft.quantum-results.v2",
            input_params={
                "emulationSettings": emulation_settings,
                **json.loads(self._settings.get("input_params")),
            },
        )
        Logger.debug(f"Job submitted: {job.id}")

        timeout = self._settings.get("timeout_secs")
        raw_results = job.get_results_histogram(timeout_secs=timeout)
        Logger.debug(f"Job completed: {raw_results}")

        saved_attachments = self._save_attachments(job)

        bitstring_counts, loss_bitstrings = _process_raw_results(raw_results)
        return CircuitExecutorData(
            bitstring_counts=bitstring_counts,
            total_shots=shots,
            executor=self.name(),
            executor_metadata={
                "results": raw_results,
                "job_id": job.id,
                "saved_attachments": saved_attachments,
            },
            loss_bitstrings=loss_bitstrings or None,
        )

    def _save_attachments(self, job: Job) -> list[str]:
        """Save the named job attachments under the configured output directory.

        Args:
            job: The completed Azure Quantum job whose container holds the attachments.

        Returns:
            list[str]: Absolute paths of the files written, empty when saving is disabled.

        """
        output_dir = self._settings.get("output_dir")
        names = list(self._settings.get("attachments"))
        if not output_dir or not names:
            return []

        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for name in names:
            # Attachment names come from the service, so keep them from escaping output_dir.
            local_path = (destination / Path(name).name).resolve()
            if local_path.parent != destination:
                raise ValueError(f"refusing to write attachment outside the output directory: {name!r}")
            local_path.write_bytes(job.download_attachment(name))
            saved.append(str(local_path))
            Logger.debug(f"Saved attachment {name} to {local_path}")
        return saved

    def name(self) -> str:
        """Return the algorithm name as azure_quantum_emulator."""
        return "azure_quantum_emulator"
