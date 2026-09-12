"""QDK/Chemistry Circuit Executor for Azure Quantum.

This module provides a CircuitExecutor implementation that submits QIR circuits
to an Azure Quantum target and returns measurement bitstring results via CircuitExecutorData.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from azure.quantum import Workspace

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, CircuitExecutorData, QuantumErrorProfile, Settings
from qdk_chemistry.plugins._azure_auth import create_credential
from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from azure.quantum.job import Job

__all__: list[str] = ["AzureQuantumBackend", "AzureQuantumBackendSettings"]

_WORKSPACE_SETTINGS = ("subscription_id", "resource_group", "workspace_name", "location", "target_name")


def _process_raw_results(raw_results: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Convert Azure Quantum job histogram results to integer bitstring counts.

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


class AzureQuantumBackendSettings(Settings):
    """Settings for the Azure Quantum circuit executor."""

    def __init__(self) -> None:
        """Initialize Azure Quantum backend settings."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("job_name", "string", "qdk-chemistry-azure-quantum-backend", "Name for the submitted job")
        self._set_default("timeout_secs", "int", 3600, "Maximum seconds to wait for job completion")
        self._set_default("input_params", "string", "{}", "Job input parameters as a JSON object string")
        self._set_default("subscription_id", "string", "", "Azure subscription ID")
        self._set_default("resource_group", "string", "", "Azure resource group name")
        self._set_default("workspace_name", "string", "", "Azure Quantum workspace name")
        self._set_default("location", "string", "", "Azure Quantum workspace region")
        self._set_default("target_name", "string", "", "Azure Quantum target to submit to")
        self._set_default("auth_mode", "string", "azure-cli", "Azure credential mode: 'azure-cli' or 'default'")
        self._set_default("output_dir", "string", "", "Local directory to save job attachments into; empty disables")
        self._set_default(
            "attachments",
            "vector<string>",
            [],
            "Attachment names to download from the job container; empty disables saving",
        )


class AzureQuantumBackend(CircuitExecutor):
    """Circuit executor that submits QIR to an Azure Quantum target."""

    def __init__(
        self,
        subscription_id: str | None = None,
        resource_group: str | None = None,
        workspace_name: str | None = None,
        location: str | None = None,
        target_name: str | None = None,
        auth_mode: str | None = None,
        job_name: str | None = None,
        timeout_secs: int | None = None,
        input_params: dict | str | None = None,
        output_dir: str | None = None,
        attachments: list[str] | None = None,
    ) -> None:
        """Initialize the Azure Quantum circuit executor.

        Args:
            subscription_id: Azure subscription ID.
            resource_group: Azure resource group name.
            workspace_name: Azure Quantum workspace name.
            location: Azure Quantum workspace region.
            target_name: Azure Quantum target to submit to.
            auth_mode: Credential mode, ``"azure-cli"`` (default) or ``"default"``.
            job_name: Name for the submitted Azure Quantum job.
            timeout_secs: Maximum seconds to wait for job completion.
            input_params: Job input parameters as a dict or JSON object string.
            output_dir: Local directory to save job attachments into; empty or None disables saving.
            attachments: Attachment names to download from the job container; empty or None disables saving.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = AzureQuantumBackendSettings()
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
        if input_params is not None:
            if not isinstance(input_params, str):
                input_params = json.dumps(input_params)
            self._settings.set("input_params", input_params)

    def _run_impl(
        self,
        circuit: Circuit,
        shots: int,
        noise: QuantumErrorProfile | None = None,
    ) -> CircuitExecutorData:
        """Execute the given quantum circuit on the Azure Quantum target.

        Args:
            circuit: The quantum circuit to execute.
            shots: The number of shots to execute the circuit.
            noise: Not used. Configure noise through the target's own ``input_params``.

        Returns:
            CircuitExecutorData: Object containing the results of the circuit execution.

        Raises:
            NotImplementedError: If a noise profile is supplied.
            ValueError: If the connection settings are incomplete or ``input_params`` is not a JSON object.

        """
        Logger.trace_entering()
        if noise is not None:
            raise NotImplementedError(
                "Custom noise profiles are not yet supported by the Azure Quantum circuit executor."
                " Configure noise through the target's own 'input_params' instead."
            )

        coordinates = {key: self._settings.get(key) for key in _WORKSPACE_SETTINGS}
        missing = [key for key, value in coordinates.items() if not value]
        if missing:
            raise ValueError("Azure Quantum target cannot be resolved; set " + ", ".join(missing))

        qir_string = str(circuit.get_qir())
        Logger.debug("QIR compiled")

        workspace = Workspace(
            subscription_id=coordinates["subscription_id"],
            resource_group=coordinates["resource_group"],
            name=coordinates["workspace_name"],
            location=coordinates["location"],
            credential=create_credential(self._settings.get("auth_mode")),
        )
        target = workspace.get_targets(name=coordinates["target_name"])

        try:
            input_params = json.loads(self._settings.get("input_params"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in 'input_params': {error.msg}") from error
        if not isinstance(input_params, dict):
            raise ValueError("'input_params' must be a JSON object")

        job = target.submit(
            name=self._settings.get("job_name"),
            shots=shots,
            input_data=qir_string,
            input_data_format="qir.v1",
            output_data_format="microsoft.quantum-results.v2",
            input_params=input_params,
        )
        Logger.debug(f"Job submitted: {job.id}")

        timeout = self._settings.get("timeout_secs")
        raw_results = job.get_results_histogram(timeout_secs=timeout)
        Logger.debug("Job completed")

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
        """Return the algorithm name as azure_quantum_backend."""
        return "azure_quantum_backend"
