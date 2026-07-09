"""QDK/Chemistry Circuit Executor for Azure Quantum neutral-atom emulators.

This module provides a CircuitExecutor implementation that submits QIR circuits
to an Azure Quantum emulator target (e.g. the AC1000 Clifford emulator) and
returns measurement bitstring results via CircuitExecutorData.

Non-Clifford gates in the QIR are rounded to the nearest Clifford gate before
submission when ``clifford_rounding`` is enabled (the default).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import numpy as np
from azure.identity import AzureCliCredential
from azure.quantum import Workspace

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import Circuit, CircuitExecutorData, QuantumErrorProfile, Settings
from qdk_chemistry.utils import Logger

__all__: list[str] = ["AzureQuantumEmulator"]


# ---------------------------------------------------------------------------
# Clifford rounding helpers
# ---------------------------------------------------------------------------

_ROT1_RE = re.compile(
    r"(call\s+void\s+@__quantum__qis__r(?:x|y|z)__body\()"
    r"double\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(,\s*.+\))"
)
_ROT2_RE = re.compile(
    r"(call\s+void\s+@__quantum__qis__r(?:xx|yy|zz)__body\()"
    r"double\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(,\s*.+\))"
)
_T_BODY_RE = re.compile(r"call\s+void\s+@__quantum__qis__t__body\((%Qubit\*\s*.+)\)")
_T_ADJ_RE = re.compile(r"call\s+void\s+@__quantum__qis__t__adj\((%Qubit\*\s*.+)\)")


def _nearest_clifford_angle(angle: float) -> float:
    """Return the multiple of π/2 closest to *angle*."""
    k = round(angle / (np.pi / 2))
    return k * (np.pi / 2)


def _inject_declaration(qir: str, decl: str) -> str:
    """Insert a missing ``declare`` line next to existing QIS declarations."""
    lines = qir.splitlines()
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("declare void @__quantum__qis__"):
            insert_idx = i
    if insert_idx is not None:
        lines.insert(insert_idx + 1, decl)
    else:
        lines.append(decl)
    return "\n".join(lines)


def round_qir_to_clifford(qir: str) -> str:
    """Round every non-Clifford gate in a QIR string to its nearest Clifford.

    Handles single-qubit rotations (rz, rx, ry), two-qubit rotations
    (rxx, ryy, rzz), T gates, and T-dagger gates. Rotation angles are
    snapped to the nearest multiple of π/2; identity rotations are removed.

    Args:
        qir: QIR string (LLVM IR text).

    Returns:
        Modified QIR string with only Clifford gates.

    """
    lines = qir.splitlines()
    out: list[str] = []

    needs_s_body = False
    has_s_body_decl = "__quantum__qis__s__body" in qir

    for line in lines:
        stripped = line.strip()

        m = _ROT1_RE.search(stripped)
        if m:
            rounded = _nearest_clifford_angle(float(m.group(2)))
            if abs(rounded % (2 * np.pi)) < 1e-12:
                continue
            out.append(line.replace(m.group(2), f"{rounded:.15e}"))
            continue

        m2 = _ROT2_RE.search(stripped)
        if m2:
            rounded = _nearest_clifford_angle(float(m2.group(2)))
            if abs(rounded % (2 * np.pi)) < 1e-12:
                continue
            out.append(line.replace(m2.group(2), f"{rounded:.15e}"))
            continue

        mt = _T_BODY_RE.search(stripped)
        if mt:
            out.append(line.replace(stripped, f"  call void @__quantum__qis__s__body({mt.group(1)})"))
            needs_s_body = True
            continue

        if _T_ADJ_RE.search(stripped):
            continue

        out.append(line)

    result = "\n".join(out)
    if needs_s_body and not has_s_body_decl:
        result = _inject_declaration(result, "declare void @__quantum__qis__s__body(%Qubit*)")
    return result


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class AzureQuantumEmulatorSettings(Settings):
    """Settings for the Azure Quantum Emulator circuit executor."""

    def __init__(self) -> None:
        """Initialize Azure Quantum Emulator settings."""
        super().__init__()
        self._set_default("subscription_id", "string", "", "Azure subscription ID")
        self._set_default("resource_group", "string", "", "Azure resource group name")
        self._set_default("workspace_name", "string", "", "Azure Quantum workspace name")
        self._set_default("location", "string", "", "Azure region (e.g. eastus)")
        self._set_default("target_name", "string", "", "Emulator target ID (e.g. <target-id>)")
        self._set_default(
            "clifford_rounding", "bool", True, "Round non-Clifford gates to nearest Clifford before submission"
        )
        self._set_default("enable_noise", "bool", False, "Enable noise model on the emulator")
        self._set_default("emulate_timing", "bool", False, "Enable timing emulation")
        self._set_default("timeout_secs", "int", 600, "Maximum seconds to wait for job completion")


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class AzureQuantumEmulator(CircuitExecutor):
    """Circuit executor that submits QIR to an Azure Quantum emulator target."""

    def __init__(self, **kwargs) -> None:
        """Initialize the Azure Quantum Emulator circuit executor.

        Args:
            **kwargs: Forwarded to settings (subscription_id, resource_group, workspace_name, location, target_name, etc.).

        """
        super().__init__()
        self._settings = AzureQuantumEmulatorSettings()
        for k, v in kwargs.items():
            self._settings.set(k, v)

    def name(self) -> str:
        """Return the algorithm name."""
        return "azure_quantum_emulator"

    def _get_workspace(self) -> Workspace:
        """Build an Azure Quantum Workspace from the current settings."""
        return Workspace(
            subscription_id=self._settings.get("subscription_id"),
            resource_group=self._settings.get("resource_group"),
            name=self._settings.get("workspace_name"),
            location=self._settings.get("location"),
            credential=AzureCliCredential(),
        )

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
            noise: Optional noise profile (unused — noise is controlled via emulator settings).

        Returns:
            CircuitExecutorData containing bitstring measurement results.

        """
        Logger.trace_entering()

        qir_string = str(circuit.get_qir())

        if self._settings.get("clifford_rounding"):
            qir_string = round_qir_to_clifford(qir_string)
            Logger.debug("QIR Clifford-rounded")

        simulation_type = "clifford" if self._settings.get("clifford_rounding") else "cliffordrounding"

        workspace = self._get_workspace()
        target = workspace.get_targets(self._settings.get("target_name"))

        job = target.submit(
            name=f"qdk-chemistry-{self.name()}",
            input_data=qir_string,
            input_data_format="qir.v1",
            output_data_format="microsoft.quantum-results.v1",
            input_params={
                "shots": shots,
                "emulationSettings": {
                    "simulationType": simulation_type,
                    "enableNoise": self._settings.get("enable_noise"),
                    "emulateTiming": self._settings.get("emulate_timing"),
                },
            },
        )
        Logger.debug(f"Job submitted: {job.id}")

        timeout = self._settings.get("timeout_secs")
        raw_results = job.get_results(timeout_secs=timeout)
        Logger.debug(f"Job completed: {raw_results}")

        bitstring_counts = _parse_emulator_results(raw_results, shots)

        return CircuitExecutorData(
            bitstring_counts=bitstring_counts,
            total_shots=shots,
            executor=self.name(),
            executor_metadata=raw_results,
        )


def _parse_emulator_results(raw_results: dict, total_shots: int) -> dict[str, int]:
    """Convert emulator probability results to integer bitstring counts.

    The emulator returns ``{'[0,1,0]': 0.45, '[1,0,1]': 0.55}`` (probabilities).
    This converts to integer counts based on total_shots.

    Args:
        raw_results: Dict mapping bitstring labels to probabilities.
        total_shots: Total number of shots.

    Returns:
        Dict mapping cleaned bitstring labels to integer counts.

    """
    counts: dict[str, int] = {}
    for key, prob in raw_results.items():
        clean_key = key.strip("[]").replace(",", "").replace(" ", "")
        count = round(prob * total_shots)
        if count > 0:
            counts[clean_key] = count
    return counts
