"""Local Azure Quantum demo targets backed by QDK CPU full-state simulation.

No Azure connection string or credentials are needed. ``demo://local`` is a
local-only identifier understood by ``DemoWorkspace.from_connection_string``;
it is NOT a connection string for the real Azure service.

Use the unchanged Azure Quantum backend with::

    from qdk_chemistry.plugins.azure_quantum.circuit_executor import AzureQuantumBackend
    from qdk_chemistry.plugins.azure_quantum.demo import DEMO_BACKEND_SETTINGS, demo_azure_quantum

    with demo_azure_quantum(seed=42):
        executor = AzureQuantumBackend(**DEMO_BACKEND_SETTINGS)
        result = executor.run(circuit, shots=100)

Or use the target directly, without patching anything::

    workspace = DemoWorkspace.from_connection_string("demo://local")
    target = workspace.get_targets("demo.target")
    job = target.submit(input_data=circuit.get_qir(), shots=100)
    print(job.get_results_histogram())

Only QIR is accepted: a QDK QIR object, LLVM IR text/UTF-8 bytes, or bitcode.
It is passed directly to QDK's ``run_qir(..., type="cpu")``: no QIR-to-QASM
conversion, sparse simulation, or Qiskit dependency. Install the
``azure-quantum`` extra of qdk-chemistry to use this module.

QIR support is determined by QDK's CPU full-state simulator; errors propagate
without falling back to cloud execution. Outputs must be measurements or
sequences of measurements. Provider options, hardware noise, and attachments
are not emulated. Every target name aliases the same ideal simulator.
Jobs finish synchronously and are held in memory, never submitted to Azure.
Full-state memory usage grows exponentially with the number of qubits.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, overload
from unittest.mock import patch
from uuid import uuid4

import azure.quantum
from azure.quantum.job import JobDetails
from qdk.qsharp import Result

from qdk_chemistry.algorithms.circuit_executor.qdk import run_qir
from qdk_chemistry.data.circuit import QirInputData

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "DEMO_BACKEND_SETTINGS",
    "DEMO_CONNECTION_STRING",
    "DemoJob",
    "DemoTarget",
    "DemoWorkspace",
    "demo_azure_quantum",
]

DEMO_CONNECTION_STRING = "demo://local"
DEMO_BACKEND_SETTINGS = {
    "subscription_id": "00000000-0000-0000-0000-000000000000",
    "resource_group": "demo",
    "workspace_name": "demo",
    "location": "local",
    "target_name": "demo.target",
}
_OUTPUT_FORMAT = "microsoft.quantum-results.v2"


def _measurement_outcome(value: Any) -> Any:
    """Convert QDK measurements to Azure-style outcomes without changing order."""
    if isinstance(value, (list, tuple)):
        return [_measurement_outcome(item) for item in value]
    if isinstance(value, Result):
        return {"Zero": 0, "One": 1, "Loss": "-"}[str(value)]
    raise ValueError(f"Demo targets require measurement outputs, got {value!r}")


class DemoJob:
    """A completed, in-memory Azure-style job with no network operations."""

    def __init__(self, target: DemoTarget, name: str, params: dict[str, Any], outcomes: list[Any]) -> None:
        """Store the simulator's measurement outcomes."""
        self.id = f"demo-{uuid4()}"
        self.workspace = target.workspace
        self.details = JobDetails(
            id=self.id,
            name=name,
            target=target.name,
            provider_id=target.provider_id,
            container_uri=f"memory://{self.id}",
            input_data_format="qir.v1",
            output_data_format=_OUTPUT_FORMAT,
            input_params=deepcopy(params),
            status="Succeeded",
            metadata={"demo": True, "simulator": "qdk-full-state-cpu", "seed": target.workspace.seed},
        )
        self._shots = deepcopy(outcomes)
        counts = Counter(json.dumps(outcome) for outcome in outcomes)
        self._histogram = {label: {"outcome": json.loads(label), "count": count} for label, count in counts.items()}

    def has_completed(self) -> bool:
        """Simulation completes before submit returns."""
        return True

    def has_succeeded(self) -> bool:
        """Simulation errors raise from submit instead of creating a job."""
        return True

    def refresh(self) -> None:
        """There is nothing to refresh for a completed local job."""

    def wait_until_completed(
        self, max_poll_wait_secs: float = 30, timeout_secs: float | None = None, print_progress: bool = True
    ) -> None:
        """Return immediately; all work has already completed locally."""

    def get_results_histogram(self, timeout_secs: float = 300) -> dict[str, dict[str, Any]]:  # noqa: ARG002
        """Return Azure-style outcome/count entries in QIR output-recording order."""
        return deepcopy(self._histogram)

    def get_results_shots(self, timeout_secs: float = 300) -> list[Any]:  # noqa: ARG002
        """Return the individual recorded outcomes."""
        return deepcopy(self._shots)

    def get_results(self, timeout_secs: float = 300) -> dict[str, float]:  # noqa: ARG002
        """Return probabilities, as the SDK does for quantum-results.v2."""
        return {label: entry["count"] / len(self._shots) for label, entry in self._histogram.items()}

    def download_attachment(self, name: str) -> bytes:
        """Provider-generated attachments do not exist locally."""
        raise NotImplementedError(f"Demo jobs do not produce Azure job attachments: {name!r}")


class DemoTarget:
    """An Azure-style QIR target wrapping QDK's ideal CPU full-state simulator."""

    def __init__(self, workspace: DemoWorkspace, name: str, provider_id: str | None = None) -> None:
        """Associate a target alias with a local workspace."""
        self.workspace = workspace
        self.name = name
        self.provider_id = provider_id or name.split(".", maxsplit=1)[0]
        self.input_data_format = "qir.v1"
        self.output_data_format = _OUTPUT_FORMAT
        self.current_availability = "Available"
        self.average_queue_time = 0

    def submit(
        self,
        input_data: QirInputData | str | bytes,
        name: str = "demo-job",
        shots: int | None = None,
        input_params: dict[str, Any] | None = None,
        *,
        input_data_format: str = "qir.v1",
        output_data_format: str = _OUTPUT_FORMAT,
    ) -> DemoJob:
        """Simulate QIR locally; explicit shots overrides count/shots input params."""
        if input_data_format != "qir.v1" or output_data_format != _OUTPUT_FORMAT:
            raise ValueError("Demo targets require qir.v1 input and microsoft.quantum-results.v2 output")
        params = deepcopy(input_params) if input_params is not None else {}
        if not isinstance(params, dict):
            raise TypeError("input_params must be a dictionary")
        unsupported = params.keys() - {"count", "shots"}
        if unsupported:
            raise ValueError(f"Unsupported demo input_params: {sorted(unsupported)}")
        if shots is None:
            if "count" in params and "shots" in params and params["count"] != params["shots"]:
                raise ValueError("input_params contains conflicting count and shots values")
            shots = params.get("count", params.get("shots", 1))
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if not isinstance(input_data, (QirInputData, str, bytes)):
            raise TypeError("Demo targets accept QIR objects, LLVM IR text, or bitcode, not Circuit or QASM")
        # run_qir interprets bytes as bitcode, whereas the SDK also accepts UTF-8 LLVM IR bytes.
        if isinstance(input_data, bytes) and not input_data.startswith((b"BC\xc0\xde", b"\xde\xc0\x17\x0b")):
            input_data = input_data.decode("utf-8")
        results = run_qir(input_data, shots=shots, seed=self.workspace.seed, type="cpu")
        if len(results) != shots:
            raise RuntimeError(f"QDK simulator returned {len(results)} shots; expected {shots}")
        outcomes = [_measurement_outcome(shot) for shot in results]
        params["count"] = shots
        if "shots" in params:
            params["shots"] = shots
        job = DemoJob(self, name, params, outcomes)
        self.workspace.jobs[job.id] = job
        return job


class DemoWorkspace:
    """An in-memory workspace; no Azure credentials or clients are constructed."""

    def __init__(self, seed: int = 42) -> None:
        """Create a workspace using a reproducible simulation seed."""
        self.seed = seed
        self.jobs: dict[str, DemoJob] = {}
        self.subscription_id = DEMO_BACKEND_SETTINGS["subscription_id"]
        self.resource_group = DEMO_BACKEND_SETTINGS["resource_group"]
        self.name = DEMO_BACKEND_SETTINGS["workspace_name"]
        self.location = DEMO_BACKEND_SETTINGS["location"]

    @classmethod
    def from_connection_string(cls, connection_string: str, *, seed: int = 42) -> DemoWorkspace:
        """Accept only demo://local, which is not a real Azure connection string."""
        if connection_string != DEMO_CONNECTION_STRING:
            raise ValueError(f"DemoWorkspace accepts only {DEMO_CONNECTION_STRING!r}")
        return cls(seed=seed)

    @overload
    def get_targets(self, name: str, provider_id: str | None = None) -> DemoTarget: ...

    @overload
    def get_targets(self, name: None = None, provider_id: str | None = None) -> list[DemoTarget]: ...

    def get_targets(self, name: str | None = None, provider_id: str | None = None) -> DemoTarget | list[DemoTarget]:
        """Resolve any named target alias, or list the default demo.target target."""
        if name is None:
            return [DemoTarget(self, f"{provider_id or 'demo'}.target", provider_id)]
        if not name.strip():
            raise ValueError("target name must not be empty")
        return DemoTarget(self, name, provider_id)

    def get_job(self, job_id: str) -> DemoJob:
        """Retrieve a job from this workspace only, never from Azure."""
        if job_id not in self.jobs:
            raise ValueError(f"No demo job with id {job_id!r}")
        return self.jobs[job_id]

    def list_jobs(self) -> list[DemoJob]:
        """Return the jobs simulated in this workspace."""
        return list(self.jobs.values())


@contextmanager
def demo_azure_quantum(seed: int = 42) -> Generator[DemoWorkspace, None, None]:
    """Temporarily redirect SDK/chemistry workspace creation to local simulation.

    The chemistry backend is unmodified and still submits QIR. Constructors and
    credential creation are patched in memory and restored even on exceptions.
    These patches are process-wide: do not mix with real submissions or threads.
    Use the yielded workspace for SDK-style calls. Existing workspace/target
    objects and Workspace aliases imported before this context are not replaced.
    Importing this module alone does not activate the demo.
    """
    from qdk_chemistry.plugins.azure_quantum import circuit_executor  # noqa: PLC0415

    workspace = DemoWorkspace(seed=seed)
    with (
        patch.object(azure.quantum, "Workspace", return_value=workspace) as constructor,
        patch.object(circuit_executor, "Workspace", return_value=workspace),
        patch.object(circuit_executor, "create_credential", return_value=None),
    ):
        constructor.from_connection_string.side_effect = lambda connection_string: DemoWorkspace.from_connection_string(
            connection_string, seed=seed
        )
        yield workspace
