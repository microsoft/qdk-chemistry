"""Test for circuit executor in QDK/Chemistry Azure Quantum plugin."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json

import pytest

from qdk_chemistry.data import Circuit, QuantumErrorProfile

pytest.importorskip("azure.quantum", reason="azure-quantum is not installed")

from qdk_chemistry.plugins.azure_quantum import circuit_executor
from qdk_chemistry.plugins.azure_quantum.circuit_executor import (
    _DEFAULT_EMULATION_SETTINGS,
    _WORKSPACE_ENV_VARS,
    AzureQuantumEmulator,
    _process_raw_results,
)


class TestProcessRawResults:
    """Test Azure Quantum histogram conversion."""

    def test_scalar_outcomes(self):
        """Scalar outcomes from single-bit measurements are accepted."""
        raw_results = {"0": {"outcome": 0, "count": 3}, "1": {"outcome": 1, "count": 2}}

        counts, loss = _process_raw_results(raw_results)

        assert counts == {"0": 3, "1": 2}
        assert loss == {}

    def test_sequence_outcomes_with_loss(self):
        """Sequence outcomes are reversed to qubit-0-rightmost order, with loss marked 'L'."""
        raw_results = {
            "[0, 1]": {"outcome": [0, 1], "count": 3},
            "[1, -]": {"outcome": [1, "-"], "count": 2},
        }

        counts, loss = _process_raw_results(raw_results)

        assert counts == {"10": 3}
        assert loss == {"L1": 2}


@pytest.fixture
def test_circuit_1() -> Circuit:
    """Create a test circuit."""
    return Circuit(
        qasm="""
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        """,
    )


class TestAzureQuantumEmulatorCircuitExecutor:
    """Test suite for Azure Quantum Emulator circuit executor."""

    def test_initialization(self):
        """Test initialization of the executor."""
        executor = AzureQuantumEmulator()
        emulation_settings = json.loads(executor.settings().get("emulation_settings"))
        assert emulation_settings["seed"] == 42
        assert emulation_settings["simulationType"] == "cliffordrounding"

    def test_workspace_coordinates_from_constructor(self):
        """Connection arguments override the environment defaults."""
        executor = AzureQuantumEmulator(
            subscription_id="sub",
            resource_group="rg",
            workspace_name="ws",
            location="location",
            auth_mode="default",
        )
        assert executor.settings().get("subscription_id") == "sub"
        assert executor.settings().get("resource_group") == "rg"
        assert executor.settings().get("workspace_name") == "ws"
        assert executor.settings().get("location") == "location"
        assert executor.settings().get("auth_mode") == "default"

    def test_missing_workspace_configuration(self, test_circuit_1: Circuit, monkeypatch: pytest.MonkeyPatch):
        """An unconfigured executor reports which workspace settings are missing."""
        for env_var in _WORKSPACE_ENV_VARS.values():
            monkeypatch.delenv(env_var, raising=False)
        executor = AzureQuantumEmulator()
        with pytest.raises(ValueError, match="Azure Quantum target cannot be resolved"):
            executor.run(test_circuit_1, shots=10)

    def test_circuit_executor_with_error_profile(
        self, test_circuit_1: Circuit, simple_error_profile: QuantumErrorProfile
    ):
        """Test that passing a noise profile raises NotImplementedError."""
        executor = AzureQuantumEmulator()
        with pytest.raises(NotImplementedError, match="Custom noise profiles are not yet supported"):
            executor.run(test_circuit_1, shots=10, noise=simple_error_profile)


class _FakeJob:
    """Stand-in for an Azure Quantum job that records what was submitted."""

    def __init__(self, submit_kwargs: dict):
        self.id = "fake-job-id"
        self.submit_kwargs = submit_kwargs
        self.requested_attachments: list[str] = []

    def get_results_histogram(self, timeout_secs: int) -> dict:
        """Return a fixed two-qubit histogram."""
        self.timeout_secs = timeout_secs
        return {"[0, 0]": {"outcome": [0, 0], "count": 6}, "[1, 1]": {"outcome": [1, 1], "count": 4}}

    def download_attachment(self, name: str) -> bytes:
        """Return deterministic bytes for *name*."""
        self.requested_attachments.append(name)
        return f"contents of {name}".encode()


class _FakeTarget:
    def __init__(self, name: str):
        self.name = name
        self.job: _FakeJob | None = None

    def submit(self, **kwargs) -> _FakeJob:
        """Record the submission and hand back a fake job."""
        self.job = _FakeJob(kwargs)
        return self.job


class _FakeWorkspace:
    """Records the coordinates it was constructed with."""

    last: _FakeWorkspace | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.target: _FakeTarget | None = None
        _FakeWorkspace.last = self

    def get_targets(self, name: str) -> _FakeTarget:
        """Return a fake target for *name*."""
        self.target = _FakeTarget(name)
        return self.target


@pytest.fixture
def fake_workspace(monkeypatch: pytest.MonkeyPatch):
    """Replace the SDK Workspace and credential so no network or auth is needed."""
    monkeypatch.setattr(circuit_executor, "Workspace", _FakeWorkspace)
    monkeypatch.setattr(circuit_executor, "create_credential", lambda auth_mode: f"credential:{auth_mode}")
    return _FakeWorkspace


@pytest.fixture
def configured_executor() -> AzureQuantumEmulator:
    """An executor with complete, fake connection settings."""
    return AzureQuantumEmulator(
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
        location="location",
        target_name="fake.emulator",
    )


class TestAzureQuantumEmulatorSubmission:
    """Exercise the submission path against a faked Azure Quantum SDK."""

    def test_workspace_built_from_settings(
        self, fake_workspace, configured_executor: AzureQuantumEmulator, test_circuit_1: Circuit
    ):
        """The workspace is constructed from the connection settings and the chosen credential."""
        configured_executor.run(test_circuit_1, shots=10)

        assert fake_workspace.last.kwargs == {
            "subscription_id": "sub",
            "resource_group": "rg",
            "name": "ws",
            "location": "location",
            "credential": "credential:azure-cli",
        }
        assert fake_workspace.last.target.name == "fake.emulator"

    def test_submission_payload(
        self, fake_workspace, configured_executor: AzureQuantumEmulator, test_circuit_1: Circuit
    ):
        """Shots, formats, and emulation settings reach target.submit()."""
        configured_executor.run(test_circuit_1, shots=10)

        submitted = fake_workspace.last.target.job.submit_kwargs
        assert submitted["shots"] == 10
        assert submitted["input_data_format"] == "qir.v1"
        assert submitted["output_data_format"] == "microsoft.quantum-results.v2"
        assert submitted["input_params"]["emulationSettings"] == _DEFAULT_EMULATION_SETTINGS

    def test_results_and_metadata(
        self, fake_workspace, configured_executor: AzureQuantumEmulator, test_circuit_1: Circuit
    ):
        """Histogram counts are converted and the job id is surfaced."""
        result = configured_executor.run(test_circuit_1, shots=10)

        assert result.bitstring_counts == {"00": 6, "11": 4}
        assert result.total_shots == 10
        metadata = result.get_executor_metadata()
        assert metadata["job_id"] == fake_workspace.last.target.job.id
        assert metadata["saved_attachments"] == []

    def test_attachments_saved(self, fake_workspace, test_circuit_1: Circuit, tmp_path):
        """Named attachments are written into output_dir."""
        executor = AzureQuantumEmulator(
            subscription_id="sub",
            resource_group="rg",
            workspace_name="ws",
            location="location",
            target_name="fake.emulator",
            output_dir=str(tmp_path),
            attachments=["rawOutputData"],
        )

        result = executor.run(test_circuit_1, shots=10)

        assert fake_workspace.last.target.job.requested_attachments == ["rawOutputData"]
        assert result.get_executor_metadata()["saved_attachments"] == [str(tmp_path / "rawOutputData")]
        assert (tmp_path / "rawOutputData").read_bytes() == b"contents of rawOutputData"

    def test_attachment_name_cannot_escape_output_dir(self, fake_workspace, test_circuit_1: Circuit, tmp_path):
        """A traversing attachment name is written inside output_dir, not above it."""
        output_dir = tmp_path / "artifacts"
        executor = AzureQuantumEmulator(
            subscription_id="sub",
            resource_group="rg",
            workspace_name="ws",
            location="location",
            target_name="fake.emulator",
            output_dir=str(output_dir),
            attachments=["../escaped"],
        )

        result = executor.run(test_circuit_1, shots=10)

        assert fake_workspace.last.target.job.requested_attachments == ["../escaped"]
        assert result.get_executor_metadata()["saved_attachments"] == [str(output_dir / "escaped")]
        assert not (tmp_path / "escaped").exists()
