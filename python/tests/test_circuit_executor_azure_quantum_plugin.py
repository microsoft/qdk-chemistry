"""Test for circuit executor in QDK/Chemistry Azure Quantum plugin."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from qdk_chemistry.data import Circuit, QuantumErrorProfile, SettingTypeMismatch

pytest.importorskip("azure.quantum", reason="azure-quantum is not installed")

from azure.quantum._client.models import JobDetails
from azure.quantum.job import Job, JobFailedWithResultsError

from qdk_chemistry.plugins.azure_quantum import circuit_executor
from qdk_chemistry.plugins.azure_quantum.circuit_executor import (
    _WORKSPACE_SETTINGS,
    AzureQuantumBackend,
    _process_raw_results,
)


class TestProcessRawResults:
    """Test Azure Quantum histogram conversion."""

    def test_scalar_outcomes(self):
        """Scalar outcomes from single-bit measurements are accepted."""
        raw_results = {"0": {"outcome": 0, "count": 3}, "1": {"outcome": 1, "count": 2}}

        counts, loss, failures = _process_raw_results(raw_results)

        assert counts == {"0": 3, "1": 2}
        assert loss == {}
        assert failures == {}

    def test_sequence_outcomes_with_loss(self):
        """Sequence outcomes are reversed to qubit-0-rightmost order, with loss marked 'L'."""
        raw_results = {
            "[0, 1]": {"outcome": [0, 1], "count": 3},
            "[1, -]": {"outcome": [1, "-"], "count": 2},
        }

        counts, loss, failures = _process_raw_results(raw_results)

        assert counts == {"10": 3}
        assert loss == {"L1": 2}
        assert failures == {}

    @pytest.mark.parametrize(
        "outcome",
        [
            {"Error": {"Name": "ExecutionFailure"}},
            [0, {"Error": {"Name": "ExecutionFailure"}}, 1],
            (1, {"Error": {"Name": "ExecutionFailure"}}),
            [{"Error": {"Name": "FirstError"}}, {"Error": {"Name": "SecondError"}}],
            ["-", {"Error": {"Name": "ExecutionFailure"}}],
            {"Error": {"Name": "OtherFailure", "Message": "Shot execution failed"}},
            2,
            "unexpected",
            None,
            [],
            [[0, 1]],
        ],
    )
    def test_failed_outcomes_are_recorded_separately(self, outcome):
        """An invalid outcome excludes the entire shot, preserving its raw details and count."""
        raw_results = {
            "valid": {"outcome": [0, 1], "count": 3},
            "loss": {"outcome": [1, "-"], "count": 2},
            "failed": {"outcome": outcome, "count": 5},
        }

        counts, loss, failures = _process_raw_results(raw_results)

        assert counts == {"10": 3}
        assert loss == {"L1": 2}
        assert failures == {"failed": {"outcome": outcome, "count": 5}}
        assert sum(counts.values()) + sum(loss.values()) + sum(e["count"] for e in failures.values()) == 10

    def test_failure_from_sdk_histogram(self, monkeypatch):
        """Parse an actual SDK histogram with only its blob download mocked."""
        outcome = {"Error": {"Name": "ExecutionFailure"}}
        payload = {
            "DataFormat": "microsoft.quantum-results.v2",
            "Results": [
                {
                    "Histogram": [
                        {"Display": "[0, 1]", "Outcome": [0, 1], "Count": 3},
                        {"Display": "ExecutionFailure", "Outcome": outcome, "Count": 2},
                    ]
                }
            ],
        }
        job = Job(
            workspace=Mock(),
            job_details=JobDetails(
                id="fake-job-id",
                name="test",
                container_uri="fake-container-uri",
                input_data_format="qir.v1",
                provider_id="test",
                target="test",
                status="Succeeded",
                output_data_format="microsoft.quantum-results.v2",
                output_data_uri="fake-results-uri",
            ),
        )
        monkeypatch.setattr(job, "download_data", lambda _uri: json.dumps(payload).encode("utf-8"))

        counts, loss, failures = _process_raw_results(job.get_results_histogram())

        assert counts == {"10": 3}
        assert loss == {}
        assert failures == {"ExecutionFailure": {"outcome": outcome, "count": 2}}


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


class TestAzureQuantumBackendCircuitExecutor:
    """Test suite for the Azure Quantum circuit executor."""

    def test_initialization(self):
        """Workspace settings are empty until the caller supplies them."""
        executor = AzureQuantumBackend()

        for key in _WORKSPACE_SETTINGS:
            assert executor.settings().get(key) == ""
        assert executor.settings().get("input_params") == "{}"
        assert executor.settings().get("auth_mode") == "azure-cli"

    def test_workspace_coordinates_from_constructor(self):
        """Connection arguments are stored in settings."""
        executor = AzureQuantumBackend(
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

    def test_input_params_accepts_json_string(self):
        """A JSON object string is stored as-is without double encoding."""
        payload = {"someOption": False, "seed": 7}
        encoded = json.dumps(payload)

        from_string = AzureQuantumBackend(input_params=encoded).settings().get("input_params")

        assert from_string == encoded
        assert json.loads(from_string) == payload

    def test_input_params_rejects_dict(self):
        """The constructor and settings both require a JSON string, not a dict."""
        payload = {"someOption": False, "seed": 7}

        with pytest.raises(SettingTypeMismatch, match="Type mismatch for setting 'input_params'"):
            AzureQuantumBackend(input_params=payload)

        executor = AzureQuantumBackend()
        with pytest.raises(SettingTypeMismatch, match="Type mismatch for setting 'input_params'"):
            executor.settings().set("input_params", payload)

    def test_settings_reach_the_workspace(self, fake_workspace, test_circuit_1: Circuit):
        """Configured values are used to build the workspace and target."""
        AzureQuantumBackend(
            subscription_id="sub",
            resource_group="rg",
            workspace_name="ws",
            location="location",
            target_name="fake.emulator",
        ).run(test_circuit_1, shots=10)

        assert fake_workspace.last.kwargs["subscription_id"] == "sub"
        assert fake_workspace.last.kwargs["resource_group"] == "rg"
        assert fake_workspace.last.kwargs["name"] == "ws"
        assert fake_workspace.last.kwargs["location"] == "location"
        assert fake_workspace.last.target.name == "fake.emulator"

    def test_missing_workspace_configuration(self, test_circuit_1: Circuit):
        """An unconfigured executor reports which workspace settings are missing."""
        executor = AzureQuantumBackend()
        with pytest.raises(ValueError, match="Azure Quantum target cannot be resolved"):
            executor.run(test_circuit_1, shots=10)

    def test_circuit_executor_with_error_profile(
        self, test_circuit_1: Circuit, simple_error_profile: QuantumErrorProfile
    ):
        """Test that passing a noise profile raises NotImplementedError."""
        executor = AzureQuantumBackend()
        with pytest.raises(NotImplementedError, match="Custom noise profiles are not yet supported"):
            executor.run(test_circuit_1, shots=10, noise=simple_error_profile)


class _FakeJob:
    """Stand-in for an Azure Quantum job that records what was submitted."""

    def __init__(self, submit_kwargs: dict):
        self.id = "fake-job-id"
        self.submit_kwargs = submit_kwargs
        self.requested_attachments: list[str] = []
        self.calls: list[str] = []

    def wait_until_completed(self, timeout_secs: int) -> None:
        """Record the explicit wait and its timeout."""
        self.timeout_secs = timeout_secs
        self.calls.append("wait_until_completed")

    def get_results_histogram(self) -> dict:
        """Return a fixed two-qubit histogram."""
        self.calls.append("get_results_histogram")
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
def configured_executor() -> AzureQuantumBackend:
    """An executor with complete, fake connection settings."""
    return AzureQuantumBackend(
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
        location="location",
        target_name="fake.emulator",
    )


class TestAzureQuantumBackendSubmission:
    """Exercise the submission path against a faked Azure Quantum SDK."""

    def test_workspace_built_from_settings(
        self, fake_workspace, configured_executor: AzureQuantumBackend, test_circuit_1: Circuit
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
        self, fake_workspace, configured_executor: AzureQuantumBackend, test_circuit_1: Circuit
    ):
        """Shots, formats, and input parameters reach target.submit()."""
        configured_executor.settings().set("input_params", json.dumps({"extra": "value"}))

        configured_executor.run(test_circuit_1, shots=10)

        submitted = fake_workspace.last.target.job.submit_kwargs
        assert submitted["shots"] == 10
        assert submitted["input_data_format"] == "qir.v1"
        assert submitted["output_data_format"] == "microsoft.quantum-results.v2"
        assert submitted["input_params"] == {"extra": "value"}

    def test_invalid_json_input_params(self, fake_workspace, configured_executor, test_circuit_1):
        """Malformed JSON reports the setting name and preserves the parsing error."""
        configured_executor.settings().set("input_params", "{bad")

        with pytest.raises(ValueError, match="Invalid JSON in 'input_params'") as error:
            configured_executor.run(test_circuit_1, shots=10)

        assert isinstance(error.value.__cause__, json.JSONDecodeError)
        assert fake_workspace.last.target.job is None

    @pytest.mark.parametrize("payload", ["[]", "null", '"text"', "42", "true"])
    def test_non_object_input_params(self, fake_workspace, configured_executor, test_circuit_1, payload):
        """input_params requires an object rather than an array or scalar."""
        configured_executor.settings().set("input_params", payload)

        with pytest.raises(ValueError, match="'input_params' must be a JSON object"):
            configured_executor.run(test_circuit_1, shots=10)

        assert fake_workspace.last.target.job is None

    def test_results_and_metadata(
        self, fake_workspace, configured_executor: AzureQuantumBackend, test_circuit_1: Circuit
    ):
        """Wait before retrieving the histogram, converting counts, and surfacing the job id."""
        configured_executor.settings().set("timeout_secs", 42)

        result = configured_executor.run(test_circuit_1, shots=10)

        job = fake_workspace.last.target.job
        assert job.timeout_secs == 42
        assert job.calls == ["wait_until_completed", "get_results_histogram"]
        assert result.bitstring_counts == {"00": 6, "11": 4}
        assert result.total_shots == 10
        metadata = result.get_executor_metadata()
        assert metadata["job_id"] == fake_workspace.last.target.job.id
        assert metadata["saved_attachments"] == []
        assert metadata["failed_shots"] == 0
        assert metadata["failed_results"] == {}

    def test_mixed_results_preserve_failures(self, fake_workspace, configured_executor, test_circuit_1, monkeypatch):
        """Clean results survive failed shots, whose counts and details remain inspectable."""
        raw_results = {
            "valid": {"outcome": [0, 1], "count": 6},
            "loss": {"outcome": [1, "-"], "count": 1},
            "failed": {"outcome": {"Error": {"Name": "ExecutionFailure"}}, "count": 3},
        }
        monkeypatch.setattr(_FakeJob, "get_results_histogram", lambda _self: raw_results)
        warnings: list[str] = []
        monkeypatch.setattr(circuit_executor.Logger, "warn", warnings.append)

        result = configured_executor.run(test_circuit_1, shots=10)

        assert result.bitstring_counts == {"10": 6}
        assert result.loss_bitstrings == {"L1": 1}
        assert result.total_shots == 10
        metadata = result.get_executor_metadata()
        assert metadata["job_id"] == fake_workspace.last.target.job.id
        assert metadata["results"] == raw_results
        assert metadata["failed_shots"] == 3
        assert metadata["failed_results"] == {"failed": raw_results["failed"]}
        assert len(warnings) == 1
        assert "3 failed shots" in warnings[0]
        assert "ExecutionFailure" in warnings[0]

    @pytest.mark.parametrize("remaining_outcome", [None, [1, "-"]])
    def test_no_clean_results_raise_with_failure_details(
        self, fake_workspace, configured_executor, test_circuit_1, monkeypatch, tmp_path, *, remaining_outcome
    ):
        """An unusable job stops estimation but retains job details and downloaded attachments."""
        raw_results = {
            "failed": {"outcome": {"Error": {"Name": "ExecutionFailure"}}, "count": 1},
        }
        if remaining_outcome is not None:
            raw_results["loss"] = {"outcome": remaining_outcome, "count": 1}
        monkeypatch.setattr(_FakeJob, "get_results_histogram", lambda _self: raw_results)
        configured_executor.settings().set("output_dir", str(tmp_path))
        configured_executor.settings().set("attachments", ["output"])

        with pytest.raises(JobFailedWithResultsError, match="no valid measurement results") as error:
            configured_executor.run(test_circuit_1, shots=len(raw_results))

        details = error.value.get_failure_results()
        assert details["job_id"] == fake_workspace.last.target.job.id
        assert details["failed_shots"] == 1
        assert details["failed_results"] == {"failed": raw_results["failed"]}
        assert details["results"] == raw_results
        assert details["saved_attachments"] == [str(tmp_path / "output")]
        assert (tmp_path / "output").read_bytes() == b"contents of output"
        assert "ExecutionFailure" in str(error.value)

    @pytest.mark.parametrize("raw_results", [{}, {"loss": {"outcome": [1, "-"], "count": 2}}])
    def test_empty_or_loss_only_results_raise(
        self, fake_workspace, configured_executor, test_circuit_1, monkeypatch, raw_results
    ):
        """No clean counts must never be passed to an estimator as usable measurements."""
        monkeypatch.setattr(_FakeJob, "get_results_histogram", lambda _self: raw_results)

        with pytest.raises(JobFailedWithResultsError, match="no valid measurement results") as error:
            configured_executor.run(test_circuit_1, shots=2)

        details = error.value.get_failure_results()
        assert details["job_id"] == fake_workspace.last.target.job.id
        assert details["results"] == raw_results

    def test_attachments_saved(self, fake_workspace, test_circuit_1: Circuit, tmp_path):
        """Named attachments are written into output_dir."""
        executor = AzureQuantumBackend(
            subscription_id="sub",
            resource_group="rg",
            workspace_name="ws",
            location="location",
            target_name="fake.emulator",
            output_dir=str(tmp_path),
            attachments=["output"],
        )

        result = executor.run(test_circuit_1, shots=10)

        assert fake_workspace.last.target.job.requested_attachments == ["output"]
        assert result.get_executor_metadata()["saved_attachments"] == [str(tmp_path / "output")]
        assert (tmp_path / "output").read_bytes() == b"contents of output"

    def test_attachment_name_cannot_escape_output_dir(self, fake_workspace, test_circuit_1: Circuit, tmp_path):
        """A traversing attachment name is written inside output_dir, not above it."""
        output_dir = tmp_path / "artifacts"
        executor = AzureQuantumBackend(
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
