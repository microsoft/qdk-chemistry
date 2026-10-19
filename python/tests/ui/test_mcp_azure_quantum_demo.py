"""MCP entry points for credential-free Azure Quantum demo execution."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from unittest.mock import Mock

import pytest

from qdk_chemistry import data
from qdk_chemistry.ui import tools
from qdk_chemistry.ui.config import config


@pytest.fixture
def demo_project(tmp_path, monkeypatch):
    """Provide serializable inputs without a workspace connection."""
    monkeypatch.setattr(config, "projects_dir", tmp_path)
    project = tmp_path / "demo"
    project.mkdir()
    data.Circuit(
        qasm='OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; output bit[2] c; h q[0]; cx q[0], q[1]; c = measure q;'
    ).to_json_file(project / "bell.circuit.json")
    data.Circuit(qasm='OPENQASM 3.0; include "stdgates.inc"; qubit q; id q;').to_json_file(
        project / "state.circuit.json"
    )
    data.QubitOperator(pauli_strings=["Z"], coefficients=[-0.5]).to_json_file(project / "z.qubit_hamiltonian.json")
    return project


@pytest.mark.usefixtures("demo_project")
@pytest.mark.parametrize(("shots", "seed"), [(0, 42), (-1, 42), (True, 42), (1, -1), (1, True)])
def test_circuit_demo_rejects_invalid_sampling(monkeypatch, shots, seed):
    runner = Mock()
    monkeypatch.setattr(tools, "_run_azure_quantum_demo", runner)
    result = tools.run_azure_quantum_demo("demo", "bell.circuit.json", shots, "out.json", seed=seed)
    assert result["status"] == "error"
    assert "Invalid" in result["message"]
    runner.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_bits": 0},
        {"num_bits": True},
        {"shots_per_bit": 0},
        {"shots_per_bit": True},
        {"evolution_time": 0},
        {"evolution_time": float("nan")},
        {"evolution_time": float("inf")},
        {"evolution_time": True},
        {"seed": -1},
    ],
)
@pytest.mark.usefixtures("demo_project")
def test_iqpe_demo_rejects_invalid_settings(monkeypatch, overrides):
    runner = Mock()
    monkeypatch.setattr(tools, "_run_azure_quantum_demo", runner)
    arguments = {"num_bits": 2, "evolution_time": math.pi, **overrides}
    result = tools.run_iqpe_azure_quantum_demo(
        "demo", "state.circuit.json", "z.qubit_hamiltonian.json", "out.json", **arguments
    )
    assert result["status"] == "error"
    runner.assert_not_called()


@pytest.mark.usefixtures("demo_project")
def test_iqpe_demo_sets_explicit_time_and_builder(monkeypatch):
    runner = Mock(return_value={"local_demo": True})
    monkeypatch.setattr(tools, "_run_azure_quantum_demo", runner)
    result = tools.run_iqpe_azure_quantum_demo(
        "demo",
        "state.circuit.json",
        "z.qubit_hamiltonian.json",
        "out.json",
        num_bits=2,
        evolution_time=math.pi,
        shots_per_bit=5,
        unitary_builder_settings={"time": 99.0, "num_divisions": 4, "order": 2},
    )
    assert result["status"] == "ok"
    settings = runner.call_args.args[3]
    assert settings["shots_per_bit"] == 5
    builder = settings["qpe_circuit_builder"]
    assert builder["algorithm_name"] == "qiskit_iterative"
    assert builder["num_bits"] == 2
    assert builder["unitary_builder"] == {"algorithm_name": "trotter", "time": math.pi, "num_divisions": 4, "order": 2}


def test_circuit_demo_runs_qir_in_isolated_process(demo_project):
    pytest.importorskip("azure.quantum")
    from qdk_chemistry.plugins.azure_quantum import circuit_executor  # noqa: PLC0415

    workspace_constructor = circuit_executor.Workspace
    result = tools.run_azure_quantum_demo("demo", "bell.circuit.json", 32, "bell-result.json")
    assert result["status"] == "ok", result
    payload = result["result"]
    assert payload["backend_display_name"] == "Azure Quantum local demo target"
    assert "once in Methods" in payload["reporting_guidance"]
    assert payload["local_demo"] is True
    assert payload["submitted_to_azure"] is False
    assert payload["simulator"] == "qdk-full-state-cpu"
    assert len(payload["jobs"]) == 1
    assert payload["jobs"][0]["job_id"].startswith("demo-")
    assert payload["jobs"][0]["target"] == "demo.target"
    counts = payload["data"]["bitstring_counts"]
    assert set(counts) == {"00", "11"}
    assert sum(counts.values()) == 32
    saved = data.CircuitExecutorData.from_json_file(demo_project / payload["output_filename"])
    assert saved.bitstring_counts == counts
    assert saved.executor == "azure_quantum_backend"
    assert circuit_executor.Workspace is workspace_constructor

    existing = tools.run_azure_quantum_demo("demo", "bell.circuit.json", 32, "bell-result.json")
    assert existing["status"] == "exists"
    repeated = tools.run_azure_quantum_demo("demo", "bell.circuit.json", 32, "bell-result.json", overwrite=True)
    assert repeated["status"] == "ok", repeated
    assert repeated["result"]["data"]["bitstring_counts"] == counts


def test_iqpe_demo_runs_each_phase_bit_through_azure_backend(demo_project):
    pytest.importorskip("azure.quantum")
    pytest.importorskip("qiskit")
    result = tools.run_iqpe_azure_quantum_demo(
        "demo",
        "state.circuit.json",
        "z.qubit_hamiltonian.json",
        "iqpe.json",
        num_bits=2,
        evolution_time=math.pi,
    )
    assert result["status"] == "ok", result
    payload = result["result"]
    assert payload["backend_display_name"] == "Azure Quantum local demo target"
    assert "once in Methods" in payload["reporting_guidance"]
    assert payload["submitted_to_azure"] is False
    assert len(payload["jobs"]) == 2
    assert all(job["target"] == "demo.target" and job["status"] == "Succeeded" for job in payload["jobs"])
    assert all(sum(entry["count"] for entry in job["histogram"].values()) == 3 for job in payload["jobs"])
    saved = data.QpeResult.from_json_file(demo_project / payload["output_filename"])
    assert saved.bitstring_msb_first == "01"
    assert saved.phase_fraction == pytest.approx(0.25)
    assert saved.raw_energy == pytest.approx(-0.5)


def test_demo_worker_failure_returns_structured_error(demo_project):
    pytest.importorskip("azure.quantum")
    result = tools.run_azure_quantum_demo("demo", "missing.circuit.json", 1, "out.json")
    assert result["status"] == "error"
    assert not (demo_project / "out.circuit_executor_data.json").exists()


def test_demo_tools_are_discoverable():
    names = tools.list_tools("qpe")["result"]["categories"]["qpe"]
    assert "run_azure_quantum_demo" in names
    assert "run_iqpe_azure_quantum_demo" in names
