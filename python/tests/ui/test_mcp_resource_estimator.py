"""MCP circuit resource-estimation tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdk import qre

from qdk_chemistry import data
from qdk_chemistry.ui import tools as srv
from qdk_chemistry.ui.config import config


@pytest.fixture
def circuit_project(tmp_path):
    """Create a project containing a serialised circuit."""
    original_projects_dir = config.projects_dir
    original_cwd = Path.cwd()
    config.projects_dir = tmp_path / "projects"
    project_dir = config.projects_dir / "qre"
    project_dir.mkdir(parents=True)
    circuit = data.Circuit(
        qasm="""
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q;
            t q;
        """
    )
    circuit.to_json_file(project_dir / "input.circuit.json")
    yield "qre"
    os.chdir(original_cwd)
    config.projects_dir = original_projects_dir


def test_resource_estimation_mcp_signature():
    """The MCP contract exposes QRE inputs without algorithm or file plumbing."""
    assert list(inspect.signature(srv.run_resource_estimation).parameters) == [
        "project_name",
        "circuit_filename",
        "architecture",
        "physical_error_rate",
        "max_error",
        "gate_time_ns",
        "measurement_time_ns",
        "use_graph",
    ]


def test_evaluate_circuit_qre_calls_qdk_estimate(monkeypatch):
    """The helper passes the circuit's QRE application to QDK QRE."""
    circuit = MagicMock()
    application = object()
    circuit.get_qre_application.return_value = application
    estimate = MagicMock(return_value=[])
    monkeypatch.setattr(qre, "estimate", estimate)

    table, assumptions = srv._evaluate_circuit_qre(
        circuit,
        architecture="majorana",
        physical_error_rate=None,
        max_error=0.01,
        gate_time_ns=50,
        measurement_time_ns=100,
        use_graph=True,
        name="input.circuit",
    )

    assert table == []
    assert assumptions == {
        "architecture": "majorana",
        "physical_error_rate": 1e-5,
        "qec_scheme": "three_aux",
        "factory": "round_based",
        "max_error": 0.01,
        "use_graph": True,
    }
    circuit.get_qre_application.assert_called_once_with()
    assert estimate.call_args.kwargs["application"] is application
    assert estimate.call_args.kwargs["max_error"] == 0.01
    assert estimate.call_args.kwargs["use_graph"] is True


def test_run_resource_estimation_returns_structured_pareto_front(circuit_project, monkeypatch):
    """The MCP tool returns sorted, JSON-serialisable Pareto points."""
    table = [
        SimpleNamespace(qubits=2000, runtime=100, error=0.008),
        SimpleNamespace(qubits=1000, runtime=300, error=0.009),
    ]
    assumptions = {
        "architecture": "majorana",
        "physical_error_rate": 1e-5,
        "qec_scheme": "three_aux",
        "factory": "round_based",
        "max_error": 0.01,
        "use_graph": True,
    }
    evaluate = MagicMock(return_value=(table, assumptions))
    monkeypatch.setattr(srv, "_evaluate_circuit_qre", evaluate)

    result = srv.run_resource_estimation(
        project_name=circuit_project,
        circuit_filename="input.circuit.json",
    )

    assert result["status"] == "ok"
    assert result["result"] == {
        "circuit_filename": "input.circuit.json",
        "assumptions": assumptions,
        "pareto_front": {
            "point_count": 2,
            "min_physical_qubits": 1000,
            "max_physical_qubits": 2000,
            "min_runtime_ns": 100,
            "max_runtime_ns": 300,
            "points": [
                {"physical_qubits": 1000, "runtime_ns": 300, "error": 0.009},
                {"physical_qubits": 2000, "runtime_ns": 100, "error": 0.008},
            ],
        },
    }
    evaluate.assert_called_once()
