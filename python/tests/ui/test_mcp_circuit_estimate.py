"""MCP circuit estimation tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import inspect
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qdk_chemistry import data
from qdk_chemistry.ui import tools as srv
from qdk_chemistry.ui.config import config


@pytest.fixture
def circuit_project(tmp_path):
    """Create a project containing a serialized circuit."""
    original_projects_dir = config.projects_dir
    original_cwd = Path.cwd()
    config.projects_dir = tmp_path / "projects"
    project_dir = config.projects_dir / "estimate"
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
    yield "estimate"
    os.chdir(original_cwd)
    config.projects_dir = original_projects_dir


def test_estimate_circuit_mcp_signature():
    """The MCP contract accepts the circuit and JSON estimator parameters."""
    assert list(inspect.signature(srv.estimate_circuit).parameters) == [
        "project_name",
        "circuit_filename",
        "params",
    ]


def test_estimate_circuit_returns_raw_estimator_data(circuit_project, monkeypatch):
    """The tool forwards parameters and returns JSON-compatible estimator data."""
    params = {
        "qubitParams": {"name": "qubit_gate_ns_e3"},
        "qecScheme": {"name": "surface_code"},
        "errorBudget": 0.001,
    }
    result_data = {
        "status": "success",
        "logicalCounts": {"numQubits": 1, "tCount": 1},
    }
    estimate_result = MagicMock()
    estimate_result.data.return_value = result_data
    estimate = MagicMock(return_value=estimate_result)
    monkeypatch.setattr(data.Circuit, "estimate", estimate)

    result = srv.estimate_circuit(
        project_name=circuit_project,
        circuit_filename="input.circuit.json",
        params=params,
    )

    assert result == {"status": "ok", "result": result_data}
    estimate.assert_called_once_with(params)
    estimate_result.data.assert_called_once_with()


def test_estimate_circuit_preserves_batch_results(circuit_project, monkeypatch):
    """A parameter batch returns estimator items in their original order."""
    params = [
        {"qubitParams": {"name": "qubit_gate_ns_e3"}},
        {"qubitParams": {"name": "qubit_maj_ns_e4"}},
    ]
    result_data = [
        {"status": "success", "physicalCounts": {"physicalQubits": 100}},
        {"status": "success", "physicalCounts": {"physicalQubits": 200}},
    ]
    estimate_result = MagicMock()
    estimate_result.data.return_value = result_data
    monkeypatch.setattr(data.Circuit, "estimate", MagicMock(return_value=estimate_result))

    result = srv.estimate_circuit(
        project_name=circuit_project,
        circuit_filename="input.circuit.json",
        params=params,
    )

    assert result == {"status": "ok", "result": result_data}
