"""MCP circuit estimation tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import inspect
import json
import os
from pathlib import Path

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


@pytest.mark.parametrize("params", [None, {"max_error": 0.001, "name": "custom"}])
def test_estimate_circuit_returns_qre_frontier(circuit_project, params):
    """The tool accepts QRE v3 options and returns a JSON-compatible physical frontier."""
    result = srv.estimate_circuit(
        project_name=circuit_project,
        circuit_filename="input.circuit.json",
        params=params,
    )

    assert result["status"] == "ok"
    assert result["result"]
    assert all(row["qubits"] > 0 and row["runtime"] > 0 for row in result["result"])
    if params is not None:
        assert all(row["name"] == "custom" for row in result["result"])
    json.dumps(result)
