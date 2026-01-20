"""End-to-end tests for PennyLane interoperability sample workflows.

This module contains tests for PennyLane interoperability samples.
Tests are skipped if PennyLane is not installed.

See Also:
- test_sample_workflow_sci.py - Sparse-CI workflow tests
- test_sample_workflow_rdkit.py - RDKit geometry tests
- test_sample_workflow_qiskit.py - Qiskit IQPE tests

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import sys
from pathlib import Path

import pytest

from .test_sample_workflow_utils import _run_workflow, _skip_for_mpi_failure

# Check if PennyLane is available
_PENNYLANE_AVAILABLE = importlib.util.find_spec("pennylane") is not None


@pytest.mark.skipif(not _PENNYLANE_AVAILABLE, reason="PennyLane is not installed")
def test_pennylane_qpe_no_trotter():
    """Test the examples/interoperability/pennylane/qpe_no_trotter.py script."""
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "examples/interoperability/pennylane/qpe_no_trotter.py"]

    result = _run_workflow(cmd, repo_root)
    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            f"qpe_no_trotter.py exited with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
