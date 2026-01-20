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

import subprocess
import sys
from pathlib import Path
import importlib.util

import pytest

# Check if pennylane is available
_PENNYLANE_AVAILABLE = importlib.util.find_spec("pennylane") is not None
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


@pytest.mark.skipif(not _PENNYLANE_AVAILABLE, reason="Pennylane is not installed")
def test_pennylane_qpe_no_trotter():
    """Test the examples/interoperability/pennylane/qpe_no_trotter.py script."""
    script_path = EXAMPLES_DIR / "interoperability" / "pennylane" / "qpe_no_trotter.py"
    assert script_path.exists(), f"Script not found: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=script_path.parent,
    )

    assert result.returncode == 0, (
        f"Script failed with exit code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
