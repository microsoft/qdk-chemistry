"""End-to-end tests for the RDKit geometry sample workflow."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import pytest

from .test_sample_workflow_utils import (
    _collect_output_lines,
    _extract_float,
    _run_workflow,
    _skip_for_mpi_failure,
)


def test_sample_rdkit_geometry():
    """Execute the RDKit geometry sample and validate reported SCF energy."""
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "examples/rdkit/sample_rdkit_geometry.py"]

    result = _run_workflow(cmd, repo_root)

    if result.returncode != 0 and "ModuleNotFoundError: No module named 'rdkit'" in result.stderr:
        pytest.skip("Skipping: RDKit not installed")
    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            "sample_rdkit_geometry.py exited with "
            f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    lines = _collect_output_lines(result)

    # Verify the script outputs molecular summary information
    assert any("Number of atoms" in line for line in lines), "Expected molecular summary not found in output."

    # Extract and validate the SCF energy
    output_text = "\n".join(lines)
    scf_energy = _extract_float(r"SCF Energy: ([+\-0-9.]+) Hartree", output_text)

    # The UFF-derived water geometry is not a fully optimized, high-accuracy structure, but
    # for this fixed approximate geometry the SCF energy is deterministic and used as a
    # precise regression reference.
    # FYI: optimized geometry has energy -76.0270535 Hartree at this level of theory
    reference_scf_energy = -76.02319617

    assert np.isclose(scf_energy, reference_scf_energy, atol=1e-6)
