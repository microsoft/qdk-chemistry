"""End-to-end tests for the Qiskit IQPE sample workflows.

These tests ensure the public QPE examples continue to emit the expected
summary values when executed as scripts.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from .reference_tolerances import (
    estimator_energy_tolerance,
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
)
from .test_sample_workflow_utils import (
    _extract_float,
    _run_workflow,
    _skip_for_mpi_failure,
)


def test_qiskit_iqpe_model_hamiltonian():
    """Execute the non-commuting IQPE sample and validate reported results."""
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "examples/qiskit/iqpe_model_hamiltonian.py"]

    result = _run_workflow(cmd, repo_root)
    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            "qpe_model_hamiltonian.py exited with "
            f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    phases = re.findall(r"Phase fraction φ \(measured\): ([0-9.+-eE]+)", result.stdout)
    assert phases == ["0.140625", "0.988770"], f"Unexpected phase fractions: {phases}"

    energies = [float(val) for val in re.findall(r"Estimated energy: ([+\-0-9.]+) Hartree", result.stdout)]
    assert np.allclose(
        energies, [1.12500000, -0.08984375], rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance
    )


def test_qiskit_iqpe_no_trotter():
    """Execute the exact-evolution IQPE sample and validate reported energies."""
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "examples/qiskit/iqpe_no_trotter.py"]

    result = _run_workflow(cmd, repo_root)
    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            f"qpe_no_trotter.py exited with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    phase_fraction = _extract_float(r"Phase fraction φ \(measured\): ([0-9.+-eE]+)", result.stdout)
    assert 0.0 <= phase_fraction < 1.0

    estimated_electronic_energy = _extract_float(
        r"Estimated electronic energy: ([+\-0-9.eE]+) Hartree",
        result.stdout,
    )
    estimated_total_energy = _extract_float(
        r"Estimated total energy: ([+\-0-9.eE]+) Hartree",
        result.stdout,
    )
    reference_total_energy = _extract_float(
        r"Reference total energy \(CASCI\): ([+\-0-9.eE]+) Hartree",
        result.stdout,
    )
    reported_difference = _extract_float(
        r"Energy difference \(QPE - CASCI\): ([+\-0-9.eE]+) Hartree",
        result.stdout,
    )

    assert math.isfinite(estimated_electronic_energy)
    assert math.isfinite(estimated_total_energy)
    assert np.isclose(
        estimated_total_energy - reference_total_energy,
        reported_difference,
        rtol=float_comparison_relative_tolerance,
        atol=estimator_energy_tolerance,
    )


def test_qiskit_iqpe_trotter():
    """Execute the Trotterized IQPE sample and validate reported energies."""
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [sys.executable, "examples/qiskit/iqpe_trotter.py"]

    result = _run_workflow(cmd, repo_root)
    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            f"qpe_trotter.py exited with {result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    phase_fraction = _extract_float(r"Phase fraction φ \(measured\): ([0-9.+-eE]+)", result.stdout)
    assert 0.0 <= phase_fraction < 1.0

    estimated_electronic_energy = _extract_float(
        r"Estimated electronic energy: ([+\-0-9.eE]+) Hartree",
        result.stdout,
    )
    estimated_total_energy = _extract_float(
        r"Estimated total energy: ([+\-0-9.eE]+) Hartree",
        result.stdout,
    )
    reference_total_energy = _extract_float(
        r"Reference total energy \(CASCI\): ([+\-0-9.eE]+) Hartree",
        result.stdout,
    )
    reported_difference = _extract_float(
        r"Energy difference \(QPE - CASCI\): ([+\-0-9.eE]+) Hartree",
        result.stdout,
    )

    assert math.isfinite(estimated_electronic_energy)
    assert math.isfinite(estimated_total_energy)
    assert np.isclose(
        estimated_total_energy - reference_total_energy,
        reported_difference,
        rtol=float_comparison_relative_tolerance,
        atol=estimator_energy_tolerance,
    )
