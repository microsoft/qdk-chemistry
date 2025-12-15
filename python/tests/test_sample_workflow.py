"""End-to-end tests for the sample workflows.

The sparse-CI finder scenarios verify the MACIS-integrated pipeline, while the
IQPE samples ensure the public QPE examples continue to emit the expected
summary values when executed as scripts.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from .reference_tolerances import (
    estimator_energy_tolerance,
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
    sci_energy_tolerance,
)

################################################################################
# Set up and utility functions
################################################################################


def _truthy_env(name: str) -> bool:
    """Return True when ``name`` is set to a truthy value."""
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


_RUNNING_IN_CI = _truthy_env("TF_BUILD")
_RUN_MACIS_WORKFLOW = _truthy_env("QDK_CHEMISTRY_RUN_MACIS_WORKFLOW")


def _run_workflow(cmd, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Execute the workflow CLI with coverage-friendly defaults."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )


def _skip_for_mpi_failure(result: subprocess.CompletedProcess[str]) -> None:
    """Skip the test when MPI cannot initialize."""
    mpi_err_indicators = [
        "PMIx server's listener thread failed to start",
        "ompi_mpi_init: ompi_rte_init failed",
        "Unable to start a daemon on the local node",
        "MPI_INIT failed",
        "pmix_ifinit: socket() failed",
        "opal_ifinit: socket() failed with errno=1",
    ]
    if any(ind in result.stderr for ind in mpi_err_indicators):
        pytest.skip("Skipping: MPI environment not available for QPE workflow")


def _collect_output_lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    """Return combined stdout/stderr lines for downstream assertions."""
    return (result.stdout + "\n" + result.stderr).splitlines()


def _extract_float(pattern: str, text: str) -> float:
    """Extract the first floating-point value matching ``pattern`` from ``text``."""
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"Pattern '{pattern}' not found in output.\n{text}")
    return float(match.group(1))


def _find_line(predicate: Callable[[str], bool], lines: list[str]) -> str:
    """Return the first line satisfying ``predicate`` or raise."""
    for line in lines:
        if predicate(line):
            return line
    raise AssertionError("Expected line not found in workflow output.")


def _extract_sparse_ci_summary(lines: list[str]) -> tuple[int, float, float]:
    """Parse the sparse-CI summary line and return determinant count, energy, and ΔE."""
    summary_line = _find_line(lambda line: "Sparse CI finder (" in line, lines)
    match = re.search(
        r"Sparse CI finder \((\d+) dets\) = ([\-0-9.]+) Hartree \(ΔE = ([\-0-9.]+) mHartree\)",
        summary_line,
    )
    if match is None:
        raise AssertionError(f"Unable to parse sparse CI finder line: {summary_line}")
    det_count = int(match.group(1))
    energy = float(match.group(2))
    delta_mhartree = float(match.group(3))
    return det_count, energy, delta_mhartree


def _assert_warning_constraints(lines: list[str], expected_warning: str | None, expect_no_warnings: bool) -> None:
    """Validate warning presence/absence expectations for a workflow run."""
    if expected_warning is not None:
        warning_line = _find_line(lambda line: expected_warning in line, lines)
        assert "[warning]" in warning_line, "Expected warning line missing logging prefix."
    if expect_no_warnings:
        assert all("[warning]" not in line for line in lines), (
            "Unexpected warning emitted by workflow.\nOutput:\n" + "\n".join(lines)
        )


################################################################################
# SCI workflow testing
################################################################################


@dataclass(frozen=True)
class WorkflowCase:
    """Descriptor for a workflow CLI regression test."""

    identifier: str
    script: str
    args: list[str]
    cwd_relative: Path
    expected_energy: float
    expected_det_count: int | None = None
    summary_det_count: int | None = None
    expected_warning: str | None = None
    expect_no_warnings: bool = False


TEST_CASES: tuple[WorkflowCase, ...] = (
    WorkflowCase(
        identifier="valence_overrides",
        script="sample_sci_workflow.py",
        args=[
            "--xyz",
            "data/water.structure.xyz",
            "--num-active-electrons",
            "6",
            "--num-active-orbitals",
            "6",
            "--initial-active-space-solver",
            "macis_cas",
        ],
        cwd_relative=Path("examples"),
        expected_energy=-76.03203471,
        expected_det_count=13,
        summary_det_count=13,
    ),
    WorkflowCase(
        identifier="valence_defaults",
        script="examples/sample_sci_workflow.py",
        args=[
            "--xyz",
            "examples/data/water.structure.xyz",
            "--initial-active-space-solver",
            "macis_cas",
        ],
        cwd_relative=Path("."),
        expected_energy=-76.02623746,
        expect_no_warnings=True,
    ),
    WorkflowCase(
        identifier="valence_autocas_fallback",
        script="examples/sample_sci_workflow.py",
        args=[
            "--xyz",
            "examples/data/water.structure.xyz",
            "--autocas",
            "--initial-active-space-solver",
            "macis_cas",
        ],
        cwd_relative=Path("."),
        expected_energy=-76.02623746,
        expected_warning="AutoCAS did not identify correlated orbitals; retaining the initial space.",
    ),
    WorkflowCase(
        identifier="valence_autocas_threshold",
        script="examples/sample_sci_workflow.py",
        args=[
            "--xyz",
            "examples/data/water.structure.xyz",
            "--initial-active-space-solver",
            "macis_cas",
            "--autocas",
            "--autocas-parameters",
            '{"entropy_threshold": 0.01}',
        ],
        cwd_relative=Path("."),
        expected_energy=-76.02623746,
        expect_no_warnings=True,
    ),
)


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda case: case.identifier)
def test_sample_sci_workflow_scenarios(case: WorkflowCase) -> None:
    """Exercise the sample workflow under several CLI configurations."""
    repo_root = Path(__file__).resolve().parents[2]
    cwd = repo_root / case.cwd_relative
    cmd = [sys.executable, case.script, *case.args]

    result = _run_workflow(cmd, cwd)

    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            "sample_sci_workflow.py exited with "
            f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    lines = _collect_output_lines(result)

    det_count, energy, _ = _extract_sparse_ci_summary(lines)
    assert np.isclose(energy, case.expected_energy, rtol=float_comparison_relative_tolerance, atol=sci_energy_tolerance)
    if case.expected_det_count is not None:
        assert det_count == case.expected_det_count

    if case.summary_det_count is not None:
        summary_line = _find_line(lambda line: "Stored wavefunction with" in line, lines)
        assert str(case.summary_det_count) in summary_line

    _assert_warning_constraints(lines, case.expected_warning, case.expect_no_warnings)


@pytest.mark.skipif(
    _RUNNING_IN_CI and not _RUN_MACIS_WORKFLOW,
    reason="Skipping MACIS ASCI workflow in CI pipeline; enable with QDK_CHEMISTRY_RUN_MACIS_WORKFLOW=1",
)
def test_sample_sci_workflow_macis_asci_autocas_with_limits():
    """Run workflow with MACIS ASCI initial solver and capped determinants."""
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "examples/sample_sci_workflow.py",
        "--xyz",
        "examples/data/water.structure.xyz",
        "--num-active-electrons",
        "10",
        "--num-active-orbitals",
        "20",
        "--initial-active-space-solver",
        "macis_asci",
        "--autocas",
        "--max-determinants",
        "100",
    ]

    result = _run_workflow(cmd, repo_root)
    if result.returncode != 0 and "Wavefunction didn't grow enough" in result.stderr:
        pytest.skip("Skipping: MACIS ASCI solver did not converge under current settings")

    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            "sample_sci_workflow.py exited with "
            f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    lines = _collect_output_lines(result)

    indices_line = _find_line(lambda line: "AutoCAS selected active space with indices:" in line, lines)
    assert "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]" in indices_line


################################################################################
# Sample RDKIT geometry workflow testing
################################################################################


@pytest.mark.xfail(reason="Skipping unimplemented examples/sample_rdkit_geometry.py test.")
def test_sample_rdkit_geometry():
    """Test the examples/sample_rdkit_geometry.py script."""
    # TODO: Need to implement this test (see https://github.com/microsoft/qdk-chemistry/issues/198)
    raise NotImplementedError("TODO: add sample_rdkit_geometry.py test.")


################################################################################
# Sample notebook testing
################################################################################


@pytest.mark.xfail(reason="Skipping unimplemented examples/factory_list.ipynb test.")
def test_factory_list():
    """Test the examples/factory_list.ipynb script."""
    # TODO: Need to implement this test (see https://github.com/microsoft/qdk-chemistry/issues/196)
    raise NotImplementedError("TODO: add factory_list.ipynb test.")


@pytest.mark.xfail(reason="Skipping unimplemented examples/state_prep_energy.ipynb test.")
def test_state_prep_energy():
    """Test the examples/state_prep_energy.ipynb script."""
    # TODO: Need to implement this test (see https://github.com/microsoft/qdk-chemistry/issues/196)
    raise NotImplementedError("TODO: add state_prep_energy.ipynb test.")


################################################################################
# Qiskit interoperability sample testing
################################################################################


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


################################################################################
# Pennylane interoperability sample testing
################################################################################


@pytest.mark.xfail(reason="Skipping unimplemented examples/pennylane/qpe_no_trotter.py test.")
def test_pennylane_qpe_no_trotter():
    """Test the examples/pennylane/qpe_no_trotter.py script."""
    # TODO: Need to implement this test (see https://github.com/microsoft/qdk-chemistry/issues/199)
    raise NotImplementedError("TODO: add pennylane/qpe_no_trotter.py test.")


################################################################################
# Qsharp interoperability sample testing
################################################################################


@pytest.mark.xfail(reason="Skipping unimplemented examples/qsharp/iqpe_no_trotter.qs test.")
def test_qsharp_iqpe_no_trotter():
    """Test the examples/qsharp/iqpe_no_trotter.qs script."""
    # TODO: Need to implement this test (see https://github.com/microsoft/qdk-chemistry/issues/200)
    raise NotImplementedError("TODO: add qsharp/iqpe_no_trotter.qs test.")
