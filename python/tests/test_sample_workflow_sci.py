"""End-to-end tests for the sparse-CI sample workflow.

The sparse-CI finder scenarios verify the MACIS-integrated pipeline.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    sci_energy_tolerance,
)
from .test_sample_workflow_utils import (
    _assert_warning_constraints,
    _collect_output_lines,
    _extract_sparse_ci_summary,
    _find_line,
    _run_workflow,
    _skip_for_mpi_failure,
)


def _truthy_env(name: str) -> bool:
    """Return True when ``name`` is set to a truthy value."""
    return os.getenv(name, "").lower() in {"1", "true", "yes"}


_RUNNING_IN_CI = _truthy_env("TF_BUILD")
_RUN_MACIS_WORKFLOW = _truthy_env("QDK_CHEMISTRY_RUN_MACIS_WORKFLOW")


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
