"""End-to-end tests for the MP2 sample workflow.

The MP2 sample workflow exercises the end-to-end pipeline for computing MP2 energies from a molecular geometry.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    mp2_energy_tolerance,
    scf_energy_tolerance,
)
from .test_sample_workflow_utils import (
    _assert_warning_constraints,
    _collect_output_lines,
    _find_line,
    _run_workflow,
    _skip_for_mpi_failure,
)


@dataclass(frozen=True)
class WorkflowCase:
    """Descriptor for a workflow CLI regression test."""

    identifier: str
    script: str
    args: list[str]
    cwd_relative: Path
    expected_scf_energy: float
    expected_reference_energy: float
    expected_mp2_energy: float
    expected_warning: str | None = None
    expect_no_warnings: bool = False


TEST_CASES: tuple[WorkflowCase, ...] = (
    WorkflowCase(
        identifier="active_space_overrides",
        script="examples/language/sample_mp2_reference_energy.py",
        args=[
            "--xyz",
            "examples/data/water.structure.xyz",
            "--num-active-electrons",
            "10",
            "--num-active-orbitals",
            "24",
        ],
        cwd_relative=Path("."),
        expected_scf_energy=-76.0228689456,
        expected_reference_energy=-76.0239418419,
        expected_mp2_energy=-76.2241719300,
    ),
    WorkflowCase(
        identifier="valence_defaults",
        script="examples/language/sample_mp2_reference_energy.py",
        args=[
            "--xyz",
            "examples/data/water.structure.xyz",
        ],
        cwd_relative=Path("."),
        expected_scf_energy=-76.0228689456,
        expected_reference_energy=-76.0239418419,
        expected_mp2_energy=-76.0276070141,
        expect_no_warnings=True,
    ),
)


@pytest.mark.parametrize("case", TEST_CASES, ids=lambda case: case.identifier)
def test_sample_mp2_workflow_scenarios(case: WorkflowCase) -> None:
    """Exercise the sample workflow under several CLI configurations."""
    repo_root = Path(__file__).resolve().parents[2]
    cwd = repo_root / case.cwd_relative
    cmd = [sys.executable, case.script, *case.args]

    result = _run_workflow(cmd, cwd)

    if result.returncode != 0:
        _skip_for_mpi_failure(result)
        pytest.fail(
            "sample_mp2_reference_energy.py exited with "
            f"{result.returncode}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    lines = _collect_output_lines(result)

    mp2_total_energy = float(_find_line(lambda line: "MP2 total energy = " in line, lines).split()[-2])
    scf_energy = float(_find_line(lambda line: "SCF energy = " in line, lines).split()[-2])
    reference_energy = float(_find_line(lambda line: "MP2 reference energy: " in line, lines).split()[-1])
    assert np.isclose(
        mp2_total_energy, case.expected_mp2_energy, rtol=float_comparison_relative_tolerance, atol=mp2_energy_tolerance
    )
    assert np.isclose(
        scf_energy, case.expected_scf_energy, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
    )
    assert np.isclose(
        reference_energy,
        case.expected_reference_energy,
        rtol=float_comparison_relative_tolerance,
        atol=scf_energy_tolerance,
    )

    _assert_warning_constraints(lines, case.expected_warning, case.expect_no_warnings)
